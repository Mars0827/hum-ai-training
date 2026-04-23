from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

PX_PER_MM = 54.6539  # Derived from a 23 mm coin measured at 1257.04 px diameter.
MM_PER_PX = 1.0 / PX_PER_MM
PHYSICAL_FEATURE_EXCLUDED_LABELS = {"foreign", "paddy"}
PNS_SIZE_REFERENCE_LABELS = {"whole"}
PNS_SIZE_THRESHOLDS_MM = {
    "extra_long": ">= 7.5",
    "long": "6.4 to 7.4",
    "medium": "5.5 to 6.3",
    "short": "< 5.5",
}


@dataclass
class Detection:
    source: str
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_norm_xyxy: tuple[float, float, float, float]
    center_norm_xy: tuple[float, float]
    area_px: int
    length_px: float = 0.0
    width_px: float = 0.0
    length_mm: float = 0.0
    width_mm: float = 0.0
    aspect_ratio: float = 0.0
    size_class: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "bbox_norm_xyxy": [round(v, 6) for v in self.bbox_norm_xyxy],
            "center_norm_xy": [round(v, 6) for v in self.center_norm_xy],
            "area_px": int(self.area_px),
            "physical_features": {
                "length_px": round(self.length_px, 2),
                "width_px": round(self.width_px, 2),
                "length_mm": round(self.length_mm, 4),
                "width_mm": round(self.width_mm, 4),
                "aspect_ratio": round(self.aspect_ratio, 4),
                "size_class": self.size_class,
                "included_in_physical_feature_summary": self.label not in PHYSICAL_FEATURE_EXCLUDED_LABELS,
                "included_in_pns_size_classification": self.label in PNS_SIZE_REFERENCE_LABELS,
            },
        }


def size_class_from_length_mm(length_mm: float) -> str:
    if length_mm >= 7.5:
        return "extra_long"
    if 6.4 <= length_mm <= 7.4:
        return "long"
    if 5.5 <= length_mm <= 6.3:
        return "medium"
    return "short"


def measure_axes_from_mask(mask: np.ndarray, *, image_w: int, image_h: int) -> tuple[float, float]:
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)
    scale_x = float(image_w) / float(mask.shape[1])
    scale_y = float(image_h) / float(mask.shape[0])
    contour = contour.astype(np.float32)
    contour[:, 0, 0] *= scale_x
    contour[:, 0, 1] *= scale_y

    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        return float(max(w, h)), float(min(w, h))

    (_, _), (axis_a, axis_b), _ = cv2.minAreaRect(contour)
    return float(max(axis_a, axis_b)), float(min(axis_a, axis_b))


def summarize_physical_features(detections: list[Detection]) -> dict[str, Any]:
    included = [det for det in detections if det.label not in PHYSICAL_FEATURE_EXCLUDED_LABELS]
    grouped: dict[str, list[Detection]] = {}
    for det in included:
        grouped.setdefault(det.label, []).append(det)

    by_label: dict[str, Any] = {}
    for label, label_detections in sorted(grouped.items()):
        lengths_mm = np.array([det.length_mm for det in label_detections], dtype=float)
        widths_mm = np.array([det.width_mm for det in label_detections], dtype=float)
        aspect_ratios = np.array([det.aspect_ratio for det in label_detections], dtype=float)
        size_counts = Counter(det.size_class for det in label_detections)
        by_label[label] = {
            "count": len(label_detections),
            "mean_length_mm": round(float(lengths_mm.mean()), 4),
            "mean_width_mm": round(float(widths_mm.mean()), 4),
            "mean_aspect_ratio": round(float(aspect_ratios.mean()), 4),
            "size_class_counts": dict(sorted(size_counts.items())),
        }

    return {
        "pixel_scale": {
            "px_per_mm": round(PX_PER_MM, 4),
            "mm_per_px": round(MM_PER_PX, 6),
            "basis": "23 mm reference coin measured at 1257.04 px diameter",
        },
        "included_labels": sorted(grouped.keys()),
        "excluded_labels": sorted(PHYSICAL_FEATURE_EXCLUDED_LABELS),
        "grain_count": len(included),
        "by_label": by_label,
    }


def summarize_pns_size_classification(detections: list[Detection]) -> dict[str, Any]:
    reference_detections = [det for det in detections if det.label in PNS_SIZE_REFERENCE_LABELS]
    size_counts = Counter(det.size_class for det in reference_detections if det.size_class != "unknown")
    total_reference = sum(size_counts.values())
    reference_lengths_mm = [det.length_mm for det in reference_detections if det.length_mm > 0]

    size_percentages = {
        size_class: round((count / total_reference) * 100.0, 4)
        for size_class, count in sorted(size_counts.items())
    } if total_reference else {}

    assigned_class = "unclassified"
    for size_class in ("extra_long", "long", "medium", "short"):
        if size_percentages.get(size_class, 0.0) >= 80.0:
            assigned_class = size_class
            break

    calibration_warning = None
    if reference_lengths_mm:
        mean_reference_length_mm = float(np.mean(reference_lengths_mm))
        if mean_reference_length_mm < 3.0 or mean_reference_length_mm > 12.0:
            calibration_warning = (
                "Measured whole-kernel lengths fall outside a typical milled-rice range. "
                "Verify that PX_PER_MM matches the current image scale before relying on the PNS size class."
            )

    return {
        "standard": "PNS/BAFS 290:2025",
        "basis": "whole milled rice kernels only",
        "criterion": "80% or more of the whole milled rice kernels must fall in one length class",
        "reference_labels": sorted(PNS_SIZE_REFERENCE_LABELS),
        "excluded_labels": sorted(PHYSICAL_FEATURE_EXCLUDED_LABELS),
        "reference_kernel_count": total_reference,
        "assigned_class": assigned_class,
        "mean_reference_length_mm": round(float(np.mean(reference_lengths_mm)), 4) if reference_lengths_mm else None,
        "calibration_warning": calibration_warning,
        "size_class_counts": dict(sorted(size_counts.items())),
        "size_class_percentages": size_percentages,
        "thresholds_mm": PNS_SIZE_THRESHOLDS_MM,
    }


def to_enriched_grains(detections: list[Detection]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for grain_id, det in enumerate(detections, start=1):
        enriched.append(
            {
                "grain_id": grain_id,
                "class_label": det.label,
                "length_mm": det.length_mm,
                "width_mm": det.width_mm,
                "area_px": float(det.area_px),
                "grain_size_class": det.size_class,
                "ir_mean_intensity": 0.0,
                "confidence": det.confidence,
                "source": det.source,
            }
        )
    return enriched
