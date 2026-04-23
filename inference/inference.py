from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from .features import (
    MM_PER_PX,
    Detection,
    measure_axes_from_mask,
    size_class_from_length_mm,
    summarize_physical_features,
    summarize_pns_size_classification,
    to_enriched_grains,
)
from .grader import (
    CLASS_COLORS,
    build_report_grader_result,
    grade_supported_factors,
    summarize_area_percentages,
    summarize_counts,
    summarize_paddy_proxy,
)

OVERRIDABLE_NORMAL_LABELS = {"whole", "broken", "damaged", "discolored", "red", "paddy"}


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _center_distance(center_a: tuple[float, float], center_b: tuple[float, float]) -> float:
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return (dx * dx + dy * dy) ** 0.5


class RiceGrader:
    def __init__(
        self,
        normal_model_path: str | Path,
        ir_model_path: str | Path,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        override_iou_threshold: float = 0.25,
        override_center_threshold: float = 0.03,
    ) -> None:
        self.normal_model_path = Path(normal_model_path)
        self.ir_model_path = Path(ir_model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.override_iou_threshold = override_iou_threshold
        self.override_center_threshold = override_center_threshold

        self.normal_model = YOLO(str(self.normal_model_path), task="segment")
        self.ir_model = YOLO(str(self.ir_model_path), task="segment")

    def grade(
        self,
        normal_image_path: str | Path,
        ir_image_path: str | Path,
        *,
        preview_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        normal_image_path = Path(normal_image_path)
        ir_image_path = Path(ir_image_path)

        normal_result = self.normal_model.predict(
            source=str(normal_image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]
        ir_result = self.ir_model.predict(
            source=str(ir_image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        normal_detections = self._extract_detections(normal_result, source="normal")
        ir_detections = self._extract_detections(ir_result, source="ir")
        merge = self._merge_detections(normal_detections, ir_detections)

        final_detections = merge["final_detections"]
        physical_features = summarize_physical_features(final_detections)
        pns_size_classification = summarize_pns_size_classification(final_detections)
        grade_summary = grade_supported_factors(final_detections)

        output = {
            "standard": "PNS/BAFS 290:2025",
            "product_type": "milled_rice_partial_grading",
            "basis_note": (
                "Percentages are estimated from segmentation area shares as a proxy for % by weight. "
                "The result is partial because some standard factors are not currently detectable."
            ),
            "inputs": {
                "normal_image": str(normal_image_path),
                "ir_image": str(ir_image_path),
                "normal_model": str(self.normal_model_path),
                "ir_model": str(self.ir_model_path),
            },
            "model_classes": {
                "normal": dict(normal_result.names),
                "ir": dict(ir_result.names),
            },
            "counts": summarize_counts(final_detections),
            "area_percentages": summarize_area_percentages(final_detections),
            "physical_features": physical_features,
            "pns_size_classification": pns_size_classification,
            "paddy_proxy": summarize_paddy_proxy(final_detections),
            "grade": grade_summary,
            "report_summary": build_report_grader_result(
                {
                    "grade": grade_summary,
                    "counts": summarize_counts(final_detections),
                    "physical_features": physical_features,
                    "pns_size_classification": pns_size_classification,
                }
            ),
            "enriched_grains": to_enriched_grains(final_detections),
            "merge_debug": {
                "normal_count": len(normal_detections),
                "ir_count": len(ir_detections),
                "normal_only_chalky_ignored": [d.as_dict() for d in merge["ignored_normal_chalky"]],
                "ir_chalky_matches": merge["override_matches"],
                "ir_chalky_without_normal_match": [d.as_dict() for d in merge["ir_chalky_unmatched"]],
            },
            "final_detections": [d.as_dict() for d in final_detections],
        }

        if preview_dir is not None:
            output["preview_paths"] = self._save_previews(
                preview_dir=Path(preview_dir),
                normal_image_path=normal_image_path,
                ir_image_path=ir_image_path,
                normal_result=normal_result,
                ir_result=ir_result,
                final_detections=final_detections,
                summary=output,
            )

        return output

    def _extract_detections(self, result: Any, *, source: str) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
        centers_xyn = result.boxes.xywhn.cpu().numpy()[:, :2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        names = result.names
        image_h, image_w = result.orig_shape

        for idx, class_id in enumerate(class_ids):
            area_px = int((masks[idx] > 0.5).sum()) if masks is not None else 0
            length_px, width_px = (
                measure_axes_from_mask(masks[idx], image_w=image_w, image_h=image_h) if masks is not None else (0.0, 0.0)
            )
            length_mm = length_px * MM_PER_PX
            width_mm = width_px * MM_PER_PX
            aspect_ratio = (length_px / width_px) if width_px > 0 else 0.0
            detections.append(
                Detection(
                    source=source,
                    label=str(names[int(class_id)]),
                    confidence=float(confidences[idx]),
                    bbox_xyxy=tuple(float(v) for v in boxes_xyxy[idx]),
                    bbox_norm_xyxy=tuple(float(v) for v in boxes_xyxyn[idx]),
                    center_norm_xy=tuple(float(v) for v in centers_xyn[idx]),
                    area_px=area_px,
                    length_px=length_px,
                    width_px=width_px,
                    length_mm=length_mm,
                    width_mm=width_mm,
                    aspect_ratio=aspect_ratio,
                    size_class=size_class_from_length_mm(length_mm) if length_mm > 0 else "unknown",
                )
            )

        return detections

    def _merge_detections(self, normal_detections: list[Detection], ir_detections: list[Detection]) -> dict[str, Any]:
        final_detections: list[Detection] = []
        ignored_normal_chalky: list[Detection] = []
        normal_pool: list[Detection] = []

        for det in normal_detections:
            if det.label == "chalky":
                ignored_normal_chalky.append(det)
            else:
                normal_pool.append(det)

        used_normal_indexes: set[int] = set()
        ir_chalky_detections = [det for det in ir_detections if det.label == "chalky"]
        override_matches: list[dict[str, Any]] = []
        ir_chalky_unmatched: list[Detection] = []

        for ir_det in ir_chalky_detections:
            best_index = None
            best_iou = -1.0
            best_distance = float("inf")

            for idx, normal_det in enumerate(normal_pool):
                if idx in used_normal_indexes or normal_det.label not in OVERRIDABLE_NORMAL_LABELS:
                    continue

                bbox_iou = _bbox_iou(ir_det.bbox_norm_xyxy, normal_det.bbox_norm_xyxy)
                center_distance = _center_distance(ir_det.center_norm_xy, normal_det.center_norm_xy)
                if bbox_iou > best_iou or (abs(bbox_iou - best_iou) < 1e-9 and center_distance < best_distance):
                    best_index = idx
                    best_iou = bbox_iou
                    best_distance = center_distance

            if best_index is not None and (
                best_iou >= self.override_iou_threshold or best_distance <= self.override_center_threshold
            ):
                used_normal_indexes.add(best_index)
                matched_normal = normal_pool[best_index]
                override_matches.append(
                    {
                        "normal_label_before_override": matched_normal.label,
                        "normal_confidence": round(matched_normal.confidence, 4),
                        "ir_label_after_override": ir_det.label,
                        "ir_confidence": round(ir_det.confidence, 4),
                        "bbox_iou": round(best_iou, 4),
                        "center_distance": round(best_distance, 4),
                    }
                )
            else:
                ir_chalky_unmatched.append(ir_det)

            final_detections.append(ir_det)

        for idx, normal_det in enumerate(normal_pool):
            if idx not in used_normal_indexes:
                final_detections.append(normal_det)

        final_detections.sort(key=lambda det: (det.center_norm_xy[1], det.center_norm_xy[0]))
        return {
            "final_detections": final_detections,
            "ignored_normal_chalky": ignored_normal_chalky,
            "override_matches": override_matches,
            "ir_chalky_unmatched": ir_chalky_unmatched,
        }

    def _save_previews(
        self,
        *,
        preview_dir: Path,
        normal_image_path: Path,
        ir_image_path: Path,
        normal_result: Any,
        ir_result: Any,
        final_detections: list[Detection],
        summary: dict[str, Any],
    ) -> dict[str, str]:
        preview_dir.mkdir(parents=True, exist_ok=True)

        normal_detected_path = preview_dir / f"{normal_image_path.stem}_normal_detected.jpg"
        ir_detected_path = preview_dir / f"{ir_image_path.stem}_ir_detected.jpg"
        merged_preview_path = preview_dir / f"{normal_image_path.stem}_merged_preview.jpg"
        json_path = preview_dir / f"{normal_image_path.stem}_grading_result.json"

        cv2.imwrite(str(normal_detected_path), normal_result.plot())
        cv2.imwrite(str(ir_detected_path), ir_result.plot())

        merged_canvas = cv2.imread(str(normal_image_path))
        if merged_canvas is None:
            raise FileNotFoundError(f"Could not read normal image: {normal_image_path}")

        image_h, image_w = merged_canvas.shape[:2]
        for det in final_detections:
            color = CLASS_COLORS.get(det.label, (255, 255, 255))
            x1 = int(det.bbox_norm_xyxy[0] * image_w)
            y1 = int(det.bbox_norm_xyxy[1] * image_h)
            x2 = int(det.bbox_norm_xyxy[2] * image_w)
            y2 = int(det.bbox_norm_xyxy[3] * image_h)
            cv2.rectangle(merged_canvas, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                merged_canvas,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(merged_preview_path), merged_canvas)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "normal_detected": str(normal_detected_path),
            "ir_detected": str(ir_detected_path),
            "merged_preview": str(merged_preview_path),
            "grading_result_json": str(json_path),
        }


def create_default_grader() -> RiceGrader:
    repo_root = Path(__file__).resolve().parent.parent
    return RiceGrader(
        normal_model_path=repo_root / "models" / "normal-image-model.onnx",
        ir_model_path=repo_root / "models" / "ir-image-model.onnx",
    )
