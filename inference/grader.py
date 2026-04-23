from __future__ import annotations

from collections import Counter
from typing import Any

from .features import Detection

GRADE_ORDER = [
    "Premium",
    "Grade no. 1",
    "Grade no. 2",
    "Grade no. 3",
    "Grade no. 4",
    "Grade no. 5",
]

GRADE_THRESHOLDS = {
    "Premium": {
        "broken": 5.0,
        "damaged": 0.5,
        "discolored": 0.5,
        "chalky": 4.0,
        "red": 1.0,
        "foreign": 0.025,
    },
    "Grade no. 1": {
        "broken": 10.0,
        "damaged": 0.7,
        "discolored": 0.7,
        "chalky": 5.0,
        "red": 2.0,
        "foreign": 0.10,
    },
    "Grade no. 2": {
        "broken": 15.0,
        "damaged": 1.0,
        "discolored": 1.0,
        "chalky": 7.0,
        "red": 4.0,
        "foreign": 0.15,
    },
    "Grade no. 3": {
        "broken": 25.0,
        "damaged": 1.5,
        "discolored": 3.0,
        "chalky": 9.0,
        "red": 5.0,
        "foreign": 0.17,
    },
    "Grade no. 4": {
        "broken": 35.0,
        "damaged": 2.0,
        "discolored": 5.0,
        "chalky": 12.0,
        "red": 6.0,
        "foreign": 0.20,
    },
    "Grade no. 5": {
        "broken": 45.0,
        "damaged": 3.0,
        "discolored": 8.0,
        "chalky": 15.0,
        "red": 7.0,
        "foreign": 0.25,
    },
}
PARAMETER_ORDER = ("broken", "damaged", "discolored", "chalky", "red", "foreign")
UNSUPPORTED_GRADE_FACTORS = ("brewers", "immature", "contrasting_types", "paddy")
CLASS_COLORS = {
    "broken": (54, 67, 244),
    "chalky": (0, 170, 255),
    "damaged": (0, 102, 204),
    "discolored": (30, 150, 90),
    "foreign": (128, 128, 128),
    "paddy": (19, 69, 139),
    "red": (220, 60, 60),
    "whole": (60, 179, 113),
}


def summarize_counts(detections: list[Detection]) -> dict[str, int]:
    counts = Counter(det.label for det in detections)
    return {label: int(counts.get(label, 0)) for label in sorted(set(counts) | set(CLASS_COLORS))}


def summarize_area_percentages(detections: list[Detection]) -> dict[str, float]:
    total_area = sum(det.area_px for det in detections)
    if total_area <= 0:
        return {label: 0.0 for label in sorted(CLASS_COLORS)}

    area_by_label = Counter()
    for det in detections:
        area_by_label[det.label] += det.area_px

    return {
        label: round((area_by_label.get(label, 0) / total_area) * 100.0, 4)
        for label in sorted(set(area_by_label) | set(CLASS_COLORS))
    }


def summarize_paddy_proxy(detections: list[Detection]) -> dict[str, Any]:
    grain_like_count = sum(1 for det in detections if det.label != "foreign")
    paddy_count = sum(1 for det in detections if det.label == "paddy")
    paddy_per_1000_detected_objects = (paddy_count / grain_like_count) * 1000.0 if grain_like_count else 0.0
    return {
        "paddy_count": paddy_count,
        "grain_like_count": grain_like_count,
        "paddy_per_1000_detected_objects": round(paddy_per_1000_detected_objects, 4),
        "used_for_grade": False,
        "note": "The standard defines paddy as count per 1,000 g, so this image-based proxy is reported but not graded.",
    }


def grade_supported_factors(detections: list[Detection]) -> dict[str, Any]:
    area_percentages = summarize_area_percentages(detections)
    factor_results: dict[str, Any] = {}
    limiting_grade_index = 0
    failed_off_grade = False

    for factor in PARAMETER_ORDER:
        observed = area_percentages.get(factor, 0.0)
        passing_grades = [grade_name for grade_name in GRADE_ORDER if observed <= GRADE_THRESHOLDS[grade_name][factor]]
        assigned_grade = passing_grades[0] if passing_grades else "Off-Grade"
        factor_results[factor] = {
            "observed_percent": round(observed, 4),
            "assigned_grade": assigned_grade,
            "thresholds": {grade_name: GRADE_THRESHOLDS[grade_name][factor] for grade_name in GRADE_ORDER},
        }

        if assigned_grade == "Off-Grade":
            failed_off_grade = True
        else:
            limiting_grade_index = max(limiting_grade_index, GRADE_ORDER.index(assigned_grade))

    overall_grade = "Off-Grade" if failed_off_grade else GRADE_ORDER[limiting_grade_index]
    limiting_factors = [
        factor
        for factor, result in factor_results.items()
        if result["assigned_grade"] == overall_grade or (overall_grade == "Off-Grade" and result["assigned_grade"] == "Off-Grade")
    ]

    return {
        "grade": overall_grade,
        "grade_scope": "partial_supported_factors_only",
        "supported_factors": list(PARAMETER_ORDER),
        "unsupported_factors": list(UNSUPPORTED_GRADE_FACTORS),
        "factor_results": factor_results,
        "limiting_factors": limiting_factors,
        "official_grade_ready": False,
        "note": (
            "This is a partial grade based only on currently detectable factors. "
            "Brewers, immature kernels, contrasting types, and standard-compliant paddy counting are not yet included."
        ),
    }


def build_report_grader_result(summary: dict[str, Any]) -> dict[str, Any]:
    grade_summary = summary["grade"]
    pns_size = summary["pns_size_classification"]
    grain_total = sum(int(count) for count in summary["counts"].values())
    parameters = {
        factor: float(result["observed_percent"])
        for factor, result in grade_summary["factor_results"].items()
    }
    limiting_factor = grade_summary["limiting_factors"][0] if grade_summary["limiting_factors"] else None
    return {
        "grade": grade_summary["grade"],
        "limiting_factor": limiting_factor,
        "grain_size_class": pns_size["assigned_class"],
        "total_grains": grain_total,
        "parameters": parameters,
    }
