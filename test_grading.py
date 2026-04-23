from __future__ import annotations

import json
from pathlib import Path

from inference import create_default_grader
from inference.report import build_report


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    samples_dir = repo_root / "samples"
    preview_dir = repo_root / "preview"

    normal_image = samples_dir / "normal-sample.jpg"
    ir_image = samples_dir / "ir-sample.jpg"

    grader = create_default_grader()
    result = grader.grade(normal_image, ir_image, preview_dir=preview_dir)
    payload, excel_path = build_report(result["report_summary"], result["enriched_grains"], stub_mode=False)
    result["report"] = {
        "payload": payload,
        "excel_path": excel_path,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
