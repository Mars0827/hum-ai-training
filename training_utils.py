"""Shared helpers for YOLO11 rice-grain segmentation training notebooks."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_REQUIREMENTS = (
    "ultralytics>=8.3.0",
    "pyyaml>=6.0",
    "opencv-python>=4.8.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
)


def seed_everything(seed: int = 42) -> None:
    """Seed common random number generators for more reproducible experiments."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def is_colab() -> bool:
    """Return True when the notebook is running inside Google Colab."""
    return "google.colab" in sys.modules


def mount_google_drive(mount_point: str = "/content/drive", force_remount: bool = False) -> Path:
    """Mount Google Drive when running in Colab and return the MyDrive path."""
    drive_root = Path(mount_point)
    mydrive_root = drive_root / "MyDrive"

    if not is_colab():
        return mydrive_root

    if mydrive_root.exists() and not force_remount:
        return mydrive_root

    from google.colab import drive

    drive.mount(mount_point, force_remount=force_remount)
    return mydrive_root


def _iter_search_roots(start: Path | None = None, project_hint: str | None = None) -> list[Path]:
    start = (start or Path.cwd()).resolve()
    roots: list[Path] = []

    def add(path: Path) -> None:
        if path.exists() and path not in roots:
            roots.append(path)

    add(start)
    for parent in start.parents:
        add(parent)

    add(Path("/content"))
    add(Path("/content") / (project_hint or ""))
    add(Path("/content/drive/MyDrive"))
    if project_hint:
        add(Path("/content/drive/MyDrive") / project_hint)

    return roots


def locate_project_root(
    start: str | Path | None = None,
    dataset_name: str | None = None,
    project_hint: str | None = None,
) -> Path | None:
    """Locate the project root by looking for training_utils.py and the target dataset."""
    roots = _iter_search_roots(Path(start).resolve() if start is not None else None, project_hint=project_hint)

    def matches(root: Path) -> bool:
        if not (root / "training_utils.py").exists():
            return False
        if dataset_name is None:
            return True
        return (root / "datasets" / dataset_name).exists()

    for root in roots:
        if matches(root):
            return root

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        nested_candidate = root / (project_hint or "")
        if project_hint and matches(nested_candidate):
            return nested_candidate

        for nested in root.glob("*/training_utils.py"):
            candidate = nested.parent
            if matches(candidate):
                return candidate

    # Fall back to a recursive search for cases where the repo lives deeper in Drive.
    recursive_roots = [
        path
        for path in (Path("/content"), Path("/content/drive/MyDrive"))
        if path.exists() and path.is_dir()
    ]
    for root in recursive_roots:
        try:
            for nested in root.rglob("training_utils.py"):
                candidate = nested.parent
                if matches(candidate):
                    return candidate
        except OSError:
            continue

    return None


def ensure_project_on_path(project_root: str | Path) -> Path:
    """Add the project root to sys.path so local notebook imports work reliably."""
    project_root = Path(project_root).resolve()
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def install_requirements(project_root: str | Path, quiet: bool = True) -> Path | None:
    """Install notebook dependencies from requirements_colab.txt when available."""
    project_root = Path(project_root)
    requirements_path = project_root / "requirements_colab.txt"
    pip_args = [sys.executable, "-m", "pip", "install"]

    if quiet:
        pip_args.append("-q")

    if requirements_path.exists():
        subprocess.check_call([*pip_args, "-r", str(requirements_path)])
        return requirements_path

    subprocess.check_call([*pip_args, *DEFAULT_REQUIREMENTS])
    return None


def collect_runtime_snapshot() -> dict[str, Any]:
    """Return a small runtime summary for notebook diagnostics."""
    snapshot: dict[str, Any] = {
        "python_version": sys.version,
        "working_directory": str(Path.cwd()),
        "running_in_colab": is_colab(),
    }

    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        snapshot["cuda_available"] = False
        snapshot["gpu_name"] = None

    return snapshot


def bootstrap_notebook(
    dataset_name: str,
    project_hint: str = "ai-training",
    auto_mount_drive: bool = True,
    quiet_pip: bool = True,
) -> dict[str, Any]:
    """Resolve the project root, optionally mount Drive, install requirements, and summarize the runtime."""
    project_root = locate_project_root(dataset_name=dataset_name, project_hint=project_hint)
    mounted_drive = False

    if project_root is None and auto_mount_drive and is_colab():
        mount_google_drive()
        mounted_drive = True
        project_root = locate_project_root(dataset_name=dataset_name, project_hint=project_hint)

    if project_root is None:
        raise FileNotFoundError(
            "Could not locate the project root. In Colab, place this folder in "
            f"'/content/drive/MyDrive/{project_hint}' or open the notebook from the repo root."
        )

    requirements_path = install_requirements(project_root, quiet=quiet_pip)
    ensure_project_on_path(project_root)

    snapshot = collect_runtime_snapshot()
    snapshot.update(
        {
            "project_root": project_root.resolve(),
            "requirements_path": requirements_path.resolve() if requirements_path else None,
            "mounted_drive": mounted_drive,
        }
    )
    return snapshot


def ensure_dataset_yaml(
    dataset_dir: str | Path,
    yaml_path: str | Path,
    class_names: list[str],
) -> Path:
    """Create or repair a dataset YAML file with stable paths for local and Colab use."""
    import yaml

    dataset_dir = Path(dataset_dir)
    yaml_path = Path(yaml_path)
    existing_payload = {}
    if yaml_path.exists():
        existing_payload = _load_yaml(yaml_path)

    val_split = "val" if (dataset_dir / "val").exists() or not (dataset_dir / "valid").exists() else "valid"

    payload = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": f"{val_split}/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    extra_items = {
        key: value
        for key, value in existing_payload.items()
        if key not in {"path", "train", "val", "test", "nc", "names"}
    }
    payload.update(extra_items)

    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def count_images_and_labels(split_dir: str | Path) -> dict[str, Any]:
    """Count images and YOLO label files for one split."""
    split_dir = Path(split_dir)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    image_files = _list_images(images_dir)
    label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    empty_labels = []
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            empty_labels.append(label_file.name)

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "empty_labels": empty_labels,
        "image_files": image_files,
        "label_files": label_files,
    }


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    import yaml

    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def validate_yolo_segmentation_dataset(
    dataset_dir: str | Path,
    yaml_path: str | Path,
) -> dict[str, Any]:
    """Validate YOLO segmentation folder structure and summarize issues."""
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    yaml_path = Path(yaml_path)
    yaml_data = _load_yaml(yaml_path)

    split_summaries = {}
    warnings: list[str] = []
    observed_class_ids: set[int] = set()

    split_names = ["train"]
    if (dataset_dir / "val").exists():
        split_names.append("val")
    elif (dataset_dir / "valid").exists():
        split_names.append("valid")
    else:
        split_names.append("val")
    split_names.append("test")

    for split in split_names:
        split_dir = dataset_dir / split
        summary = count_images_and_labels(split_dir)
        split_summaries[split] = summary

        if not (split_dir / "images").exists():
            warnings.append(f"Missing directory: {split_dir / 'images'}")
        if not (split_dir / "labels").exists():
            warnings.append(f"Missing directory: {split_dir / 'labels'}")
        if summary["missing_labels"]:
            warnings.append(f"{split}: {len(summary['missing_labels'])} image(s) are missing label files.")
        if summary["missing_images"]:
            warnings.append(f"{split}: {len(summary['missing_images'])} label file(s) are missing images.")
        if summary["empty_labels"]:
            warnings.append(f"{split}: {len(summary['empty_labels'])} empty label file(s) detected.")

        for label_path in summary["label_files"]:
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            for line_index, line in enumerate(lines, start=1):
                pieces = line.strip().split()
                if not pieces:
                    continue

                try:
                    class_id = int(float(pieces[0]))
                except ValueError:
                    warnings.append(f"{label_path} line {line_index}: invalid class id '{pieces[0]}'.")
                    continue

                if len(pieces) < 7 or len(pieces[1:]) % 2 != 0:
                    warnings.append(
                        f"{label_path} line {line_index}: segmentation row should contain class id plus polygon pairs."
                    )

                observed_class_ids.add(class_id)

    yaml_names = yaml_data.get("names", [])
    yaml_nc = yaml_data.get("nc")
    if yaml_data:
        if not isinstance(yaml_names, list):
            warnings.append("Dataset YAML 'names' field is not a list.")
            yaml_names = []
        if yaml_nc is not None and isinstance(yaml_nc, int) and yaml_names and yaml_nc != len(yaml_names):
            warnings.append(
                f"Dataset YAML mismatch: nc={yaml_nc} but len(names)={len(yaml_names)}."
            )

    if yaml_names:
        expected_ids = set(range(len(yaml_names)))
        if observed_class_ids and observed_class_ids != expected_ids.intersection(observed_class_ids):
            out_of_range = sorted(class_id for class_id in observed_class_ids if class_id >= len(yaml_names))
            if out_of_range:
                warnings.append(
                    f"Observed class ids outside YAML class-name range: {out_of_range}."
                )

    summary_rows = []
    for split, summary in split_summaries.items():
        summary_rows.append(
            {
                "split": split,
                "images": summary["images"],
                "labels": summary["labels"],
                "missing_labels": len(summary["missing_labels"]),
                "missing_images": len(summary["missing_images"]),
                "empty_labels": len(summary["empty_labels"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return {
        "dataset_dir": dataset_dir,
        "yaml_path": yaml_path,
        "yaml_data": yaml_data,
        "class_names": yaml_names,
        "observed_class_ids": sorted(observed_class_ids),
        "summary_df": summary_df,
        "split_summaries": split_summaries,
        "warnings": warnings,
    }


def _load_polygon_from_row(parts: list[str], image_width: int, image_height: int) -> np.ndarray:
    import numpy as np

    coords = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= image_width
    coords[:, 1] *= image_height
    return coords.astype(np.int32)


def plot_sample_annotations(
    dataset_dir: str | Path,
    split: str = "train",
    num_samples: int = 4,
) -> list[Path]:
    """Render a few annotated examples from a YOLO segmentation dataset."""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    image_files = _list_images(images_dir)
    if not image_files:
        print(f"No images found in {images_dir}")
        return []

    sample_count = min(num_samples, len(image_files))
    selected = random.sample(image_files, sample_count)

    figure, axes = plt.subplots(1, sample_count, figsize=(5 * sample_count, 5))
    if sample_count == 1:
        axes = [axes]

    for axis, image_path in zip(axes, selected):
        image = cv2.imread(str(image_path))
        if image is None:
            axis.set_title(f"Failed to load\n{image_path.name}")
            axis.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image.copy()
        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                polygon = _load_polygon_from_row(parts, image.shape[1], image.shape[0])
                color = tuple(int(value) for value in np.random.randint(40, 255, size=3))
                cv2.fillPoly(overlay, [polygon], color)
                cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)

        blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
        axis.imshow(blended)
        axis.set_title(image_path.name, fontsize=9)
        axis.axis("off")

    plt.tight_layout()
    return selected


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Simple sigmoid helper for optional downstream calibration."""
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def extract_prediction_table(results: Any, class_names: list[str]) -> pd.DataFrame:
    """Convert Ultralytics inference results into a per-instance table."""
    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []

    for result in results:
        image_name = Path(getattr(result, "path", "unknown")).name
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        if boxes is None or boxes.cls is None:
            continue

        cls_values = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        conf_values = boxes.conf.detach().cpu().numpy().tolist() if boxes.conf is not None else []
        xyxy_values = boxes.xyxy.detach().cpu().numpy().tolist() if boxes.xyxy is not None else []

        polygons = []
        if masks is not None and getattr(masks, "xy", None) is not None:
            polygons = masks.xy

        for index, class_id in enumerate(cls_values):
            confidence = float(conf_values[index]) if index < len(conf_values) else float("nan")
            bbox = xyxy_values[index] if index < len(xyxy_values) else [np.nan, np.nan, np.nan, np.nan]
            polygon = polygons[index] if index < len(polygons) else None

            rows.append(
                {
                    "image_name": image_name,
                    "class_id": class_id,
                    "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}",
                    "confidence": confidence,
                    "confidence_percent": confidence * 100.0 if np.isfinite(confidence) else np.nan,
                    "mask_points_count": int(len(polygon)) if polygon is not None else 0,
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "mask_polygon": json.dumps(np.asarray(polygon).tolist()) if polygon is not None else "",
                }
            )

    return pd.DataFrame(rows)


def save_metrics_summary(output_path: str | Path, metrics_dict: dict[str, Any]) -> Path:
    """Persist metrics to JSON and a flat CSV companion file."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.with_suffix(".json")
    csv_path = output_path.with_suffix(".csv")

    json_path.write_text(json.dumps(metrics_dict, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([metrics_dict]).to_csv(csv_path, index=False)
    return json_path


def find_best_weights_path(project_dir: str | Path, run_name: str) -> Path:
    """Locate the saved best-model weights for a YOLO run."""
    project_dir = Path(project_dir)
    candidate = project_dir / run_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate

    recursive_matches = sorted(project_dir.glob(f"**/{run_name}/weights/best.pt"))
    if recursive_matches:
        return recursive_matches[0]

    raise FileNotFoundError(f"Could not find best.pt under {project_dir} for run '{run_name}'.")
