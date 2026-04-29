# AI Training Codebase Structure

Last analyzed: 2026-04-29

This repository contains the rice-grain segmentation training notebooks and the local inference/reporting code used to grade rice samples from paired normal-light and IR images.

## Quick Orientation

- Training uses Ultralytics YOLO11 instance segmentation.
- There are two model tracks:
  - Normal/RGB rice images.
  - IR rice images, mainly used to identify chalky grains.
- Runtime inference loads two exported ONNX models from `models/`, merges their detections, computes physical features, applies partial PNS/BAFS 290:2025 grade rules, and writes previews plus Excel reports.
- Main local smoke-test entry point: `python test_grading.py`.

## Top-Level Layout

```text
ai-training/
|-- CODEBASE_STRUCTURE.md          # This orientation file.
|-- CODEX_RICE_TRAINING_SKILL.md   # Original Codex task/spec for creating training notebooks.
|-- README_training.md             # User-facing training guide.
|-- requirements_colab.txt         # Notebook/runtime dependencies.
|-- training_utils.py              # Shared helpers used by both notebooks.
|-- train_normal_yolo11_seg.ipynb  # Normal/RGB YOLO11 segmentation training notebook.
|-- train_ir_yolo11_seg.ipynb      # IR YOLO11 segmentation training notebook.
|-- prepare_dual_modality_yolo_dataset.ipynb # Splits, letterboxes, and augments datasets.
|-- test_grading.py                # Local inference/report smoke test.
|-- datasets/                      # YOLO segmentation datasets.
|-- inference/                     # Runtime grading package.
|-- models/                        # Exported ONNX models used by inference.
|-- samples/                       # Example normal/IR image pairs.
|-- preview/                       # Generated preview images and grading JSON.
|-- outputs/                       # Generated Excel reports.
`-- __pycache__/                   # Python cache; not source.
```

## Important Current Path Note

The current on-disk dataset folders are hyphenated:

```text
datasets/normal-image-datasets/
datasets/ir-image-datasets/
```

Some docs and notebook configuration cells still refer to underscore paths:

```text
datasets/normal_image_datasets/
datasets/ir_image_datasets/
```

Git status also shows many deleted underscore-path dataset files and new untracked hyphen-path dataset folders. Treat this as an active dataset rename/migration and do not revert it unless the user explicitly asks.

## Datasets

Current dataset counts on disk:

| Path | Files |
| --- | ---: |
| `datasets/normal-image-datasets/train/images` | 75 |
| `datasets/normal-image-datasets/train/labels` | 75 |
| `datasets/ir-image-datasets/train/images` | 76 |
| `datasets/ir-image-datasets/train/labels` | 76 |

Current dataset folders contain only `train/images` and `train/labels`. The older expected `valid/` and `test/` split folders are not present on disk right now.

Dataset YAML files:

- `datasets/normal-image-datasets/data.yaml`
  - `nc: 8`
  - names: `broken`, `chalky`, `damaged`, `discolored`, `foreign`, `paddy`, `red`, `whole`
- `datasets/ir-image-datasets/data.yaml`
  - `nc: 4`
  - names: `broken`, `chalky`, `foreign `, `whole`
  - Note the trailing space in `foreign `.

The YAML files currently use paths like `../train/images`, `../valid/images`, and `../test/images`; verify these before training because `valid/` and `test/` are missing in the current tree.

## Training Notebooks

### `prepare_dual_modality_yolo_dataset.ipynb`

Purpose: prepare raw/current IR and normal image folders for YOLOv11 training.

What it does:

- Scans `datasets/` and detects `ir` versus `normal` images from paths and filenames.
- Creates `datasets_prepared/yolo_dual_640/{normal,ir}/{train,val,test}/{images,labels}`.
- Uses a deterministic 70/20/10 split per modality.
- Applies EXIF auto-orientation and rewrites images without EXIF orientation metadata.
- Letterboxes all images to `640 x 640` with black padding, with no cropping or stretching.
- Transforms YOLO labels into the letterboxed coordinate system.
- Applies training-only strict augmentations: horizontal flip, vertical flip, 90 CW, 90 CCW, and 180 rotation.
- Supports both YOLO bounding-box rows and YOLO segmentation polygon rows.
- Writes per-modality `data.yaml`, `manifest.csv`, and `summary.csv`.

### `train_normal_yolo11_seg.ipynb`

Purpose: train a YOLO11 segmentation model for normal/RGB rice images.

Key configuration in the notebook:

- Dataset path currently set to `datasets/normal_image_datasets`.
- Expected classes: `broken`, `chalky`, `whole`, `damaged`, `discolored`, `foreign`, `paddy`, `red`.
- Default weights: `yolo11n-seg.pt`.
- Image size: `640`.
- Epochs: `100`.
- Batch: `16`.
- Run name: `normal_yolo11n_seg_640`.
- Training output root: `runs_segment/`.
- Notebook artifact root: `artifacts/normal_yolo11n_seg_640/`.

Notebook flow:

1. Mount/locate project in Colab.
2. Install requirements and bootstrap local imports.
3. Configure dataset/model/training parameters.
4. Validate YOLO segmentation dataset.
5. Visualize sample annotations.
6. Train YOLO11 segmentation.
7. Validate and test best weights.
8. Run inference demo.
9. Save prediction CSV, metrics JSON/CSV, paper figures, exported model, and artifact manifest.

### `train_ir_yolo11_seg.ipynb`

Purpose: train a YOLO11 segmentation model for IR rice images.

Key configuration in the notebook:

- Dataset path currently set to `datasets/ir_image_datasets`.
- Expected classes: `chalky`, `broken`, `whole`, `foreign`.
- Default weights: `yolo11n-seg.pt`.
- Image size: `640`.
- Epochs: `100`.
- Batch: `16`.
- Run name: `ir_yolo11n_seg_640`.
- Training output root: `runs_segment/`.
- Notebook artifact root: `artifacts/ir_yolo11n_seg_640/`.

The flow mirrors the normal notebook.

## Shared Training Helpers

File: `training_utils.py`

Responsibilities:

- Colab/project setup:
  - `is_colab`
  - `mount_google_drive`
  - `locate_project_root`
  - `ensure_project_on_path`
  - `install_requirements`
  - `bootstrap_notebook`
- Dataset preparation and validation:
  - `ensure_dataset_yaml`
  - `count_images_and_labels`
  - `validate_yolo_segmentation_dataset`
  - `plot_sample_annotations`
- Experiment helpers:
  - `seed_everything`
  - `collect_runtime_snapshot`
  - `extract_prediction_table`
  - `save_metrics_summary`
  - `find_best_weights_path`
  - `sigmoid`

Important assumption in validation: it checks `train`, `valid`, and `test` splits. If only `train` exists, warnings are expected.

## Runtime Inference Package

Package: `inference/`

### `inference/__init__.py`

Exports the public API:

- `RiceGrader`
- `create_default_grader`
- `build_payload`
- `build_report`
- `save_excel`

### `inference/inference.py`

Main orchestration for paired-image grading.

Key pieces:

- `RiceGrader`
  - Loads normal and IR YOLO segmentation models.
  - Runs predictions for normal and IR images.
  - Extracts detections with boxes, masks, physical dimensions, and size class.
  - Merges normal and IR detections.
  - Uses IR `chalky` detections to override matching normal detections.
  - Builds counts, area percentages, physical summaries, PNS size classification, grade summary, enriched per-grain output, and debug details.
  - Optionally saves preview images and JSON.
- `create_default_grader`
  - Loads:
    - `models/normal-image-model.onnx`
    - `models/ir-image-model.onnx`

Merge behavior:

- Normal detections labeled `chalky` are ignored.
- IR detections labeled `chalky` are matched against normal detections with labels in:
  - `whole`
  - `broken`
  - `damaged`
  - `discolored`
  - `red`
  - `paddy`
- Matching uses normalized bounding-box IoU and center distance.

### `inference/features.py`

Feature extraction and physical measurements.

Key pieces:

- `Detection` dataclass stores one detected object.
- Pixel scale:
  - `PX_PER_MM = 54.6539`
  - Basis: 23 mm reference coin measured at 1257.04 px diameter.
- `measure_axes_from_mask` estimates length and width from segmentation masks.
- `size_class_from_length_mm` maps rice length to PNS classes:
  - `extra_long`: `>= 7.5`
  - `long`: `6.4 to 7.4`
  - `medium`: `5.5 to 6.3`
  - `short`: `< 5.5`
- `summarize_physical_features`
- `summarize_pns_size_classification`
- `to_enriched_grains`

Physical feature summaries exclude `foreign` and `paddy`. PNS size classification currently uses only `whole` kernels.

### `inference/grader.py`

Partial PNS/BAFS 290:2025 grading logic.

Supported factors:

- `broken`
- `damaged`
- `discolored`
- `chalky`
- `red`
- `foreign`

Unsupported factors:

- `brewers`
- `immature`
- `contrasting_types`
- standard-compliant `paddy` counting

Important limitation: percentages are based on segmentation area shares as a proxy for percent by weight. The result is explicitly marked as a partial grade, not official-grade ready.

Other responsibilities:

- Class color map for preview drawing.
- Count summary.
- Area percentage summary.
- Paddy proxy as count per 1,000 detected grain-like objects.
- Report-ready summary conversion.

### `inference/report.py`

Report payload and Excel generation.

Key pieces:

- `build_payload` builds a compact JSON-like report payload.
- `save_excel` writes an Excel workbook to `outputs/report_YYYYMMDD_HHMMSS.xlsx`.
- `build_report` returns both payload and Excel path.

Excel workbook sheets:

- `Summary`
- `Per-Grain Detail`

## Local Inference Smoke Test

File: `test_grading.py`

What it does:

1. Uses sample pair:
   - `samples/normal-sample.jpg`
   - `samples/ir-sample.jpg`
2. Creates default grader from ONNX files in `models/`.
3. Runs paired-image grading.
4. Saves previews to `preview/`.
5. Builds Excel report in `outputs/`.
6. Prints the full JSON result.

Run:

```powershell
python test_grading.py
```

This requires the dependencies in `requirements_colab.txt`, including `ultralytics`, `opencv-python`, `numpy`, `pandas`, and `openpyxl`.

## Models, Samples, Outputs

`models/`

- `normal-image-model.onnx`
- `ir-image-model.onnx`

`samples/`

- `normal-sample.jpg`
- `normal-sample2.jpg`
- `ir-sample.jpg`
- `ir-sample2.jpg`

`preview/`

- Generated detection previews and JSON grading results.
- Examples include:
  - `normal-sample_normal_detected.jpg`
  - `ir-sample_ir_detected.jpg`
  - `normal-sample_merged_preview.jpg`
  - `normal-sample_grading_result.json`

`outputs/`

- Generated Excel reports named like `report_20260424_015712.xlsx`.

## Dependencies

Declared in `requirements_colab.txt`:

- `ultralytics>=8.3.0`
- `pyyaml>=6.0`
- `opencv-python>=4.8.0`
- `matplotlib>=3.7.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `openpyxl>=3.1.0`

Training helpers also import `torch` indirectly through Ultralytics/notebook setup.

## Common Next-Session Tasks

### Fix dataset path mismatch before training

Likely files to update:

- `README_training.md`
- `train_normal_yolo11_seg.ipynb`
- `train_ir_yolo11_seg.ipynb`
- `CODEX_RICE_TRAINING_SKILL.md`, if keeping the spec current matters.

Decision needed: use hyphenated current paths or restore underscore paths.

### Restore validation/test splits or adjust training

Current disk only has `train` splits. Before running notebooks, either:

- recreate `valid/` and `test/` folders with images and labels, or
- update YAML/notebooks to split train data or train without test evaluation.

### Update IR class spelling

The IR YAML has `foreign ` with a trailing space. If class labels are meant to match runtime grading labels exactly, trim it to `foreign` and confirm label IDs remain correct.

### Re-export trained models for runtime inference

After successful notebook training, copy or export ONNX outputs into:

- `models/normal-image-model.onnx`
- `models/ir-image-model.onnx`

`create_default_grader()` depends on exactly those paths.

## Git/Workspace Notes

At analysis time, the worktree had many deleted files under old underscore dataset directories and untracked files under new hyphenated dataset directories. Those appear unrelated to this documentation file and should be preserved unless the user asks for cleanup.
