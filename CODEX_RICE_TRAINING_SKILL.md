# Rice Quality Training Skill for Codex

## Goal
Create **two well-commented Jupyter notebooks** for training **Ultralytics YOLO11 instance segmentation** models in **VS Code with the Google Colab extension**.

The project has two datasets:

- **Normal dataset** at `datasets/normal_image_datasets`
- **IR dataset** at `datasets/ir_image_datasets`

Both datasets are already split into `train`, `valid`, and `test` in YOLO segmentation format.

The notebooks must be:
- easy to run in **Google Colab free tier**
- written clearly enough for a thesis/demo workflow
- reproducible
- well-commented
- safe for small datasets

---

## Important technical choices

### 1) Model choice
Use **YOLO11 segmentation** from Ultralytics.

Why this is a good fit:
- YOLO11 supports **instance segmentation**, which is useful because rice grains are individual objects and mask-level output is better than boxes for counting/separating grains.
- Ultralytics documents YOLO11 as supporting segmentation tasks, and publishes segmentation model variants such as `yolo11n-seg`, `yolo11s-seg`, etc. citeturn841489search0turn841489search3
- For a constrained environment like **Colab free** and future low-cost deployment, start with **`yolo11n-seg.pt`** as the default pretrained checkpoint. Ultralytics shows the nano segmentation model is the lightest YOLO11 segmentation option. citeturn841489search0

### 2) Dataset/task format
The datasets are already in a YOLO segmentation-compatible structure with `images/`, `labels/`, and a `data.yaml`, which matches Ultralytics’ supported segmentation workflow. citeturn841489search1turn841489search4

### 3) Input size
Use **`imgsz=640`** for first-pass training.

### 4) Small-dataset mindset
This project currently has a relatively small dataset, so the notebooks must:
- avoid unnecessarily large models
- use transfer learning from pretrained weights
- save the best model
- show validation and test metrics clearly
- include error analysis helpers

---

## Files to create
Create the following files in the project root:

1. `train_normal_yolo11_seg.ipynb`
2. `train_ir_yolo11_seg.ipynb`
3. `training_utils.py`
4. `requirements_colab.txt`
5. `README_training.md`

---

## Dataset assumptions
Assume this project structure:

```text
AI-TRAINING/
├── datasets/
│   ├── normal_image_datasets/
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   ├── data.yaml
│   │   ├── README.dataset.txt
│   │   └── README.roboflow.txt
│   └── ir_image_datasets/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── data.yaml
│       ├── README.dataset.txt
│       └── README.roboflow.txt
```

If `data.yaml` is missing in the IR dataset, the notebook must:
- detect that condition
- create a valid `data.yaml` automatically using the discovered class names and relative paths
- print the generated YAML

---

## Class definitions

### Normal classes
Use these classes for the normal notebook, matching the dataset YAML if already present:

- `broken`
- `chalky`
- `whole`
- `damaged`
- `discolored`
- `foreign`
- `paddy`
- `red`

### IR classes
Use these classes for the IR notebook, matching the dataset YAML if already present:

- `chalky`
- `broken`
- `whole`
- `foreign`

If the YAML already defines names, do not silently override them. Validate first, then warn if there is any mismatch.

---

## Notebook requirements
Both notebooks should follow the same structure and quality level.

### Section 1 — Project overview
Add a markdown introduction that explains:
- what the notebook trains
- the dataset path used
- the task type: **instance segmentation**
- why YOLO11 segmentation is being used
- why `yolo11n-seg.pt` is the default starting point

### Section 2 — Environment setup
Install only what is needed.

Include cells that:
- print Python version
- detect whether the runtime is Colab
- install:
  - `ultralytics`
  - `pyyaml`
  - `opencv-python`
  - `matplotlib`
  - `pandas`
  - `seaborn` only if truly needed for tables; otherwise prefer matplotlib

Do not assume a GPU is always present. Print whether CUDA is available.

### Section 3 — Imports and helper setup
Import:
- `os`
- `sys`
- `json`
- `math`
- `random`
- `shutil`
- `pathlib.Path`
- `yaml`
- `numpy`
- `pandas`
- `matplotlib.pyplot as plt`
- `cv2`
- `torch`
- `from ultralytics import YOLO`

Also import helpers from `training_utils.py`.

### Section 4 — Configuration block
Create a clear config cell with variables such as:

```python
PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / "datasets" / "normal_image_datasets"  # or ir_image_datasets
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_WEIGHTS = "yolo11n-seg.pt"
IMAGE_SIZE = 640
EPOCHS = 100
BATCH = 16
PATIENCE = 20
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 2
RUN_NAME = "normal_yolo11n_seg_640"
```

For Colab free tier, batch size should be adaptive:
- start with `16`
- if OOM occurs, comments should tell the user to reduce to `8` or `4`

### Section 5 — Dataset validation
Before training, validate:
- `train/images`, `train/labels`
- `valid/images`, `valid/labels`
- `test/images`, `test/labels`
- YAML existence and correctness
- class count and class names
- number of image files per split
- number of label files per split
- missing labels
- empty labels

Create a summary table and print warnings for:
- missing images
- missing labels
- class mismatch
- split mismatch

### Section 6 — Sample visualization
Show a few random training images with masks/labels overlaid.
Use clear plots.
This is mandatory because annotation mistakes are common and expensive.

### Section 7 — Training
Train using Ultralytics YOLO11 segmentation.
Use a cell similar to:

```python
model = YOLO(MODEL_WEIGHTS)
results = model.train(
    data=str(DATA_YAML),
    imgsz=IMAGE_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    patience=PATIENCE,
    device=DEVICE,
    workers=WORKERS,
    project="runs_segment",
    name=RUN_NAME,
    pretrained=True,
    verbose=True,
    seed=42,
    deterministic=True,
    exist_ok=True,
)
```

Prefer conservative defaults appropriate for a small dataset.

### Section 8 — Validation and testing
After training:
- validate the best model on the validation set
- evaluate on the test set
- print key metrics in a readable way
- save outputs in a structured folder

Include:
- mask mAP if available
- box mAP if available
- per-class metrics if accessible
- confusion-matrix-related outputs if supported by the run

### Section 9 — Inference demo
Run inference on:
- a few test images
- at least one manually selected image path if available

Show visual outputs inline.

### Section 10 — Confidence scoring
This section is important:

Each predicted grain should already expose a **native YOLO confidence score** from the model output. Instance segmentation predictions in Ultralytics include class labels and confidence scores for each detected instance. citeturn841489search1

Because of that, do **not** replace YOLO’s built-in confidence with an unnecessary sigmoid over already-thresholded confidences.

Instead, implement both:

#### A. Native instance confidence
For every predicted grain, extract and store:
- class id
- class name
- confidence
- bounding box
- mask polygon if available

#### B. Optional sigmoid-based calibrated score helper
Include a helper sigmoid function for **custom post-processing only**, clearly labeled as optional:

```python
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
```

Then add an example of a **custom quality score** using sigmoid on a user-defined signal, such as:
- a normalized count deviation
- a confidence aggregation statistic
- or another post-processing feature

But add a markdown explanation stating:
- YOLO already outputs per-instance confidence
- the sigmoid helper is optional for downstream scoring/calibration
- it should not be confused with the model’s internal confidence mechanics

Also create a dataframe of per-grain predictions with columns like:
- `image_name`
- `class_id`
- `class_name`
- `confidence`
- `confidence_percent`
- `mask_points_count`
- `x1`
- `y1`
- `x2`
- `y2`

### Section 11 — Export
Export the best model to:
- `.pt`
- `onnx`

Use a cell similar to:

```python
best_model = YOLO(best_model_path)
best_model.export(format="onnx")
```

Add comments about using ONNX later for backend inference.

### Section 12 — Save artifacts
Each notebook must save:
- model path
- metrics summary CSV/JSON
- per-image prediction CSV
- inference preview images

### Section 13 — Next steps
End with a markdown section explaining:
- how to improve performance
- importance of more data
- checking confusion between visually similar classes
- why deployment should start with the nano model

---

## `training_utils.py` requirements
Create a reusable utility module that both notebooks import.

It should contain well-commented helpers for:

1. `seed_everything(seed=42)`
2. `is_colab()`
3. `ensure_dataset_yaml(dataset_dir, yaml_path, class_names)`
4. `count_images_and_labels(split_dir)`
5. `validate_yolo_segmentation_dataset(dataset_dir, yaml_path)`
6. `plot_sample_annotations(dataset_dir, split="train", num_samples=4)`
7. `sigmoid(x)`
8. `extract_prediction_table(results, class_names)`
9. `save_metrics_summary(output_path, metrics_dict)`
10. `find_best_weights_path(project_dir, run_name)`

The utilities must be defensive and readable.

---

## `requirements_colab.txt`
Include a small requirements file like:

```txt
ultralytics>=8.3.0
pyyaml>=6.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
```

Keep it lean.

---

## `README_training.md`
Create a short guide that explains:
- what each notebook is for
- expected folder structure
- how to run in VS Code + Colab extension
- which variables to change first
- how to reduce batch size on CUDA OOM
- where outputs are saved

---

## Model recommendation rules
Default to:
- `yolo11n-seg.pt`

Allow an easy single-line change to:
- `yolo11s-seg.pt`

Add a markdown note saying:
- start with nano on Colab free
- try small only after a successful baseline

This matches Ultralytics’ published segmentation model lineup and keeps the first pass realistic for limited compute. citeturn841489search0

---

## Things to avoid
Do **not**:
- hardcode absolute Windows-only paths
- assume GPU always exists
- assume the IR dataset always has a valid YAML
- silently override class names from YAML
- use giant models by default
- create overly complex notebook abstractions that make debugging harder
- misuse sigmoid as if it is required to obtain YOLO confidence values

---

## Output quality expectations
The generated notebooks should be:
- runnable with minimal edits
- readable by a student team
- suitable for experimentation and thesis demonstrations
- strong enough to later adapt into backend inference

