# Rice Segmentation Training Guide

This project includes two Jupyter notebooks for training Ultralytics YOLO11 instance segmentation models on rice-grain datasets:

- `train_normal_yolo11_seg.ipynb` for RGB or normal-light images
- `train_ir_yolo11_seg.ipynb` for IR images

Both notebooks use the shared helpers in `training_utils.py` and are written to be easy to run in VS Code with the Google Colab extension or in a regular local Jupyter environment.

## Expected folder structure

```text
ai-training/
|-- datasets/
|   |-- normal_image_datasets/
|   |   |-- train/
|   |   |-- valid/
|   |   |-- test/
|   |   `-- data.yaml
|   `-- ir_image_datasets/
|       |-- train/
|       |-- valid/
|       `-- test/
|-- training_utils.py
|-- train_normal_yolo11_seg.ipynb
`-- train_ir_yolo11_seg.ipynb
```

## How to run with VS Code and the Colab extension

1. Open the project folder in VS Code.
2. Open one of the notebooks.
3. Connect the notebook to a Google Colab runtime if you want free-tier GPU access.
4. Run the cells from top to bottom.

## Colab setup

The first code cell now handles the repetitive setup for you:

- it looks for the repo root automatically
- it mounts Google Drive automatically when running in Colab and the repo is not already visible
- it installs dependencies from `requirements_colab.txt`
- it prints a short runtime summary so you can confirm GPU detection quickly

For the smoothest Colab workflow, store the repo at:

```text
/content/drive/MyDrive/ai-training
```

If you keep the folder there, the notebook should mount Drive and find the project without extra edits.

## Variables you will likely change first

The main configuration block in each notebook includes:

- `MODEL_WEIGHTS`
- `IMAGE_SIZE`
- `EPOCHS`
- `BATCH`
- `PATIENCE`
- `RUN_NAME`

The recommended baseline is `yolo11n-seg.pt`. After you have a successful baseline run, you can try `yolo11s-seg.pt` with a single-line change.

## If CUDA runs out of memory

Start with `BATCH = 16` on Colab free tier. If you hit an out-of-memory error, lower it to:

- `8`
- or `4`

The notebooks call this out in comments near the configuration block.

## Saved outputs

Each notebook saves artifacts under a run folder and a notebook-specific analysis folder, including:

- best model weights path
- metrics summary JSON and CSV
- per-image prediction CSV
- inference preview images
- exported ONNX model when export succeeds
