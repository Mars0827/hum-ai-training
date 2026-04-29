# Negative Background Images

Put background-only photos here. These images should show the same tray, lighting,
camera angle, and capture setup used for rice images, but with no rice grains and
no target objects.

Use these folders:

- `normal/images/` for normal-light/RGB background images
- `ir/images/` for IR background images

Do not annotate these images. The preparation notebook will automatically create
empty YOLO `.txt` label files for them, which teaches YOLO to suppress false
detections on background.

Suggested names:

- `normal_background_001.jpg`
- `ir_background_001.jpg`

