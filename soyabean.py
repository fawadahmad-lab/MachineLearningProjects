import os
import random
import shutil
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import cv2

# ==============================
# CONFIGURATION
# ==============================
DATA_PATH = "soyabean_pods.v1i.yolov8"   # dataset folder
MODEL_SIZE = "s"                         # choose from: n, s, m, l, x
EPOCHS = 200
BATCH_SIZE = 16
IMAGE_SIZE = 1024
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==============================
# GPU INFO
# ==============================
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Using CPU")

# ==============================
# CHECK DATA.YAML
# ==============================
data_yaml_path = Path(DATA_PATH) / "data.yaml"
if not data_yaml_path.exists():
    raise FileNotFoundError(f"data.yaml not found in {DATA_PATH}")

with open(data_yaml_path, "r") as f:
    data_config = yaml.safe_load(f)

print("Data.yaml content:")
print(yaml.dump(data_config, default_flow_style=False))

# ==============================
# VERIFY DATASET
# ==============================
def verify_dataset():
    image_dir = Path(DATA_PATH) / "train/images"
    label_dir = Path(DATA_PATH) / "train/labels"
    images = list(image_dir.glob("*.*"))
    missing_labels = []
    for img in images:
        label = label_dir / f"{img.stem}.txt"
        if not label.exists():
            missing_labels.append(img.name)
    print(f"Total images: {len(images)}")
    print(f"Missing labels: {len(missing_labels)}")
    if missing_labels:
        print("Examples:", missing_labels[:10])

verify_dataset()

# ==============================
# TRAINING
# ==============================
model = YOLO(f"yolov8{MODEL_SIZE}.pt")

results = model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=8,
    optimizer="AdamW",        # more stable than SGD
    lr0=0.001,                # smaller learning rate
    cos_lr=True,              # cosine decay
    label_smoothing=0.1,
    cache=True,               # cache images
    amp=True,                 # mixed precision
    project="soybean_detection",
    name=f"yolov8{MODEL_SIZE}_improved",
    verbose=True,
    patience=50,              # early stopping patience
    augment=True              # enable strong augmentations
)

print("Training completed!")

# ==============================
# EVALUATION
# ==============================
metrics = model.val(
    data=str(data_yaml_path),
    split="test",
    device=DEVICE,
    conf=0.25,
    iou=0.6,
    plots=True
)

print("Evaluation Results:")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")

# ==============================
# VISUALIZE ON TEST IMAGES
# ==============================
save_dir = Path("visualizations")
save_dir.mkdir(exist_ok=True)

test_images = list((Path(DATA_PATH) / "test/images").glob("*.*"))
if test_images:
    for img_path in random.sample(test_images, min(5, len(test_images))):
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            device=DEVICE,
            imgsz=IMAGE_SIZE
        )
        plotted = results[0].plot()
        cv2.imwrite(str(save_dir / f"detected_{img_path.name}"), plotted)
        print(f"Saved visualization: detected_{img_path.name}")
