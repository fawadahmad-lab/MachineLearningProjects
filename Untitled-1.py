# %%
import os
import yaml
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from ultralytics import YOLO
import cv2
import numpy as np

# %%
class SoybeanDetection:
    def __init__(self, data_path="soyabean_pods.v1i.yolov8", test_size=0.2, random_state=42):
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "train"
        self.test_size = test_size
        self.random_state = random_state
def setup_gpu(self):
    """Setup GPU configuration for optimal performance"""
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def check_data_yaml(self):
    """Check and read the existing data.yaml file"""
    data_yaml_path = self.data_path / "data.yaml"
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("Current data.yaml content:")
    print(yaml.dump(data_config, default_flow_style=False))
    
    return data_config, data_yaml_path
SoybeanDetection.setup_gpu = setup_gpu
SoybeanDetection.check_data_yaml = check_data_yaml

# %%

def split_dataset(self):
    """Split the dataset into train and test sets"""
    print("Splitting dataset into train and test...")
    
    # Create test directory structure
    test_path = self.data_path / "test"
    test_path.mkdir(exist_ok=True)
    (test_path / "images").mkdir(exist_ok=True)
    (test_path / "labels").mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list((self.train_path / "images").glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {self.train_path / 'images'}")
    
    print(f"Found {len(image_files)} images in train folder")
    
    # Split into train and test
    train_files, test_files = train_test_split(
        image_files, 
        test_size=self.test_size, 
        random_state=self.random_state
    )
    
    # Move test files
    for img_path in test_files:
        # Move image
        dest_img = test_path / "images" / img_path.name
        shutil.move(str(img_path), str(dest_img))
        
        # Move corresponding label
        label_path = self.train_path / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            dest_label = test_path / "labels" / f"{img_path.stem}.txt"
            shutil.move(str(label_path), str(dest_label))
        else:
            print(f"Warning: Label file not found for {img_path.name}")
    
    print(f"Dataset split completed!")
    print(f"Train images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")
    
    return len(train_files), len(test_files)

SoybeanDetection.split_dataset = split_dataset

# %%
def update_data_yaml(self):
    """Update data.yaml file with correct paths after splitting"""
    data_config, data_yaml_path = self.check_data_yaml()
    
    # Update paths to use absolute paths
    data_config.update({
        'train': str(self.train_path),
        'val': str(self.data_path / "test"),  # Using test as validation
        'test': str(self.data_path / "test"),
        'nc': data_config.get('nc', 1),  # Keep existing number of classes
        'names': data_config.get('names', ['soybean'])  # Keep existing class names
    })
    
    # Write updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Updated data.yaml:")
    print(yaml.dump(data_config, default_flow_style=False))
    return data_yaml_path

SoybeanDetection.update_data_yaml = update_data_yaml

# %%
def train_model(self, model_size='n', epochs=50, batch_size=16):
    """Train YOLOv8 model with GPU optimization"""
    print("Starting model training...")
    
    # Setup GPU
    device = self.setup_gpu()
    
    # Update data.yaml
    data_yaml_path = self.update_data_yaml()
    
    # Load pre-trained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Training parameters with GPU optimization
    training_params = {
        'data': str(data_yaml_path),
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'optimizer': 'auto',
        'lr0': 0.01,
        'cos_lr': True,
        'label_smoothing': 0.1,
        'cache': True,
        'amp': True,  # Automatic Mixed Precision
        'project': 'soybean_detection',
        'name': f'yolov8{model_size}_train',
        'verbose': True
    }
    
    # Start training
    results = model.train(**training_params)
    
    print("Training completed!")
    return model, results

SoybeanDetection.train_model = train_model

# %%
def evaluate_model(self, model_path='runs/detect/yolov8n_train/weights/best.pt'):
    """Evaluate trained model performance"""
    print("Evaluating model...")
    
    model = YOLO(model_path)
    data_yaml_path = self.data_path / "data.yaml"
    
    # Evaluate on test set
    metrics = model.val(
        data=str(data_yaml_path),
        split='test',
        device=0 if torch.cuda.is_available() else 'cpu',
        conf=0.25,
        iou=0.6,
        plots=True
    )
    
    print(f"Evaluation Results:")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    return metrics

SoybeanDetection.evaluate_model = evaluate_model

# %%
def detect_and_count(self, model_path=None, source_path=None, conf_threshold=0.25):
    """Detect and count soybeans in images"""
    print("Starting detection and counting...")
    
    # Default model path
    if model_path is None:
        model_path = 'runs/detect/yolov8n_train/weights/best.pt'
    
    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine source path
    if source_path is None:
        source_path = self.data_path / "test" / "images"
    
    # Run inference
    results = model(
        source=str(source_path),
        conf=conf_threshold,
        device=0 if torch.cuda.is_available() else 'cpu',
        imgsz=640,
        save=True,
        save_txt=True,
        project='soybean_detection',
        name='predictions'
    )
    
    # Count and display results
    total_count = 0
    image_counts = {}
    
    for result in results:
        if result.boxes is not None:
            count = len(result.boxes)
            total_count += count
            image_name = Path(result.path).name
            image_counts[image_name] = count
            print(f"Image: {image_name}, Soybeans detected: {count}")
    
    print(f"\nTotal soybeans counted: {total_count}")
    print(f"Average per image: {total_count / len(image_counts) if image_counts else 0:.2f}")
    
    return results, image_counts, total_count

SoybeanDetection.detect_and_count = detect_and_count

# %%
def visualize_results(self, image_path, model_path=None, save_dir='visualizations'):
    """Visualize detection results for a specific image"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if model_path is None:
        model_path = 'runs/detect/yolov8n_train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Run inference on single image
    results = model(
        image_path,
        conf=0.25,
        device=0 if torch.cuda.is_available() else 'cpu',
        imgsz=640
    )
    
    # Visualize
    plotted_image = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0
    
    # Add count text
    cv2.putText(plotted_image, f'Soybeans: {count}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    output_path = save_dir / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), plotted_image)
    
    print(f"Visualization saved to: {output_path}")
    return plotted_image, count

SoybeanDetection.visualize_results = visualize_results

# %%
soybean_detector = SoybeanDetection(data_path="soyabean_pods.v1i.yolov8", test_size=0.2)

# Check data.yaml first
data_config, data_yaml_path = soybean_detector.check_data_yaml()

# Setup GPU
soybean_detector.setup_gpu()

# %%
train_count, test_count = soybean_detector.split_dataset()
print(f"Split dataset: {train_count} train, {test_count} test images")

# %%
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)


# %%
model, results = soybean_detector.train_model(
    model_size='n',  # 'n', 's', 'm', 'l', 'x'
    epochs=50,       # Reduced for testing
    batch_size=8     # Reduced for testing
)

# %%
print("\n" + "="*50)
print("EVALUATING MODEL")
print("="*50)

# %%
print("\n" + "="*50)
print("DETECTING AND COUNTING")
print("="*50)

# %%


# %%
detection_results, image_counts, total_count = soybean_detector.detect_and_count(
    conf_threshold=0.25
)

# %%
test_images_path = soybean_detector.data_path / "test" / "images"
if test_images_path.exists():
    test_images = list(test_images_path.glob("*.*"))
    if test_images:
        sample_image = test_images[0]
        print(f"\nVisualizing sample image: {sample_image.name}")
        visualized_image, count = soybean_detector.visualize_results(
            str(sample_image),
            save_dir='visualizations'
        )

print("\n" + "="*50)
print("SOYBEAN DETECTION PIPELINE COMPLETED!")
print("="*50)

# %%
try:
    #execution here
    pass
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check your file paths and directory structure.")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()


