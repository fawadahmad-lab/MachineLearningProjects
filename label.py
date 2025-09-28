import os

# 👇 update this path to your labels folder (train/labels, valid/labels, test/labels)
labels_root = "home/sws/workFolder/cuda/soybean_pods.v1i.yolov8/merged_dataset"
nc = 1  # number of classes from your data.yaml

for subset in ["train", "valid", "test"]:
    labels_dir = os.path.join(labels_root, subset, "labels")
    if not os.path.exists(labels_dir):
        continue

    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            path = os.path.join(labels_dir, file)
            with open(path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, start=1):
                try:
                    cls = int(line.strip().split()[0])
                    if cls < 0 or cls >= nc:
                        print(f"⚠️ Invalid class {cls} in {path}, line {i}")
                except Exception as e:
                    print(f"❌ Error reading {path}, line {i}: {e}")
