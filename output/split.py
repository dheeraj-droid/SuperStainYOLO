import os
import shutil
import random

# Input paths for images and labels
images_dir = "filtered/images"  # Directory containing image files
labels_dir = "filtered/labels"  # Directory containing label files

# Output directories for splits
splits = {
    "train": 80,  # Percentage of data for training
    "val": 20,    # Percentage of data for validation
}
output_dirs = {split: f"{split}_data" for split in splits}
for split, path in output_dirs.items():
    os.makedirs(f"{path}/images", exist_ok=True)
    os.makedirs(f"{path}/labels", exist_ok=True)

# Gather all valid file pairs (image and corresponding label)
images = sorted(os.listdir(images_dir))
valid_pairs = [(img, img.replace(".png", ".txt")) for img in images if os.path.exists(os.path.join(labels_dir, img.replace(".png", ".txt")))]

# Shuffle the valid pairs
random.shuffle(valid_pairs)

# Compute split indices
total_count = len(valid_pairs)
val_count = int(total_count * (splits["val"] / 100))
train_count = total_count - val_count

# Split the data
splits_data = {
    "train": valid_pairs[:train_count],
    "val": valid_pairs[train_count:]
}

# Copy files to corresponding directories
for split, pairs in splits_data.items():
    for img, lbl in pairs:
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dirs[split], "images", img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(output_dirs[split], "labels", lbl))

print(f"Data successfully split into train ({train_count} files) and val ({val_count} files) datasets!")

