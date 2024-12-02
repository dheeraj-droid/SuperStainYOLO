import nibabel as nib
import numpy as np
import cv2
import os
from shapely.geometry import Polygon
import shutil

# Paths
input_dir = "datasets"  # Directory containing BraTS2021_XXXXX folder
output_images_dir = "output/images"
output_labels_dir = "output/labels"
filtered_images_dir = "output/filtered/images"
filtered_labels_dir = "output/filtered/labels"

# Create necessary directories
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs(filtered_images_dir, exist_ok=True)
os.makedirs(filtered_labels_dir, exist_ok=True)

# Global counter for file naming
counter = 0

def normalize_image(image):
    """Normalize MRI image intensity to range [0, 255]."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Min-max normalization
    return (image * 255).astype(np.uint8)

def save_slices_and_labels(data_path, seg_path, classes):
    global counter
    modality_data = nib.load(data_path).get_fdata()
    seg_data = nib.load(seg_path).get_fdata()

    for i in range(modality_data.shape[-1]):
        # Extract and normalize the image slice
        slice_image = modality_data[:, :, i]
        normalized_image = normalize_image(slice_image)
        
        # Extract the segmentation mask for the current slice
        slice_mask = seg_data[:, :, i]
        
        # Check if slice contains any relevant labels
        if np.any(np.isin(slice_mask, classes)):
            # Save the image
            image_filename = f"{counter:07d}.png"
            label_filename = f"{counter:07d}.txt"
            
            cv2.imwrite(os.path.join(output_images_dir, image_filename), normalized_image)

            # Generate YOLO-style label
            with open(os.path.join(output_labels_dir, label_filename), "w") as label_file:
                for class_id in classes:
                    binary_mask = (slice_mask == class_id).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) < 3:  # Skip small contours
                            continue
                        polygon = Polygon(contour.squeeze())
                        if polygon.is_valid:
                            height, width = binary_mask.shape
                            normalized_points = [(x / width, y / height) for x, y in polygon.exterior.coords[:-1]]
                            points_str = " ".join([f"{x} {y}" for x, y in normalized_points])
                            label_file.write(f"{class_id} {points_str}\n")
            
            counter += 1

# Main loop for processing all folders
classes = [1, 2, 4]  # Tumor classes
for folder in sorted(os.listdir(input_dir)):
    folder_path = os.path.join(input_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    flair_path = os.path.join(folder_path, f"{folder}_flair.nii.gz")
    seg_path = os.path.join(folder_path, f"{folder}_seg.nii.gz")
    
    if os.path.exists(flair_path) and os.path.exists(seg_path):
        save_slices_and_labels(flair_path, seg_path, classes)

# Filter valid images and labels
for label_file in os.listdir(output_labels_dir):
    label_path = os.path.join(output_labels_dir, label_file)
    
    if os.path.getsize(label_path) > 0:  # Non-empty label file
        image_file = label_file.replace(".txt", ".png")
        image_path = os.path.join(output_images_dir, image_file)
        
        if os.path.exists(image_path):
            shutil.copy(label_path, os.path.join(filtered_labels_dir, label_file))
            shutil.copy(image_path, os.path.join(filtered_images_dir, image_file))

print("Processing complete! Valid images and labels have been saved.")

