import os
import random
import shutil

# Paths
dataset_dir = "..\\..\\Datasets\\CelebA-HQ - With Validation\\celeba_hq_256"
train_dir = "..\\..\\Datasets\\CelebA-HQ - With Validation\\train"
val_dir = "..\\..\\Datasets\\CelebA-HQ - With Validation\\val"

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all image paths
all_images = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir) if img.endswith((".jpg", ".png"))]
print(len(all_images))

# Shuffle images
random.seed(42)  # For reproducibility
random.shuffle(all_images)

# Split dataset
val_images = all_images[:2000]
train_images = all_images[2000:]

# Move images to respective directories
for img in val_images:
    shutil.copy(img, val_dir)  # Use shutil.move() if you want to move instead of copying

for img in train_images:
    shutil.copy(img, train_dir)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
