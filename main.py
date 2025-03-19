import os 
import shutil
import random

# Paths
train_path = "C:\\Users\\merchant\\Desktop\\code\\Road surface classification\\train"
folder_path = "C:\\Users\\merchant\\Desktop\\code\\Road surface classification\\train_subsamples"

# Create folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Define target classes
classes = ["wet_asphalt_smooth", "water_asphalt_smooth", "dry_concrete_smooth", "dry_asphalt_smooth", "fresh_snow", "wet_concrete_smooth", "dry_gravel", "melted_snow", "ice", "dry_mud", "wet_mud", "wet_gravel", "water_concrete_smooth", "water_mud"]

# Iterate over selected classes
for class_name in classes:
    class_train_path = os.path.join(train_path, class_name)   # Source folder
    class_subsample_path = os.path.join(folder_path, class_name)  # Destination folder

    # Ensure class folder exists in train_subsamples
    os.makedirs(class_subsample_path, exist_ok=True)

    # Get list of image files
    image_files = []
    for f in os.listdir(class_train_path):  
        file_path = os.path.join(class_train_path, f)  
        if os.path.isfile(file_path):  
            image_files.append(f)  

    # Compute 5% of images to sample
    sample_num = round(0.02 * len(image_files))

    # Randomly sample images
    sampled_images = random.sample(image_files, sample_num)

    # Copy sampled images to the new location
    for image in sampled_images:
        src = os.path.join(class_train_path, image)
        dest = os.path.join(class_subsample_path, image)
        shutil.copy(src, dest)

    print(f"Copied {sample_num} images to {class_subsample_path}")
