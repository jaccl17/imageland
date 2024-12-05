import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_dir = "/home/unitx/wabbit_playground/nn/weather_dataset"
base_dir = "/home/unitx/wabbit_playground/nn/split_weather_dataset"

# Ratios
train_ratio = 0.7
val_ratio = 0.15  # Test will automatically take the rest (0.15)

# Creating directories for split
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Splitting process
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        train_files, temp_files = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=(val_ratio / (1 - train_ratio)), random_state=42)

        for split, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            split_class_dir = os.path.join(base_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for file_name in file_list:
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(split_class_dir, file_name)
                shutil.copy(src_path, dst_path)

print("Data split completed!")
