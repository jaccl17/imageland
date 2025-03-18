import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Read the data
minecraft_data = pd.read_csv('/home/unitx/wabbit_playground/nn/minecraft_features_and_decisions.csv')

# Separate image names and labels
image_names = minecraft_data.iloc[:, 0]
labels = minecraft_data.iloc[:, 1:]

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    image_names, labels, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

def safe_image_parser(filename, label):
    try:
        # Convert filename to string and ensure it's a path
        filename = tf.strings.as_string(filename)
        full_path = tf.strings.join(['/home/unitx/wabbit_playground/nn/minecraft/', filename])
        
        # Read and decode image
        image_read = tf.io.read_file(full_path)
        image_decode = tf.image.decode_png(image_read, channels=3)
        
        # Resize and normalize
        image_resize = tf.image.resize(image_decode, [224, 224])
        image_normalize = image_resize / 255.0
        
        return image_normalize, label
    except Exception as e:
        # Return a placeholder tensor and original label
        print(f"Error processing {filename}: {e}")
        return tf.zeros([224, 224, 3]), label

def create_dataset(filenames, labels, is_training=True, batch_size=32):
    # Ensure filenames and labels are in the correct format
    filenames = tf.constant(filenames.tolist())
    labels = tf.constant(labels.tolist())
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Map parsing function
    dataset = dataset.map(
        safe_image_parser, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filenames))
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# Diagnostic function to check dataset
def diagnose_dataset(dataset, dataset_name):
    print(f"\nDiagnosing {dataset_name}:")
    try:
        # Try to iterate through the entire dataset
        total_batches = 0
        total_samples = 0
        
        for batch_features, batch_labels in dataset:
            total_batches += 1
            total_samples += batch_features.shape[0]
            
            # Print first batch details
            if total_batches == 1:
                print(f"First batch features shape: {batch_features.shape}")
                print(f"First batch labels shape: {batch_labels.shape}")
        
        print(f"Total batches: {total_batches}")
        print(f"Total samples: {total_samples}")
    
    except Exception as e:
        print(f"Error iterating through {dataset_name}: {e}")

# Create datasets
train_dataset = create_dataset(X_train, y_train.values, is_training=True)
val_dataset = create_dataset(X_val, y_val.values, is_training=False)
test_dataset = create_dataset(X_test, y_test.values, is_training=False)

# Diagnose each dataset
diagnose_dataset(train_dataset, "Training Dataset")
diagnose_dataset(val_dataset, "Validation Dataset")
diagnose_dataset(test_dataset, "Test Dataset")

# Optional: Verify image existence
def verify_image_files(filenames):
    missing_files = []
    for filename in filenames:
        full_path = os.path.join('/home/unitx/wabbit_playground/nn/minecraft/', str(filename))
        if not os.path.exists(full_path):
            missing_files.append(filename)
    
    if missing_files:
        print("\nMissing image files:")
        print(missing_files)
        print(f"Total missing files: {len(missing_files)}")
    else:
        print("\nAll image files exist!")

# Verify files
verify_image_files(X_train)