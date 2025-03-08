import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Check if file exists first
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            # Return a placeholder black image instead of failing
            # img = np.zeros((224, 224, 3), dtype=np.uint8)
            # img = Image.fromarray(img)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                # Return a placeholder black image
                # img = np.zeros((224, 224, 3), dtype=np.uint8)
                # img = Image.fromarray(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = Image.fromarray(img)  # Convert to PIL for torchvision transforms
        
        if self.transform:
            img = self.transform(img)
        
        return img

def preprocess_images(image_paths, batch_size=32, augment=True):
    """
    Preprocess images: resize, normalize, convert to tensor, apply augmentations, and create DataLoader.

    Args:
        image_paths (list): List of image file paths.
        batch_size (int): Batch size for DataLoader.
        augment (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Print number of images being processed
    print(f"Processing {len(image_paths)} images")

    # Define transformations
    transform_list = [
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet mean/std)
    ]

    # Add augmentations if enabled
    if augment:
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
            transforms.RandomRotation(30),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # Color jitter
        ]
        transform_list = augmentations + transform_list  # Prepend augmentations

    transform = transforms.Compose(transform_list)
    
    # Create dataset and DataLoader
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    
    base_dir = "./content/Diabetic_Balanced_Data"

    # Train dataset
    zero_images_train = [os.path.join(base_dir, "train/0", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "train/0"))]
    one_images_train = [os.path.join(base_dir, "train/1", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "train/1"))]
    two_images_train = [os.path.join(base_dir, "train/2", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "train/2"))]
    three_images_train = [os.path.join(base_dir, "train/3", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "train/3"))]
    four_images_train = [os.path.join(base_dir, "train/4", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "train/4"))]

    # Test dataset
    zero_images_test = [os.path.join(base_dir, "test/0", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "test/0"))]
    one_images_test = [os.path.join(base_dir, "test/1", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "test/1"))]
    two_images_test = [os.path.join(base_dir, "test/2", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "test/2"))]
    three_images_test = [os.path.join(base_dir, "test/3", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "test/3"))]
    four_images_test = [os.path.join(base_dir, "test/4", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "test/4"))]

    # Validation dataset
    zero_images_val = [os.path.join(base_dir, "val/0", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "val/0"))]
    one_images_val = [os.path.join(base_dir, "val/1", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "val/1"))]
    two_images_val = [os.path.join(base_dir, "val/2", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "val/2"))]
    three_images_val = [os.path.join(base_dir, "val/3", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "val/3"))]
    four_images_val = [os.path.join(base_dir, "val/4", f).replace("\\", "/") for f in os.listdir(os.path.join(base_dir, "val/4"))]

    # Combine all images for each class
    train_images = {
        0: zero_images_train,
        1: one_images_train,
        2: two_images_train,
        3: three_images_train,
        4: four_images_train
    }
    
    test_images = {
        0: zero_images_test,
        1: one_images_test,
        2: two_images_test,
        3: three_images_test,
        4: four_images_test
    }
    
    val_images = {
        0: zero_images_val,
        1: one_images_val,
        2: two_images_val,
        3: three_images_val,
        4: four_images_val
    }
    
    # Create DataLoaders for train, test, and validation sets
    train_loader = preprocess_images([path for paths in train_images.values() for path in paths], batch_size=32, augment=True)
    test_loader = preprocess_images([path for paths in test_images.values() for path in paths], batch_size=32, augment=False)
    val_loader = preprocess_images([path for paths in val_images.values() for path in paths], batch_size=32, augment=False)
    
    print(f"Created train_loader with {len(train_loader.dataset)} images")
    print(f"Created test_loader with {len(test_loader.dataset)} images")
    print(f"Created val_loader with {len(val_loader.dataset)} images")
    
    # Example of how to use the DataLoader
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: Shape {batch.shape}")
        if batch_idx == 0:  # Just show first batch
            break