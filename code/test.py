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
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Check if file exists first
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            # Return a placeholder black image instead of failing
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                # Return a placeholder black image
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = Image.fromarray(img)  # Convert to PIL for torchvision transforms
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def preprocess_images(image_paths, labels, batch_size=32, augment=True):
    """
    Preprocess images: resize, normalize, convert to tensor, apply augmentations, and create DataLoader.

    Args:
        image_paths (list): List of image file paths.
        labels (list): List of corresponding labels.
        batch_size (int): Batch size for DataLoader.
        augment (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
        labels = [labels]
    
    # Print number of images being processed
    print(f"Processing {len(image_paths)} images with {len(labels)} labels")

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
    dataset = ImageDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Function to visualize batch with labels
def visualize_batch(dataloader, class_names):
    """
    Visualize a batch of images with their labels
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: Dictionary mapping class indices to class names
    """
    images, labels = next(iter(dataloader))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display sample images
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        # Convert tensor to numpy array and transpose to correct dimensions
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"Class: {class_names[labels[i].item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to check class distribution
def check_class_distribution(labels):
    """
    Check the distribution of classes in the dataset
    
    Args:
        labels: List of labels
    
    Returns:
        Dictionary with class counts
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    # Visualize distribution
    plt.figure(figsize=(10, 5))
    plt.bar(distribution.keys(), distribution.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(list(distribution.keys()))
    plt.show()
    
    return distribution

if __name__ == "__main__":
    
    base_dir = "./content/Diabetic_Balanced_Data"
    
    # Define class names for diabetic retinopathy
    class_names = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

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

    # Create lists of images and corresponding labels
    train_images = []
    train_labels = []
    
    for class_idx, image_list in {
        0: zero_images_train,
        1: one_images_train,
        2: two_images_train,
        3: three_images_train,
        4: four_images_train
    }.items():
        train_images.extend(image_list)
        train_labels.extend([class_idx] * len(image_list))
    
    test_images = []
    test_labels = []
    
    for class_idx, image_list in {
        0: zero_images_test,
        1: one_images_test,
        2: two_images_test,
        3: three_images_test,
        4: four_images_test
    }.items():
        test_images.extend(image_list)
        test_labels.extend([class_idx] * len(image_list))
    
    val_images = []
    val_labels = []
    
    for class_idx, image_list in {
        0: zero_images_val,
        1: one_images_val,
        2: two_images_val,
        3: three_images_val,
        4: four_images_val
    }.items():
        val_images.extend(image_list)
        val_labels.extend([class_idx] * len(image_list))
    
    # Check class distribution
    print("Training dataset class distribution:")
    train_distribution = check_class_distribution(train_labels)
    
    print("Testing dataset class distribution:")
    test_distribution = check_class_distribution(test_labels)
    
    print("Validation dataset class distribution:")
    val_distribution = check_class_distribution(val_labels)
    
    # Create DataLoaders for train, test, and validation sets
    train_loader = preprocess_images(train_images, train_labels, batch_size=32, augment=True)
    test_loader = preprocess_images(test_images, test_labels, batch_size=32, augment=False)
    val_loader = preprocess_images(val_images, val_labels, batch_size=32, augment=False)
    
    print(f"Created train_loader with {len(train_loader.dataset)} images")
    print(f"Created test_loader with {len(test_loader.dataset)} images")
    print(f"Created val_loader with {len(val_loader.dataset)} images")
    
    # Visualize a batch from the training set
    print("Visualizing a batch from the training set:")
    visualize_batch(train_loader, class_names)
    
    # Example of how to use the DataLoader
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
        print(f"Sample labels: {labels[:5]}")  # Print first 5 labels
        if batch_idx == 0:  # Just show first batch
            break