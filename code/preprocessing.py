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
import random

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preprocessing_fn=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

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
                # Apply specialized preprocessing for retinal images
                if self.preprocessing_fn:
                    img = self.preprocessing_fn(img)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                img = Image.fromarray(img)  # Convert to PIL for torchvision transforms
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def preprocess_retinal_image(image):
    """
    Advanced preprocessing for retinal images to enhance features
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        Preprocessed image
    """
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    height, width = image.shape[:2]
    
    # Create a circular mask for the retina
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 10
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # CLAHE for contrast enhancement (on LAB color space for better results)
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    
    return blurred

def preprocess_images(image_paths, labels, batch_size=32, augment=True, val_split=None):
    """
    Preprocess images: resize, normalize, convert to tensor, apply augmentations, and create DataLoader.
    Improved with advanced preprocessing and stronger augmentations.

    Args:
        image_paths (list): List of image file paths.
        labels (list): List of corresponding labels.
        batch_size (int): Batch size for DataLoader.
        augment (bool): Whether to apply data augmentation.
        val_split (float): If provided, split data into train and validation sets.

    Returns:
        DataLoader or tuple of DataLoaders: PyTorch DataLoader object(s).
    """
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
        labels = [labels]
    
    # Print number of images being processed
    print(f"Processing {len(image_paths)} images with {len(labels)} labels")
    
    # Check for balanced classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique_labels, counts))
    print("Class distribution:", class_distribution)

    # Define transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Define different transformation pipelines based on train/val/test
    if augment:
        # Strong augmentations for training
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size for random crops
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Retina images can be flipped vertically too
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Minimal processing for validation/test
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    if val_split is not None:
        # Split data into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, stratify=labels, random_state=42
        )
        
        # Create train dataset and DataLoader
        train_dataset = ImageDataset(
            train_paths, train_labels, transform=transform, preprocessing_fn=preprocess_retinal_image
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        
        # Create validation dataset and DataLoader (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = ImageDataset(
            val_paths, val_labels, transform=val_transform, preprocessing_fn=preprocess_retinal_image
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
    
    # Create dataset and DataLoader
    dataset = ImageDataset(
        image_paths, labels, transform=transform, preprocessing_fn=preprocess_retinal_image
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

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
    plt.title("Dataset Visualizer")
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

# Helper function to sample data while preserving class distribution
def sample_balanced_data(image_paths, labels, sample_percent=0.2):
    """
    Sample data while maintaining class distribution
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        sample_percent: Percentage of data to sample (0.0-1.0)
        
    Returns:
        Sampled image paths and labels
    """
    # Get unique classes
    unique_labels = np.unique(labels)
    
    sampled_paths = []
    sampled_labels = []
    
    # Sample from each class separately to maintain distribution
    for class_label in unique_labels:
        # Get indices for this class
        indices = [i for i, label in enumerate(labels) if label == class_label]
        
        # Calculate sample size for this class
        sample_size = max(1, int(len(indices) * sample_percent))
        
        # Randomly sample indices
        sampled_indices = random.sample(indices, sample_size)
        
        # Add sampled data
        sampled_paths.extend([image_paths[i] for i in sampled_indices])
        sampled_labels.extend([labels[i] for i in sampled_indices])
    
    # Print sampling stats
    print(f"Sampled {len(sampled_paths)} images ({sample_percent*100:.1f}% of {len(image_paths)})")
    
    # Verify class distribution
    unique, counts = np.unique(sampled_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Sampled class distribution: {distribution}")
    
    return sampled_paths, sampled_labels
    
def create_train_loader(sample_percent=0.5):
    """Create train dataloader with improved sampling and augmentation"""
    base_dir = "./content/Diabetic_Balanced_Data"
    
    # Train dataset paths
    class_dirs = [f"train/{i}" for i in range(5)]
    train_images = []
    train_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        full_dir = os.path.join(base_dir, class_dir).replace("\\", "/")
        if os.path.exists(full_dir):
            class_images = [os.path.join(full_dir, f).replace("\\", "/") for f in os.listdir(full_dir)]
            train_images.extend(class_images)
            train_labels.extend([class_idx] * len(class_images))
    
    # Sample data while preserving class distribution
    if sample_percent < 1.0:
        train_images, train_labels = sample_balanced_data(train_images, train_labels, sample_percent)
    
    # Use the preprocess_images function with strong augmentations
    return preprocess_images(train_images, train_labels, batch_size=32, augment=True)

def create_test_loader(sample_percent=0.5):
    """Create test dataloader with improved preprocessing"""
    base_dir = "./content/Diabetic_Balanced_Data"
    
    # Test dataset paths
    class_dirs = [f"test/{i}" for i in range(5)]
    test_images = []
    test_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        full_dir = os.path.join(base_dir, class_dir).replace("\\", "/")
        if os.path.exists(full_dir):
            class_images = [os.path.join(full_dir, f).replace("\\", "/") for f in os.listdir(full_dir)]
            test_images.extend(class_images)
            test_labels.extend([class_idx] * len(class_images))
    
    # Sample data if needed
    if sample_percent < 1.0:
        test_images, test_labels = sample_balanced_data(test_images, test_labels, sample_percent)
    
    # No augmentation for test data
    return preprocess_images(test_images, test_labels, batch_size=32, augment=False)

def create_val_loader(sample_percent=0.5):
    """Create validation dataloader with improved preprocessing"""
    base_dir = "./content/Diabetic_Balanced_Data"
    
    # Validation dataset paths
    class_dirs = [f"val/{i}" for i in range(5)]
    val_images = []
    val_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        full_dir = os.path.join(base_dir, class_dir).replace("\\", "/")
        if os.path.exists(full_dir):
            class_images = [os.path.join(full_dir, f).replace("\\", "/") for f in os.listdir(full_dir)]
            val_images.extend(class_images)
            val_labels.extend([class_idx] * len(class_images))
    
    # Sample data if needed
    if sample_percent < 1.0:
        val_images, val_labels = sample_balanced_data(val_images, val_labels, sample_percent)
    
    # No augmentation for validation data
    return preprocess_images(val_images, val_labels, batch_size=32, augment=False)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Define class names for diabetic retinopathy
    class_names = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    # Create DataLoaders with higher sample percentage for better performance
    train_loader = create_train_loader(sample_percent=0.5)
    test_loader = create_test_loader(sample_percent=0.5)
    val_loader = create_val_loader(sample_percent=0.5)
    
    print(f"Created train_loader with {len(train_loader.dataset)} images")
    print(f"Created test_loader with {len(test_loader.dataset)} images")
    print(f"Created val_loader with {len(val_loader.dataset)} images")
    
    # Visualize a batch from the training set
    print("Visualizing a batch from the training set:")
    visualize_batch(train_loader, class_names)