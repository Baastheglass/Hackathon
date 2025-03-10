import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Import the model class from the module where you saved it
from diabetic_retinopathy_cnn import DiabeticRetinopathyCNN, predict_single_image

# Define class names for reference
class_names = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# 1. OPTION 1: TRAIN A NEW MODEL FROM SCRATCH
# ---------------------------------------------
def train_new_model(train_loader, val_loader, test_loader):
    """
    Train a new model from scratch
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # Initialize model, loss function, and optimizer
    model = DiabeticRetinopathyCNN(num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3, 
                                                         verbose=True)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders dictionary
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Train the model
    from diabetic_retinopathy_cnn import train_model, evaluate_model
    
    trained_model = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=20,  # Adjust number of epochs as needed
        device=device
    )
    
    # Evaluate on test set
    evaluate_model(trained_model, test_loader, device=device)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'diabetic_retinopathy_model.pth')
    print("Model saved to diabetic_retinopathy_model.pth")
    
    return trained_model


# 2. OPTION 2: LOAD A PRE-TRAINED MODEL
# --------------------------------------
def load_pretrained_model(model_path='diabetic_retinopathy_cnn_model.pth'):
    """
    Load a previously trained model
    
    Args:
        model_path: Path to saved model weights
        
    Returns:
        model: Loaded model
    """
    # Initialize model architecture
    model = DiabeticRetinopathyCNN(num_classes=5)
    
    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        # If trained on GPU but loading on CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path}")
    return model


# 3. MAKE PREDICTIONS ON SINGLE IMAGES
# ------------------------------------
def predict_image(model, image_path):
    """
    Predict DR grade for a single image
    
    Args:
        model: Trained model
        image_path: Path to retina image
    """
    # Define image transformations (same as used in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Get prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_class, probabilities = predict_single_image(model, image_path, transform, device)
    
    # Display results
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"Predicted class: {class_names[predicted_class]}")
    
    # Show probabilities for each class
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob*100:.2f}%")
    
    # Display the image with prediction
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}\nConfidence: {probabilities[predicted_class]*100:.2f}%")
    plt.axis('off')
    plt.show()
    
    return predicted_class, probabilities


# 4. BATCH PREDICTION ON MULTIPLE IMAGES
# --------------------------------------
def predict_batch(model, image_folder, extension='.jpg'):
    """
    Predict DR grade for all images in a folder
    
    Args:
        model: Trained model
        image_folder: Folder containing retinal images
        extension: Image file extension to look for
    """
    # Get all image files
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.endswith(extension)]
    
    if len(image_paths) == 0:
        print(f"No {extension} images found in {image_folder}")
        return
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Process all images
    results = []
    for img_path in image_paths:
        try:
            predicted_class, probabilities = predict_single_image(
                model, img_path, transform, 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            results.append({
                'image': os.path.basename(img_path),
                'prediction': class_names[predicted_class],
                'confidence': probabilities[predicted_class],
                'probabilities': probabilities
            })
            
            print(f"Processed {os.path.basename(img_path)}: {class_names[predicted_class]}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
    
    # Display summary
    print(f"\nProcessed {len(results)} images")
    
    # Count predictions by class
    class_counts = {}
    for result in results:
        pred = result['prediction']
        class_counts[pred] = class_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images ({count/len(results)*100:.1f}%)")
    
    return results


# 5. EXAMPLE USAGE
# ----------------
if __name__ == "__main__":
    # Example 1: Training a new model
    # Assuming you have already created the data loaders as in your original code
    # train_new_model(train_loader, val_loader, test_loader)
    
    # Example 2: Loading a pretrained model and making predictions
    # First, load a trained model
    print(os.getcwd())
    model = load_pretrained_model('./model/diabetic_retinopathy_cnn_model.pth')
    
    # Make prediction on a single image
    predict_image(model, './content/Diabetic_Balanced_Data/test/4/IDRiD_040.jpg')
    
    # Or make predictions on all images in a folder
    # results = predict_batch(model, './content/Diabetic_Balanced_Data/test/3/')