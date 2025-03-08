import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocessing import create_val_loader, create_test_loader, create_train_loader
class DiabeticRetinopathyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(DiabeticRetinopathyCNN, self).__init__()
        
        # CNN architecture
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Dropout for regularization
            nn.Dropout2d(0.5),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train the model and return the best model weights
    
    Args:
        model: PyTorch model
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs for training
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Trained model with best weights
    """
    since = time.time()
    
    # Use GPU if available
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Track loss and accuracy history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Accuracy on test set
    """
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    class_names = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    target_names = [class_names[i] for i in range(len(class_names))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    return accuracy

def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show class labels
    classes = [class_names[i] for i in range(len(class_names))]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

def predict_single_image(model, image_path, transform, device='cuda'):
    """
    Make a prediction on a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        transform: PyTorch transform to apply to image
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        predicted_class: Predicted class index
        probabilities: Class probabilities
    """
    from PIL import Image
    import cv2
    import numpy as np
    
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item(), probabilities.cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = DiabeticRetinopathyCNN(num_classes=5)
    
    # Use your dataloaders from the provided code
    dataloaders = {
        'train': create_train_loader(),
        'val': create_val_loader()
    }
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trained_model = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        exp_lr_scheduler,
        num_epochs=15,
        device=device
    )

    test_loader = create_test_loader()
    # Evaluate on test set
    test_accuracy = evaluate_model(trained_model, test_loader, device=device)
    
    # Save the model
    torch.save(trained_model.state_dict(), './model/diabetic_retinopathy_model.pth')
    print("Model saved to diabetic_retinopathy_model.pth")
    
    # Example of how to load the model and make a prediction
    # model = DiabeticRetinopathyCNN(num_classes=5)
    # model.load_state_dict(torch.load('diabetic_retinopathy_model.pth'))
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # predicted_class, probabilities = predict_single_image(model, 'path/to/image.jpg', transform)
    # print(f"Predicted class: {class_names[predicted_class]}")
    # print(f"Probabilities: {probabilities}")