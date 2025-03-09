import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator
from preprocessing import create_val_loader, create_test_loader, create_train_loader

class DiabeticRetinopathyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(DiabeticRetinopathyCNN, self).__init__()
        
        # Improved CNN architecture with residual connections and deeper network
        # First convolutional block
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block with residual connection
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.skip2 = nn.Conv2d(64, 128, kernel_size=1)  # Skip connection
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block with residual connection
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.skip3 = nn.Conv2d(128, 256, kernel_size=1)  # Skip connection
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block with residual connection
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1)  # Skip connection
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fifth convolutional block
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.skip5 = nn.Conv2d(512, 512, kernel_size=1)  # Skip connection
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers with skip connections
        self.fc1 = nn.Linear(512, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.pool1(x)
        
        # Block 2 with residual connection
        identity = self.skip2(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = x + identity  # Add skip connection
        x = nn.functional.relu(x, inplace=True)
        x = self.pool2(x)
        
        # Block 3 with residual connection
        identity = self.skip3(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = x + identity  # Add skip connection
        x = nn.functional.relu(x, inplace=True)
        x = self.pool3(x)
        
        # Block 4 with residual connection
        identity = self.skip4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = x + identity  # Add skip connection
        x = nn.functional.relu(x, inplace=True)
        x = self.pool4(x)
        
        # Block 5 with residual connection
        identity = self.skip5(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = x + identity  # Add skip connection
        x = nn.functional.relu(x, inplace=True)
        x = self.pool5(x)
        
        # Global average pooling (replaces large fully connected layers)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers with skip and BN
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dropout(x)
        
        identity = x  # Save for skip connection
        x = self.fc2(x)
        x = self.fc_bn2(x)
        # No direct skip connection since dimensions differ, but we keep the concept
        x = nn.functional.relu(x, inplace=True)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Extract features from input for use with AdaBoost"""
        with torch.no_grad():
            # Block 1
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.pool1(x)
            
            # Block 2
            identity = self.skip2(x)
            x = self.conv2_1(x)
            x = self.bn2_1(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv2_2(x)
            x = self.bn2_2(x)
            x = x + identity
            x = nn.functional.relu(x, inplace=True)
            x = self.pool2(x)
            
            # Block 3
            identity = self.skip3(x)
            x = self.conv3_1(x)
            x = self.bn3_1(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv3_2(x)
            x = self.bn3_2(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv3_3(x)
            x = self.bn3_3(x)
            x = x + identity
            x = nn.functional.relu(x, inplace=True)
            x = self.pool3(x)
            
            # Block 4
            identity = self.skip4(x)
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv4_3(x)
            x = self.bn4_3(x)
            x = x + identity
            x = nn.functional.relu(x, inplace=True)
            x = self.pool4(x)
            
            # Block 5
            identity = self.skip5(x)
            x = self.conv5_1(x)
            x = self.bn5_1(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv5_2(x)
            x = self.bn5_2(x)
            x = nn.functional.relu(x, inplace=True)
            x = self.conv5_3(x)
            x = self.bn5_3(x)
            x = x + identity
            x = nn.functional.relu(x, inplace=True)
            x = self.pool5(x)
            
            # Global average pooling
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            
            # Get features after first FC layer
            x = self.fc1(x)
            x = self.fc_bn1(x)
            x = nn.functional.relu(x, inplace=True)
        
        return x

# PyTorch model wrapper for AdaBoost (unchanged)
class PyTorchClassifierWrapper(BaseEstimator):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
        self.model.to(self.device)
        self.classes_ = np.array([0, 1, 2, 3, 4])  # Assuming 5 classes
    
    def fit(self, X, y):
        # AdaBoost will use this to initialize and doesn't need to fit the CNN
        return self
    
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            # For the improved model, we need to use fc2 and fc3
            intermediate = nn.functional.relu(self.model.fc_bn2(self.model.fc2(X_tensor)))
            outputs = self.model.fc3(intermediate)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()
    
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            # For the improved model, we need to use fc2 and fc3
            intermediate = nn.functional.relu(self.model.fc_bn2(self.model.fc2(X_tensor)))
            outputs = self.model.fc3(intermediate)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()

# Improved training function with cosine annealing and mixed precision
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train the model with improved training routine
    
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
    
    # Use mixed precision training if on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Track loss and accuracy history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    patience = 5
    early_stop_counter = 0
    prev_val_loss = float('inf')
    
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
                    if phase == 'train' and scaler is not None:
                        # Use mixed precision for forward pass
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Backward + optimize with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            # Add gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
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
                early_stop_counter = 0  # Reset counter on improvement
            
            # Early stopping check
            if phase == 'val':
                if epoch_loss >= prev_val_loss:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                prev_val_loss = epoch_loss
        
        print()
        
        # Check for early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# The rest of the functions remain largely unchanged
def extract_features(model, dataloader, device='cuda'):
    """
    Extract features from the convolutional base for use with AdaBoost
    
    Args:
        model: Pre-trained CNN model
        dataloader: DataLoader for dataset
        device: Device to use
        
    Returns:
        features: Extracted features
        labels: Corresponding labels
    """
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
    model = model.to(device)
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            features_batch = model.get_features(inputs)
            features.append(features_batch.cpu().numpy())
            labels.append(targets.numpy())
    
    return np.vstack(features), np.concatenate(labels)

def train_adaboost(cnn_model, dataloaders, device='cuda', n_estimators=100):
    """
    Train an AdaBoost classifier on features extracted from the CNN
    
    Args:
        cnn_model: Pre-trained CNN model
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        device: Device to use
        n_estimators: Number of weak classifiers in AdaBoost
        
    Returns:
        adaboost_model: Trained AdaBoost model
    """
    print("Extracting features from CNN for AdaBoost training...")
    X_train, y_train = extract_features(cnn_model, dataloaders['train'], device)
    
    print(f"Training AdaBoost with {n_estimators} estimators...")
    
    # Create a wrapper for the CNN final layer as the base estimator
    base_estimator = PyTorchClassifierWrapper(cnn_model, device)
    
    # Create and train the AdaBoost classifier with improved parameters
    adaboost = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        algorithm='SAMME.R',  # Use real-valued prediction confidence
        learning_rate=0.1,    # Adjusted learning rate
        random_state=42
    )
    
    adaboost.fit(X_train, y_train)
    
    # Evaluate on validation set
    X_val, y_val = extract_features(cnn_model, dataloaders['val'], device)
    val_accuracy = adaboost.score(X_val, y_val)
    print(f"AdaBoost validation accuracy: {val_accuracy:.4f}")
    
    return adaboost

def evaluate_model(model, test_loader, device='cuda', adaboost=None):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: Device to evaluate on ('cuda' or 'cpu')
        adaboost: Trained AdaBoost model (optional)
        
    Returns:
        accuracy: Accuracy on test set
    """
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    if adaboost:
        # AdaBoost evaluation
        print("Evaluating AdaBoost model...")
        X_test, y_true = extract_features(model, test_loader, device)
        y_pred = adaboost.predict(X_test)
    else:
        # Standard CNN evaluation
        print("Evaluating CNN model...")
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

def predict_single_image(model, image_path, transform, device='cuda', adaboost=None):
    """
    Make a prediction on a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        transform: PyTorch transform to apply to image
        device: Device to evaluate on ('cuda' or 'cpu')
        adaboost: Trained AdaBoost model (optional)
        
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
    if adaboost:
        # Extract features and use AdaBoost for prediction
        with torch.no_grad():
            features = model.get_features(img)
            features_np = features.cpu().numpy()
            probabilities = adaboost.predict_proba(features_np)[0]
            predicted_class = np.argmax(probabilities)
    else:
        # Use standard CNN prediction
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_class = torch.max(outputs, 1)
            probabilities = probabilities.cpu().numpy()
            predicted_class = predicted_class.item()
    
    return predicted_class, probabilities

def compare_models(cnn_model, adaboost_model, test_loader, device='cuda'):
    """
    Compare CNN and AdaBoost models
    
    Args:
        cnn_model: Trained CNN model
        adaboost_model: Trained AdaBoost model
        test_loader: DataLoader for test set
        device: Device to evaluate on
    """
    # Evaluate CNN model
    print("Evaluating CNN model:")
    cnn_accuracy = evaluate_model(cnn_model, test_loader, device)
    
    # Evaluate AdaBoost model
    print("\nEvaluating AdaBoost model:")
    adaboost_accuracy = evaluate_model(cnn_model, test_loader, device, adaboost_model)
    
    # Compare results
    print("\nModel Comparison:")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"AdaBoost Accuracy: {adaboost_accuracy:.4f}")
    print(f"Improvement: {(adaboost_accuracy - cnn_accuracy) * 100:.2f}%")
    
    # Plot comparison
    models = ['CNN', 'CNN + AdaBoost']
    accuracies = [cnn_accuracy, adaboost_accuracy]
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the improved model
    model = DiabeticRetinopathyCNN(num_classes=5)
    
    # Use your dataloaders from the provided code
    dataloaders = {
        'train': create_train_loader(),
        'val': create_val_loader()
    }
    test_loader = create_test_loader()
    
    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Improved optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Cosine annealing learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # Train the model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25,
        device='cuda'
    )
    
    # Train AdaBoost on the trained CNN features
    adaboost_model = train_adaboost(
        cnn_model=trained_model,
        dataloaders=dataloaders,
        device='cuda',
        n_estimators=100
    )
    
    # Evaluate the models on the test set
    print("Evaluating CNN model:")
    cnn_accuracy = evaluate_model(trained_model, test_loader, device='cuda')
    
    print("\nEvaluating CNN + AdaBoost model:")
    adaboost_accuracy = evaluate_model(trained_model, test_loader, device='cuda', adaboost=adaboost_model)
    
    # Compare the models
    compare_models(trained_model, adaboost_model, test_loader, device='cuda')
    
    # Example of making a prediction on a single image
    from torchvision import transforms
    
    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Predict on a single image
    image_path = "path_to_your_image.jpg"
    predicted_class, probabilities = predict_single_image(
        model=trained_model,
        image_path=image_path,
        transform=transform,
        device='cuda',
        adaboost=adaboost_model
    )
    
    # Print the prediction
    class_names = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    print(f"Predicted Class: {class_names[predicted_class]}")
    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob:.4f}")