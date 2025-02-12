import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from viz import plot_confusion_matrix
import numpy as np

def train_model(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001, weight_decay=1e-5, device="cpu", numerical_labels=None):
    model.to(device)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(numerical_labels), y=numerical_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate/10.0, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # Learning rate scheduler

    train_losses = []  # Store training losses
    test_losses = []   # Store test losses
    train_accuracies = [] # Store training accuracies
    test_accuracies = [] # Store test accuracies

    best_test_loss = float('inf')  # Initialize with infinity
    patience = 5  # Number of epochs to wait before stopping
    counter = 0  # Counter for patience

    for epoch in range(num_epochs):
        model.train() # Training mode 
        running_loss = 0.0  # Loss for the epoch
        correct_train = 0
        total_train = 0

        for images, masks, labels in train_loader:
            # Move data to device 
            images = images.to(device)
            labels = labels.to(device)

            # Set gradients to zero at every epoch
            optimizer.zero_grad()

            # Prediction
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Back-Propogation
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1) # Training accuracy
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_train_acc = 100 * correct_train / total_train
        train_accuracies.append(epoch_train_acc)

        # Evaluation (on the test set)
        model.eval()
        correct = 0
        total = 0
        running_test_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels) # Test Loss
                running_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                 
        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(cm, ['Benign', 'Malignant', 'Normal'], "imgs/confusion_matrix_quad_cnn_weighted_model.png")
        # print("Classification Report:\n", classification_report(all_labels, all_predictions, target_names=["Benign", "Malignant", "Normal"]))
        
        epoch_test_loss = running_test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)

        epoch_test_acc = 100 * correct / total
        test_accuracies.append(epoch_test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%")

        # Early Stopping Implementation
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            torch.save(model.state_dict(), 'models/breast_cancer_quad_cnn_weighted_class_model.pth')  # Save best model weights
            counter = 0  # Reset counter
            print("Model Saved!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break  # Stop training

        scheduler.step(epoch_test_loss)  # Step the scheduler using the test loss

    return model, train_losses, test_losses, train_accuracies, test_accuracies  # Return the lists