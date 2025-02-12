import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from viz import plot_confusion_matrix

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, device="cpu"):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []  # Store training losses
    test_losses = []   # Store test losses
    train_accuracies = [] # Store training accuracies
    test_accuracies = [] # Store test accuracies

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
        plot_confusion_matrix(cm, ['Benign', 'Malignant', 'Normal'], "confusion_matrix_simple_cnn.png")
        print("Classification Report:\n", classification_report(all_labels, all_predictions, target_names=["Benign", "Malignant", "Normal"]))
        
        epoch_test_loss = running_test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)

        epoch_test_acc = 100 * correct / total
        test_accuracies.append(epoch_test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%")

    return model, train_losses, test_losses, train_accuracies, test_accuracies  # Return the lists