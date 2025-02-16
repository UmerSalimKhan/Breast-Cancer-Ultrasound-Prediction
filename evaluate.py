import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from viz import plot_confusion_matrix

def evaluate_UNet_model(model, test_loader, device="cpu", num_classes=3, label_mapping=None):  # num_classes is important
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, _, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # For confusion matrix and classification report
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion matrix: \n", cm)

    # Convert numerical labels to text labels if mapping is provided
    if label_mapping:
        label_names = [key for key in label_mapping]  # Get text labels
    else:
        label_names = [str(i) for i in np.unique(all_labels)] # If no mapping, use string of number

    # Classification Report
    cr = classification_report(all_labels, all_predictions, target_names=label_names) # target_names added
    print("Classification Report:\n", cr)

    plot_confusion_matrix(cm, label_names, file_name="imgs/unet/confusion_matrix_unet_model_lr_sched_droupout60per.png") # Visualize and save

    return accuracy


def evaluate_model(model, test_loader, device="cpu", label_mapping=None):  # Added label_mapping
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())  # Store labels for metrics
            all_predictions.extend(predicted.cpu().numpy())  # Store predictions

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Metrics and Visualization
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    # Convert numerical labels to text labels if mapping is provided
    if label_mapping:
        label_names = [key for key in label_mapping]  # Get text labels
    else:
        label_names = [str(i) for i in np.unique(all_labels)] # If no mapping, use string of number

    # Classification Report
    print("Classification Report:\n", classification_report(all_labels, all_predictions, target_names=label_names))

    # Plot Confusion Matrix (using seaborn for better visualization)
    plot_confusion_matrix(cm, label_names, "imgs/confusion_matrix_evaluation_qaud_cnn_weighted_class.png")

    return accuracy  # You can return the accuracy if needed


if __name__ == "__main__":
    from models import UNetEncoderClassifier  # Or your model
    from utils import load_data

    data_dir = "D:/Datasets/Breast_Cancer_Ultrasound_dataset/Dataset_BUSI_with_GT"
    _, test_loader, label_mapping, _ = load_data(data_dir) # Only need the test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetEncoderClassifier() # model
    model.load_state_dict(torch.load("models/unet/breast_cancer_unet_model_lr_sched_droupout60per.pth", map_location=device)) # Load to correct device
    model.to(device) # Move to correct device
    evaluate_UNet_model(model, test_loader, device=device, label_mapping=label_mapping) # Pass label mapping