import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def visualize_data(train_loader, label_mapping):
    """Visualizes sample images and their masks from the training set."""
    images, masks, labels = next(iter(train_loader))  # Get a batch of data
    num_images = min(3, images.size(0))  # Show a maximum of 3 images

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Change from C, H, W to H, W, C
        mask = masks[i].squeeze().cpu().numpy()  # Remove channel dimension

        # Denormalize image (important for proper visualization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean  # Denormalize
        image = np.clip(image, 0, 1)  # Keep pixel values in [0, 1] range

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image (Label: {label_mapping.get(labels[i].item(), labels[i].item())})") # Show text label
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")  # Use gray colormap for mask
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("imgs/sample_images_and_masks.png")
    plt.show()


def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, file_name="training_history.png"):
    """Plots training loss and accuracy over epochs."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def plot_confusion_matrix(cm, label_names, file_name="confusion_matrix.png"):
    """Plots the confusion matrix using seaborn."""

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(file_name)
    plt.show()



def visualize_results(model, test_loader, device="cpu", label_mapping=None): # Combined visualization
    """Combines all visualizations."""

    # 1. Visualize Sample Data
    # visualize_data(test_loader, label_mapping)

    # 2. Confusion Matrix and Classification Report (moved from evaluate.py)
    model.eval()  # Set to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    if label_mapping:
        label_names = [key for key in label_mapping]  # Get text labels
    else:
        label_names = [str(i) for i in np.unique(all_labels)]

    print("Classification Report:\n", classification_report(all_labels, all_predictions, target_names=label_names))
    plot_confusion_matrix(cm, label_names) # Plot confusion matrix

'''
Example code to test viz.py
if __name__ == "__main__":
    # Example usage (you would call this from your main.py after training)
    from models import SimpleCNN  # Or your model
    from utils import load_data

    data_dir = "D:/Datasets/Breast_Cancer_Ultrasound_dataset/Dataset_BUSI_with_GT"
    train_loader, test_loader, label_mapping = load_data(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()  # model
    model.load_state_dict(torch.load("breast_cancer_model.pth", map_location=device))
    model.to(device)

    visualize_results(model, test_loader, device=device, label_mapping=label_mapping) # Call combined visualization function

    # Example of plotting training history (you'd have to collect these values during training)
    # train_losses = [...]
    # test_losses = [...]
    # train_accuracies = [...]
    # test_accuracies = [...]
    # plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)
'''