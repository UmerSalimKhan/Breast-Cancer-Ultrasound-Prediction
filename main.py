import torch
from utils import load_data
from models import QuadCNN  # CNN model 
from train import train_model  # Import the training function
from viz import plot_training_history, visualize_data  # Import the plotting function

if __name__ == "__main__":
    data_dir = "D://Datasets//Breast_Cancer_Ultrasound_dataset//Dataset_BUSI_with_GT"
    train_loader, test_loader, label_mapping, numerical_labels = load_data(data_dir)

    # Sample image visualization
    # visualize_data(train_loader, label_mapping)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device
    model = QuadCNN(dropout_rate=0.3)
    trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, device=device, numerical_labels=numerical_labels) # Pass device to train_model

    # Saving model
    torch.save(trained_model.state_dict(), "models/breast_cancer_quad_cnn_weighted_class_model.pth")
    print("Training complete. Model saved as models/breast_cancer_quad_cnn_weighted_class_model.pth")

    # Plot model history
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, "imgs/training_history_quad_cnn_weighted_class.png") # Calling the plotting function