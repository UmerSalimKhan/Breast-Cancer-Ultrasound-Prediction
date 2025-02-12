import torch
from utils import load_data
from models import SimpleCNN  # CNN model 
from train import train_model  # Import the training function
from viz import plot_training_history, visualize_data  # Import the plotting function

if __name__ == "__main__":
    data_dir = "D://Datasets//Breast_Cancer_Ultrasound_dataset//Dataset_BUSI_with_GT"
    train_loader, test_loader, label_mapping = load_data(data_dir)

    # Sample image visualization
    visualize_data(train_loader, label_mapping)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device
    model = SimpleCNN()
    trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, device=device) # Pass device to train_model

    # Saving model
    torch.save(trained_model.state_dict(), "breast_cancer_simple_cnn_model.pth")
    print("Training complete. Model saved as breast_cancer_simple_cnn_model.pth")

    # Plot model history
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, "training_history_simple_cnn.png") # Calling the plotting function