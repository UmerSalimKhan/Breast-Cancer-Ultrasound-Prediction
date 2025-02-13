import torch
from utils import load_data
from models import QuadCNN, UNetEncoderClassifier  # CNN model 
from train import train_model, train_UNet_model  # Import the training function
from viz import plot_training_history, visualize_data  # Import the plotting function

if __name__ == "__main__":
    data_dir = "D://Datasets//Breast_Cancer_Ultrasound_dataset//Dataset_BUSI_with_GT"
    train_loader, test_loader, label_mapping, numerical_labels = load_data(data_dir)

    # Sample image visualization
    # visualize_data(train_loader, label_mapping)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device

    # UNET 
    model = UNetEncoderClassifier()
    trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_UNet_model(model, train_loader, test_loader, device=device, numerical_labels=numerical_labels)
    
    # QuadCNN
    # model = QuadCNN(dropout_rate=0.3)
    # trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, device=device, numerical_labels=numerical_labels) # Pass device to train_model

    # Saving model
    torch.save(trained_model.state_dict(), "models/unet/breast_cancer_unet_model_lr_sched_droupout60per_wtdecay_class_wt.pth")
    print("Training complete. Model saved as models/unet/breast_cancer_unet_model_lr_sched_droupout60per_wtdecay_class_wt.pth")

    # Plot model history
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, 'imgs/unet/breast_cancer_unet_training_history_lr_sched_droupout60per_wtdecay_class_wt.png') # Calling the plotting function