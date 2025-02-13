import torch
import torch.nn as nn
import torch.nn.functional as F  # For activation functions
import segmentation_models_pytorch as smp

class UNetEncoderClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(UNetEncoderClassifier, self).__init__()

        # Detail about model is present in it's doc
        # Loading a pre-trained UNet model (e.g., resnet18, efficientnet-b0)
        self.unet = smp.Unet(
            encoder_name="resnet34",      # Choose encoder (resnet18, resnet34, etc.)
            encoder_weights="imagenet",  # Use pre-trained weights from ImageNet
            in_channels=3,               # 3 input channels (RGB images)
            classes=1                  # Output channels for segmentation (not needed, but required by the library)
        )

        # Remove the decoder part of UNet
        self.unet.decoder = nn.Identity() # Remove the decoder part

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # Adjust input features (512) based on the encoder
            nn.ReLU(),
            nn.Dropout(p=0.65), # Add dropout
            nn.Linear(256, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.unet.encoder(x)  # Use only the encoder
        if isinstance(features, list):  # Check if it's a list
            features = features[-1]  # Taking out the last feature map - Last feature map is a good starting point
        pooled_features = self.global_avg_pool(features)
        pooled_features = pooled_features.view(features.size(0), -1) # Flatten
        logits = self.fc(pooled_features)
        probabilities = self.softmax(logits)
        return probabilities

class QuadCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):  # 3 classes: benign, malignant, normal
        super(QuadCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Increased filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Increased filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Increased filters
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Added conv layer + increased filters
        self.bn1 = nn.BatchNorm2d(32)  # After conv1
        self.bn2 = nn.BatchNorm2d(64)  # After conv2
        self.bn3 = nn.BatchNorm2d(128)  # After conv3
        self.bn4 = nn.BatchNorm2d(256) # After conv4
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling for downsampling
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # Increased neurons, adjusted input size
        self.fc2 = nn.Linear(512, 256)  # Increased neurons
        self.fc3 = nn.Linear(256, 128)  # Increased neurons
        self.fc4 = nn.Linear(128, num_classes)  # New FC layer

    def forward(self, x):
        # Conv Layer 1
        x = F.relu(self.bn1(self.conv1(x)))  # ReLU activation + BN
        x = self.pool(x)  # Max pooling

        # Conv Layer 2
        x = F.relu(self.bn2(self.conv2(x)))  # ReLU activation + BN
        x = self.pool(x)  # Max pooling

        # Conv Layer 3
        x = F.relu(self.bn3(self.conv3(x)))  # ReLU activation + BN
        x = self.pool(x)

        # Conv Layer 4
        x = F.relu(self.bn4(self.conv4(x)))  # New conv layer + BN + ReLU
        x = self.pool(x)

        # Flatten
        x = x.reshape(-1, 256 * 16 * 16)  # Correct size after the 4th conv layer
        x = self.dropout(x)  # Apply dropout

        # FC Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # FC Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # FC Layer 3
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc4(x)  # No activation here, CrossEntropyLoss will handle softmax
        return x

'''
Example code to test models.py
if __name__ == "__main__":
    model = SimpleCNN()
    # Example input:
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (1, 3) - probabilities for 3 classes

    # Check number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable_params:,} trainable parameters.")
'''