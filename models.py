import torch 
import torch.nn as nn
import torch.nn.functional as F  # For activation functions

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: benign, malignant, normal
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling for downsampling
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Fully connected layer 
        self.fc2 = nn.Linear(128, num_classes)  # Output layer with num_classes neurons

    def forward(self, x):
        x = F.relu(self.conv1(x))  # ReLU activation
        x = self.pool(x)  # Max pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Max pooling
        x = x.view(-1, 32 * 64 * 64)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here, CrossEntropyLoss will handle softmax
        return x

'''
Example code to test models.py
if __name__ == "__main__":
    model = SimpleCNN()
    # Example input:
    input_tensor = torch.randn(1, 3, 256, 256) # Batch size 1, 3 channels, 256x256
    output = model(input_tensor)
    print("Output shape:", output.shape) # Should be (1, 3) - probabilities for 3 classes

    # Check number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable_params:,} trainable parameters.")
'''