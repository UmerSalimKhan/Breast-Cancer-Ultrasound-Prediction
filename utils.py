import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform # Dictionary in our case 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale ("L")

        if self.transform:
            image = self.transform["image"](image) # Applying image transform 
            mask = self.transform["mask"](mask) # Applying masked transform

        return image, mask, label


def load_data(data_dir):
    image_paths = []
    mask_paths = []
    labels = []

    for class_name in os.listdir(data_dir):  # Iterate through benign, malignant, normal
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(".png") and "_mask" not in filename:  # Find image files
                    image_path = os.path.join(class_dir, filename)
                    mask_filename = filename.replace(".png", "_mask.png")
                    mask_path = os.path.join(class_dir, mask_filename)

                    if os.path.exists(mask_path): # Check if corresponding mask exists
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
                        labels.append(class_name)  # Store the class name as label
                    else:
                        print(f"Warning: No mask found for {filename}")

    # Convert labels to numerical values 
    label_mapping = {"benign": 0, "malignant": 1, "normal": 2}  # Define your mapping
    numerical_labels = [label_mapping[label] for label in labels]


    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize RGB images
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]) # No normalization for masks!

    dataset = BreastCancerDataset(image_paths, mask_paths, numerical_labels, 
                                  transform={"image": image_transform, "mask": mask_transform}) # Passing transforms as a dictionary

    # Stratified train/test split
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=numerical_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader, label_mapping, numerical_labels  

'''
Example to test utils.py
if __name__ == "__main__":
    data_dir = "D://Datasets//Breast_Cancer_Ultrasound_dataset//Dataset_BUSI_with_GT"  # Path of dataset folder 
    train_loader, test_loader, label_mapping = load_data(data_dir)

    # Print the number of items in the training set
    print(f"Number of training examples: {len(train_loader.dataset)}")
    print(f"Number of testing examples: {len(test_loader.dataset)}")
    print(f"Label Mapping: {label_mapping}")


    # Iterate through a batch and check shapes
    for images, masks, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        print("Label batch shape:", labels.shape)
        break # Just print one batch to avoid too much output
'''