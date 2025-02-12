import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from models import SimpleCNN  # model class

# Vaiables
# Convert labels to numerical values 
label_mapping = {"benign": 0, "malignant": 1, "normal": 2}  # Define your mapping

# Load the trained model and label mapping (do this ONCE at the start)
@st.cache_resource  # Caching the loaded model to avoid reloading on every interaction
def load_model_and_mapping(model_path, device):
    model = SimpleCNN()  # model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model, label_mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/breast_cancer_quad_cnn_weighted_class_model.pth"  # Path to the saved model
model, label_mapping = load_model_and_mapping(model_path, device)


st.title("Breast Cancer Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB even if PNG is grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # predicted_class_index = 0 # index

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)  # Get probabilities
            predicted_class_index = torch.argmax(probabilities).item()  # Get the index

        # print("Predicted class index: ", predicted_class_index)
        # Find the key (class name) associated with the predicted index
        for label, index in label_mapping.items():
            # print("Label: ", label)
            # print("Index: ", index)

            if index == predicted_class_index:
                predicted_label = label
                break  # Exit loop once you've found the label
        else:  # This else is associated with the for loop, in case no break occurs.
            predicted_label = "Unknown" #Handles the case where the predicted class is not found.
        
        # print("Predict labels: ", predicted_label)

        st.write(f"## Prediction: {predicted_label}")

        # Display probabilities for each class
        for label, index in label_mapping.items():
            probability = probabilities[0][index].item() * 100
            st.write(f"{label}: {probability:.2f}%")