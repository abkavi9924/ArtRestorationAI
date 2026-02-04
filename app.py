import torch
import cv2
from torchvision import transforms
from scriptv1 import RestorationUNet
import streamlit as st
from PIL import Image
import numpy as np

# Function to load the saved model
def load_model(gpu_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RestorationUNet().to(device)
    
    # Select the appropriate model path based on the device
    model_path = gpu_model_path
    
    # Use map_location to ensure compatibility with CPU if CUDA is not available
    
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    
    return model, device

# Your original restore_image function remains UNCHANGED
def restore_image(model, distorted_img_path, transform, device):
    distorted_img = cv2.imread(distorted_img_path)
    
    # Ensure the image is loaded correctly
    if distorted_img is None:
        print(f"Error: Could not load image at {distorted_img_path}")
        return None

    # Check if the image is read in the correct color space
    distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)
    
    # Transform image and add batch dimension
    distorted_img = transform(distorted_img).unsqueeze(0).to(device)

    print(f"Distorted image shape: {distorted_img.shape}")  # Debugging step

    with torch.no_grad():
        restored_img = model(distorted_img)
        print(f"Restored image tensor: {restored_img}")  # Debugging step

    if restored_img is None:
        print("Error: Model returned None")
        return None

    # Remove batch dimension and move to CPU
    restored_img = restored_img.squeeze(0).cpu()

    # Convert to PIL image for displaying
    restored_img = transforms.ToPILImage()(restored_img)  
    return restored_img

# Setup transformation (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Streamlit UI setup
st.title("AI Art Restoration Model")
st.write("Upload a distorted image, and the model will restore it.")
gpu_model_path = "(og)restoration_unet.pth"
# Upload the model file

# Upload the distorted image
uploaded_file = st.file_uploader("Choose a distorted image...", type=["jpg", "jpeg", "png"])

# Load model and run inference if an image is uploaded
if uploaded_file is not None:
    # Display the distorted image
    st.image(uploaded_file, caption='Distorted Image', use_column_width=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_model(gpu_model_path)
    
    # Save uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Restore the image using your original function
    restored_img = restore_image(model, "temp_image.jpg", transform, device)

    # Show the restored image
    if restored_img:
        st.image(restored_img, caption='Restored Image', use_column_width=True)
        # Optionally, save the image
        save_option = st.checkbox("Save restored image?")
        if save_option:
            restored_img.save("restored_image.png")
            st.write("Restored image saved as 'restored_image.png'.")
else:
    st.write("Please upload a distorted image.")
