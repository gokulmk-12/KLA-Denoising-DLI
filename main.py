import os
import torch
import random
import numpy as np
from config import *
from PIL import Image
import streamlit as st
from networks import RIDNet
import  torch.nn.functional as F
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim

##################### Metrics ##################### 

def PSNR(ground_truth, predicted_image):
    mse = F.mse_loss(predicted_image, ground_truth)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(torch.tensor(max_pixel)/torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.moveaxis(img1, 0, -1)
        img2 = np.moveaxis(img2, 0, -1)

    score, _ = ssim(img1, img2, full=True, channel_axis=2, data_range=img1.max() - img1.min())
    
    return score

################################################### 

unetmodel = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
checkpointUnet = torch.load(UNET_MODEL_PATH)
unetmodel.load_state_dict(checkpointUnet)
unetmodel.eval()

ridnetmodel = RIDNet()
checkpointRIDnet = torch.load(RIDNET_MODEL_PATH)
ridnetmodel.load_state_dict(checkpointRIDnet)
ridnetmodel.eval()

class_names = ["choose", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
               "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

st.title("Welcome to Image Denoiser App!")

st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a Model", ("Choose", "RIDNet"))

if model_name == "RIDNet":
    st.sidebar.markdown("""
    **RIDNet (Real Image Denoising Network)**:
    - RIDNet is a deep learning model specifically designed to remove noise from images.
    - It uses residual connections to handle high-frequency noise effectively.
    """)

    st.sidebar.markdown("""
    **UNet (Defect Mask Segmentation)**:
    - UNet is the default model for defect mask segmentation
    """)

if "model_selected" not in st.session_state:
    st.session_state.model_selected = False
if "image_loaded" not in st.session_state:
    st.session_state.image_loaded = False
if "success_message_model" not in st.session_state:
    st.session_state.success_message_model = ""
if "success_message_image" not in st.session_state:
    st.session_state.success_message_image = ""

if model_name != "Choose" and st.sidebar.button("Use Selected Model"):
    st.session_state.model_selected = True
    st.session_state.success_message_model = f"Model '{model_name}' & U-Net selected successfully!"


st.markdown("""
This application is designed to denoise and restore images that are affected by noise and blur. Whether you have a noisy image from a dataset or you want to upload your own, this app provides a simple way to clean and enhance your images using advanced deep learning models.
""")

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

upload_option = st.radio("Choose Image Source", ("Random Image", "Upload Image"))

if "last_option" not in st.session_state:
    st.session_state.last_option = upload_option

if st.session_state.last_option != upload_option:
    st.session_state.selected_image = None
    st.session_state.last_option = upload_option
    st.session_state.image_loaded = False
    st.session_state.success_message_image = ""

if upload_option == "Random Image":
    selected_class = st.selectbox("Select Class for Random Image", class_names)
    
    if st.button("Select Random Image"):
        if selected_class == "choose":
            st.warning("Please select a valid class from the dropdown.")
        else:
            image_folder = f"random_images/{selected_class}"
            image_files = [f"{selected_class}_{i}.png" for i in range(1, 4)]
            random_image_file = random.choice(image_files)
            st.session_state.selected_image = Image.open(os.path.join(image_folder, random_image_file))
            st.session_state.image_loaded = True
            st.session_state.success_message_image = "Image loaded successfully!"
else:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.session_state.selected_image = Image.open(uploaded_file)
        st.session_state.image_loaded = True
        st.session_state.success_message_image = "Image loaded successfully!"

if st.session_state.selected_image is not None:
    st.image(st.session_state.selected_image, caption="Selected Image", use_column_width=True)
if st.session_state.success_message_model and not st.session_state.image_loaded:
    st.success(st.session_state.success_message_model)
if st.session_state.success_message_image and not st.session_state.model_selected:
    st.success(st.session_state.success_message_image)
if st.session_state.model_selected and st.session_state.image_loaded:
    st.success("Model and Image loaded successfully! Ready to denoise.")

st.markdown("""
    <style>
    .stButton button {
        background-color: black;
        color: white;
        font-size: 20px;
        padding: 10px 20px;
        margin: auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

if st.session_state.selected_image is not None:
    if st.button("Denoise Image"):
        if st.session_state.model_selected and st.session_state.image_loaded:
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            input_img = transform(st.session_state.selected_image).unsqueeze(0)
            input_image = input_img.detach().squeeze().numpy()
            input_image_np = (input_image*255).astype(np.uint8).transpose(1, 2, 0)
            input_image_pil = Image.fromarray(input_image_np)

            with torch.no_grad():
                denoised_torch = ridnetmodel(input_img)
                denoised= (denoised_torch - denoised_torch.min()) / (denoised_torch.max() - denoised_torch.min())
                mask_torch = unetmodel(denoised)
            
            denoised_img = denoised_torch.detach().squeeze().numpy()
            denoised_image_np = (denoised_img*255).astype(np.uint8).transpose(1, 2, 0)
            denoised_image_pil = Image.fromarray(denoised_image_np)
            
            mask = torch.sigmoid(mask_torch).squeeze(0).cpu().numpy()
            mask_np = (mask*255).astype(np.uint8)
            mask_image_pil = Image.fromarray(mask_np.squeeze())


            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(input_image_pil, caption="Input Image", use_column_width=True)
            with col2:
                st.image(denoised_image_pil, caption="Denoised Image", use_column_width=True)
            with col3:
                st.image(mask_image_pil, caption="Defect Mask", use_column_width=True)

        else:
            if not st.session_state.model_selected:
                st.warning("Please select a model before denoising.")
            if not st.session_state.image_loaded:
                st.warning("Please select an image to denoise.")
else:
    st.write("Please select an image to denoise.")