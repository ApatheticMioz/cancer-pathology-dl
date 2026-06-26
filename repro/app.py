import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure imports from the local package work. 
# The original code expects to be inside a package called 'repro'.
# We can alias it so the internal absolute imports (e.g. from repro.config) succeed.
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

import types
repro = types.ModuleType("repro")
sys.modules["repro"] = repro
repro.__path__ = [str(root)]

from repro.config import DATASET_META
from repro.modeling import MultiTaskUNet
from repro.data import build_transforms

# --- Config ---
st.set_page_config(page_title="Multi-task UNet Inference", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"

AVAILABLE_DATASETS = ["tcga", "siim", "pannuke"]  # Excluded PANDA per your request
AVAILABLE_ENCODERS = ["vgg16", "mobilenet_v2"]

@st.cache_resource
def load_model(dataset: str, encoder: str):
    """Loads the model and weights statically so we don't reload on every UI interaction."""
    num_classes = DATASET_META[dataset]["num_classes"]
    seg_classes = DATASET_META[dataset]["seg_classes"]
    
    model = MultiTaskUNet(encoder_name=encoder, num_classes=num_classes, seg_classes=seg_classes)
    
    weights_path = WEIGHTS_DIR / f"{dataset}_{encoder}_best.pth"
    if weights_path.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(weights_path, map_location=device)
        # Handle case if it's nested or plain state_dict
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, None

def get_image_files(dataset: str):
    """Fetch images for a specific dataset from data/{dataset}/images/"""
    ds_dir = DATA_DIR / dataset / "images"
    if not ds_dir.exists():
        return []
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images = sorted([f for f in ds_dir.iterdir() if f.suffix.lower() in valid_exts])

    if dataset != "siim":
        return images

    non_empty_images = []
    for image_path in images:
        mask_path = get_mask_file(image_path)
        if not mask_path:
            continue
        mask = Image.open(mask_path).convert("L")
        if np.array(mask).max() > 0:
            non_empty_images.append(image_path)

    return non_empty_images

def get_mask_file(image_path: Path):
    """Attempts to find the corresponding mask for an image"""
    mask_dir = image_path.parent.parent / "masks"
    mask_path = mask_dir / image_path.name
    if mask_path.exists():
        return mask_path
    
    # Try different mask extensions or naming conventions just in case
    for ext in [".png", ".jpg", "_mask.png", "_mask.tif"]:
        alt_path = mask_dir / (image_path.stem + ext)
        if alt_path.exists():
            return alt_path
    return None

def run_inference(model, device, image, img_size):
    """Preprocess image, run inference, and return raw predictions"""
    _, val_tf = build_transforms(img_size)
    
    img_np = np.array(image.convert("RGB"))
    augmented = val_tf(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_pred, cls_pred = model(img_tensor)
        
    return seg_pred, cls_pred

# --- UI Setup ---
st.title("Multi-task UNet Inference")
st.markdown("Explore predictions for TCGA, SIIM, and PanNuke datasets.")

st.sidebar.header("Configuration")
selected_dataset = st.sidebar.selectbox("Select Dataset", AVAILABLE_DATASETS)
selected_encoder = st.sidebar.selectbox("Select Encoder", AVAILABLE_ENCODERS)

model, device = load_model(selected_dataset, selected_encoder)

if model is None:
    st.error(f"Weights not found for requested configuration: `weights/{selected_dataset}_{selected_encoder}_best.pth`")
else:
    st.sidebar.success(f"Model loaded successfully on {device}!")

    images = get_image_files(selected_dataset)
    if not images:
        st.warning(f"No images found in `data/{selected_dataset}/images/`")
    else:
        selected_img_name = st.sidebar.selectbox("Select Image", [img.name for img in images])
        selected_img_path = DATA_DIR / selected_dataset / "images" / selected_img_name
        
        col1, col2, col3 = st.columns(3)
        image = Image.open(selected_img_path)
        col1.subheader("Original Image")
        col1.image(image, use_container_width=True)
        
        # Load and show ground truth mask
        mask_path = get_mask_file(selected_img_path)
        if mask_path:
            mask = Image.open(mask_path)
            col2.subheader("Ground Truth Mask")
            col2.image(mask, use_container_width=True)
        else:
            col2.subheader("Ground Truth Mask")
            col2.info("No corresponding mask found.")

        # Inference button
        if st.button("Run Inference"):
            with st.spinner("Running prediction..."):
                img_size = DATASET_META[selected_dataset]["img_size"]
                seg_pred, cls_pred = run_inference(model, device, image, img_size)
                
                # Class prediction processing
                predicted_class_idx = torch.argmax(cls_pred, dim=1).item()
                st.subheader(f"Predicted Class Index: `{predicted_class_idx}`")
                
                # Seg prediction processing
                # seg_pred shape: [1, num_classes, H, W]
                # Map to [H, W] label mask
                seg_pred_mask = torch.argmax(seg_pred, dim=1).squeeze(0).cpu().numpy()
                
                # Display output mask using matplotlib to assign colors properly
                fig, ax = plt.subplots()
                ax.imshow(seg_pred_mask, cmap='jet')
                ax.axis("off")
                col3.subheader("Predicted Mask")
                col3.pyplot(fig)
