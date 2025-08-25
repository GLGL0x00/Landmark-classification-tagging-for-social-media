import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIG & STYLING
# ==============================
st.set_page_config(
    page_title="Landmark Classification App",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark CSS
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3, h4 {
            color: #FAFAFA !important;
        }
        .css-1d391kg {
            color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODELS (TorchScript)
# ==============================
@st.cache_resource
def load_model(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model

cnn_model = load_model("models/cnn_scratch.pt")
rescnn_model = load_model("models/cnn_residual.pt")
resnet_model = load_model("models/transfer_exported.pt")

# ==============================
# PREDICTION HELPER
# ==============================

def predict(image, model):
    img_t = T.ToTensor()(image).unsqueeze(0)
    labels = []

    outputs = model(img_t)
    probs = outputs.data.cpu().numpy().squeeze()
    idxs = np.argsort(probs)[::-1]
    
    for i in range(5):
        # Get softmax value
        p = probs[idxs[i]]

        # Get class name
        landmark_name = model.class_names[idxs[i]]

        labels.append((landmark_name, float(p)))

    return labels

# ==============================
# UI LAYOUT
# ==============================
st.title("🗺️ Landmark Classification App")
st.markdown("Upload a landmark image to compare predictions from three different models.")

# Sidebar info
st.sidebar.header("About this project")
st.sidebar.info("""
This project was built as part of the **AWS ML Engineer Nanodegree (Udacity + AWS)**.  
It compares three models:  
- CNN (Scratch)  
- CNN + Residual Connections  
- Transfer Learning (ResNet34)  

Author: [Ahmed Abdelgelel](https://linkedin.com/in/glgl0x00)  
Code: [GitHub Repo](https://github.com/GLGL0x00/Landmark-classification-tagging-for-social-media)
""")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

   
    # class_names = cnn_model.class_names

    # Predictions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("CNN (Scratch)")
        preds = predict(image, cnn_model)
        st.write(f"**Top-1:** {preds[0][0]} ({preds[0][1]*100:.1f}%)")
        fig, ax = plt.subplots()
        ax.barh([p[0] for p in preds], [p[1] for p in preds], color="#1f77b4")
        ax.invert_yaxis()
        st.pyplot(fig)

    with col2:
        st.subheader("CNN + Residuals")
        preds = predict(image, rescnn_model)
        st.write(f"**Top-1:** {preds[0][0]} ({preds[0][1]*100:.1f}%)")
        fig, ax = plt.subplots()
        ax.barh([p[0] for p in preds], [p[1] for p in preds], color="#ff7f0e")
        ax.invert_yaxis()
        st.pyplot(fig)

    with col3:
        st.subheader("Transfer Learning (ResNet34)")
        preds = predict(image, resnet_model)
        st.write(f"**Top-1:** {preds[0][0]} ({preds[0][1]*100:.1f}%)")
        fig, ax = plt.subplots()
        ax.barh([p[0] for p in preds], [p[1] for p in preds], color="#2ca02c")
        ax.invert_yaxis()
        st.pyplot(fig)
