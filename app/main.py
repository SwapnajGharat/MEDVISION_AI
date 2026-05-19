import streamlit as st
import torch
import cv2
import numpy as np
from src.models import MultimodalFusionNet
from PIL import Image
import matplotlib.pyplot as plt

# 1. DARK MEDICAL THEME
st.set_page_config(page_title="MedVision AI | Dark Mode", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Global text forced to white/cyan */
    h1, h2, h3, p, span, label, .stMarkdown { 
        color: #e0e0e0 !important; 
        font-family: 'Inter', sans-serif;
    }
    
    h1 { color: #00d4ff !important; text-shadow: 0px 0px 10px rgba(0,212,255,0.3); }

    /* Container boxes */
    .main .block-container {
        background-color: #161b22;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
    }

    /* Professional Blue Button */
    .stButton>button {
        background-color: #238636 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0px 0px 15px rgba(46,160,67,0.4);
    }

    /* Result Card */
    .report-card {
        background-color: #0d1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00d4ff;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. GRAD-CAM LOGIC
def generate_heatmap(model, img_tensor, clinical_tensor):
    model.eval()
    target_layer = model.vision_model.features[-1]
    
    features, gradients = [], []
    def save_f(m, i, o): features.append(o)
    def save_g(m, gi, go): gradients.append(go[0])
    
    h_f = target_layer.register_forward_hook(save_f)
    h_g = target_layer.register_full_backward_hook(save_g)
    
    output = model(img_tensor, clinical_tensor)
    model.zero_grad()
    output.backward()
    
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features[0], dim=1).squeeze().detach().numpy()
    
    h_f.remove()
    h_g.remove()
    
    cam = np.maximum(cam, 0)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    model = MultimodalFusionNet(tabular_dim=3)
    model.load_state_dict(torch.load("models_saved/final_model.pth", map_location=torch.device('cpu')))
    return model

model = load_trained_model()

# --- 4. DASHBOARD ---
st.title("🩺 MedVision AI Dashboard")
st.write("Multimodal Tuberculosis Analysis System")
st.divider()

c1, c2 = st.columns([1, 2], gap="large")

with c1:
    st.subheader("📊 Clinical Vitals")
    age = st.slider("Patient Age", 18, 95, 45)
    glu = st.number_input("Glucose (mg/dL)", 70, 400, 110)
    bp = st.number_input("Systolic BP", 80, 200, 125)
    
    st.divider()
    up_file = st.file_uploader("📤 Upload X-ray", type=["png", "jpg", "jpeg"])

with c2:
    if up_file:
        raw_img = Image.open(up_file).convert('RGB')
        img_np = np.array(raw_img)
        
        st.subheader("🧪 AI Diagnostics")
        t1, t2 = st.tabs(["View Radiograph", "Neural Heatmap"])
        
        # Prep tensors
        img_res = cv2.resize(img_np, (224, 224))
        img_t = torch.tensor(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_t.requires_grad = True
        clin_t = torch.tensor([[age, glu, bp]], dtype=torch.float32)

        with t1:
            st.image(raw_img, use_container_width=True)
        
        with t2:
            if st.button("🔥 Run Heatmap Analysis"):
                cam = generate_heatmap(model, img_t, clin_t)
                cam_res = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_res), cv2.COLORMAP_JET)
                result_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                st.image(result_img, caption="AI Focus Areas (Red = High Suspicion)", use_container_width=True)

        # Prediction
        with torch.no_grad():
            prob = model(img_t, clin_t).item()
        
        res_color = "#ff4b4b" if prob > 0.5 else "#00ffcc"
        st.markdown(f"""
            <div class='report-card'>
                <h2 style='color: {res_color} !important;'>RESULT: {'POSITIVE' if prob > 0.5 else 'NEGATIVE'}</h2>
                <p>AI Certainty: {prob*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Waiting for X-ray upload...")