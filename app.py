import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import LungCNN

st.set_page_config(
    page_title="LungCancerDetection",
    page_icon="🩻",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display:ital@0;1&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #080e1c;
        color: #dde3f0;
    }

    section[data-testid="stFileUploadDropzone"] {
        background-color: #0e1728;
        border: 1.5px dashed #1e3a5f;
        border-radius: 14px;
        transition: border-color 0.2s;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #2a6496;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #1a2a42;
    }

    .verdict-wrap {
        background: #0e1728;
        border: 1px solid #1a2a42;
        border-radius: 16px;
        padding: 28px 26px;
        margin-top: 8px;
    }

    .verdict-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4a6080;
        margin-bottom: 10px;
    }

    .verdict-class {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        line-height: 1.2;
        margin-bottom: 6px;
    }

    .verdict-class.benign    { color: #34d399; }
    .verdict-class.malignant { color: #fb7185; }
    .verdict-class.normal    { color: #60a5fa; }

    .verdict-conf {
        font-size: 13px;
        color: #4a6080;
        margin-bottom: 18px;
    }

    .verdict-conf span {
        font-weight: 500;
        color: #dde3f0;
    }

    .conf-track {
        height: 6px;
        background: #141f35;
        border-radius: 99px;
        overflow: hidden;
        margin-bottom: 20px;
    }

    .conf-fill-benign    { height: 100%; background: #34d399; border-radius: 99px; }
    .conf-fill-malignant { height: 100%; background: #fb7185; border-radius: 99px; }
    .conf-fill-normal    { height: 100%; background: #60a5fa; border-radius: 99px; }

    .verdict-msg {
        font-size: 13px;
        line-height: 1.65;
        padding: 12px 14px;
        border-radius: 10px;
    }

    .msg-benign    { background: rgba(52,211,153,0.08); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.2); }
    .msg-malignant { background: rgba(251,113,133,0.08); color: #fda4af; border: 1px solid rgba(251,113,133,0.2); }
    .msg-normal    { background: rgba(96,165,250,0.08);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.2); }

    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #dde3f0;
        margin-bottom: 4px;
    }

    .header-sub {
        font-size: 13px;
        color: #4a6080;
        margin-bottom: 0;
    }

    .header-tag {
        display: inline-block;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2a8ab0;
        background: rgba(42,138,176,0.12);
        border: 1px solid rgba(42,138,176,0.25);
        padding: 3px 10px;
        border-radius: 99px;
        margin-bottom: 12px;
    }

    .disclaimer {
        font-size: 12px;
        color: #3d5070;
        text-align: center;
        padding: 12px 0 4px;
    }

    hr {
        border-color: #111e33 !important;
    }

    div[data-testid="stSpinner"] { color: #2a8ab0 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Benign cases", "Malignant cases", "Normal cases"]

CLASS_META = {
    "Benign cases": {
        "key":   "benign",
        "short": "Benign",
        "msg":   "No malignant cells detected. The scan shows benign characteristics. Regular monitoring is still recommended.",
    },
    "Malignant cases": {
        "key":   "malignant",
        "short": "Malignant",
        "msg":   "Malignant characteristics detected. Please consult an oncologist immediately for further evaluation.",
    },
    "Normal cases": {
        "key":   "normal",
        "short": "Normal",
        "msg":   "The scan appears normal with no abnormalities detected. Continue with routine check-ups.",
    },
}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = LungCNN(num_classes=3)
    mdl.load_state_dict(torch.load("model.pth", map_location=device))
    mdl.to(device)
    mdl.eval()
    return mdl, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict(image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    return CLASS_NAMES[pred_idx], round(float(probs[pred_idx]) * 100, 2)


st.markdown('<div class="header-tag">IQ-OTHNCCD Dataset · PyTorch CNN</div>', unsafe_allow_html=True)
st.markdown('<div class="header-title">Lung Cancer Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Upload a CT scan — the model will detect what type of lung condition is present.</div>', unsafe_allow_html=True)

st.write("")
st.divider()

uploaded_file = st.file_uploader(
    "Drop a scan image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            pred_class, confidence = predict(image)

        meta = CLASS_META[pred_class]
        key  = meta["key"]

        st.markdown(f"""
        <div class="verdict-wrap">
            <div class="verdict-label">Detected condition</div>
            <div class="verdict-class {key}">{meta['short']}</div>
            <div class="verdict-conf">Confidence &nbsp;<span>{confidence:.1f}%</span></div>
            <div class="conf-track">
                <div class="conf-fill-{key}" style="width:{confidence}%"></div>
            </div>
            <div class="verdict-msg msg-{key}">{meta['msg']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()
    st.markdown('<div class="disclaimer">⚠ Research use only — not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)
