import streamlit as st
import torch
import timm
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from fpdf import FPDF
import tempfile
import datetime

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.markdown('<p class="main-title">🦠 Malaria Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered detection of parasitized blood cells</p>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        text-align:center;
        color:#2c7be5;
    }

    .subtitle {
        font-size:18px;
        text-align:center;
        color:gray;
    }

    .card {
        padding:20px;
        border-radius:10px;
        background-color:#f5f7fa;
        text-align:center;
        box-shadow:0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🦠 Malaria Detection System")
st.markdown("Upload a **blood cell image** to detect malaria infection using AI.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("About")
st.sidebar.write(
"""
This system detects **Parasitized and Uninfected cells**
using a **Vision Transformer deep learning model**.

Features:
- Image classification
- Confidence score
- Explainable AI heatmap
- Downloadable medical report
"""
)

st.sidebar.info("Model: Vision Transformer (ViT)")

# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cpu")

model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("malaria_vit_model.pth", map_location=device))
model.eval()

classes = ["Parasitized", "Uninfected"]

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# GRADCAM TRANSFORM
# -----------------------------
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:,1:,:].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0,3,1,2)
    return result

# -----------------------------
# PDF REPORT
# -----------------------------
def generate_pdf(image, heatmap, prediction, confidence):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200,10,"Malaria Detection Report", ln=True, align="C")

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf.set_font("Arial", size=12)

    pdf.ln(5)
    pdf.cell(200,10,f"Date: {date}", ln=True)
    pdf.cell(200,10,f"Prediction: {prediction}", ln=True)
    pdf.cell(200,10,f"Confidence: {confidence:.2f}%", ln=True)

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    heatmap_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    image.save(img_path)
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    pdf.ln(5)
    pdf.cell(200,10,"Input Image:", ln=True)
    pdf.image(img_path, x=10, w=60)

    pdf.ln(45)
    pdf.cell(200,10,"Heatmap:", ln=True)
    pdf.image(heatmap_path, x=10, w=60)

    report_path = "malaria_report.pdf"
    pdf.output(report_path)

    return report_path

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Blood Cell Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=350)

    with st.spinner("Analyzing image..."):

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs,1)

    prediction = classes[pred.item()]
    confidence_score = confidence.item()*100

    with col2:

        st.subheader("Prediction Result")

        if prediction == "Parasitized":
            st.error(prediction)
        else:
            st.success(prediction)

        st.metric(
            label="Confidence",
            value=f"{confidence_score:.2f}%"
        )

    colA, colB = st.columns(2)

    with colA:
        st.markdown(
            f"""
            <div class="card">
            <h3>Parasitized Probability</h3>
            <h2>{probs[0][0].item()*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with colB:
        st.markdown(
            f"""
            <div class="card">
            <h3>Uninfected Probability</h3>
            <h2>{probs[0][1].item()*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Probability")

    prob_data = {
        "Parasitized": probs[0][0].item(),
        "Uninfected": probs[0][1].item()
    }

    st.bar_chart(prob_data)

    # -----------------------------
    # GRADCAM
    # -----------------------------
    st.subheader("Explainable AI Heatmap")

    rgb_img = np.array(image.resize((224,224))) / 255.0

    target_layer = model.blocks[-1].norm1

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.image(heatmap, caption="Model Attention Heatmap", width=350)

    # -----------------------------
    # PDF REPORT
    # -----------------------------
    report_file = generate_pdf(image, heatmap, prediction, confidence_score)

    with open(report_file, "rb") as file:
        st.download_button(
            label="Download PDF Report",
            data=file,
            file_name="malaria_report.pdf",
            mime="application/pdf"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
"""
### About this project
This AI system detects malaria parasites in microscopic blood cell images using
a **Vision Transformer deep learning model**.

Developed for medical image analysis and AI-assisted diagnosis.
"""
)

#streamlit run malaria_app.py