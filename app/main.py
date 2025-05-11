import os
import streamlit as st
import pandas as pd
from dicom_utils import dicom_to_image
from image_utils import draw_bounding_boxes
from model_utils import predict_dicom
from annotation_logic import interactive_annotation
from model import load_model
from constants import class_colors

st.set_page_config(
    page_title="Medical Imaging Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("css/style.css")

st.sidebar.markdown(
    '''
    <div class="sidebar-card">
        <img src="https://img.icons8.com/ios-filled/100/1d3557/medical-doctor.png" width="65" style="display:block; margin:auto;"/>
        <div class="sidebar-title">Medical Imaging Analyzer</div>
    </div>
    ''',
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "",
    type=["dicom"],
    help="Limit 200MB per file • DICOM",
    key="dicom_upload"
)

st.markdown('<div class="title">Web Application for Computer-Aided Diagnosis and Radiology Report Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This application enables automatic detection and visualization of anomalies in radiological images using state-of-the-art deep neural network architectures for object detection.</div>', unsafe_allow_html=True)
st.markdown("")

csv_path = "../data/train.csv"
real_boxes_df = pd.DataFrame()
if os.path.exists(csv_path):
    try:
        real_boxes_df = pd.read_csv(csv_path)
    except Exception as e:
        st.sidebar.warning(f"Reference data loading issue: {str(e)}")

if uploaded_file:
    with st.spinner("Processing medical imaging..."):
        dicom_image = dicom_to_image(uploaded_file)

    if dicom_image is not None:
        model = load_model()
        predicted_boxes_df = predict_dicom(dicom_image, model)
        if predicted_boxes_df is not None and not predicted_boxes_df.empty:
            detection_image = draw_bounding_boxes(dicom_image.copy(), predicted_boxes_df)
        st.subheader("Interactive Validation Interface")
        interactive_annotation(
            dicom_image,
            list(class_colors.keys()),
            uploaded_file,
            predicted_boxes_df if predicted_boxes_df is not None else pd.DataFrame(),
            class_colors
        )

else:
    st.markdown("""
    <div class="welcome-message">
        <h3>Instructions</h3>
        <ul>
            <li>Upload DICOM scans for automated anomaly detection using deep learning.</li>
            <li>Compare AI predictions with clinical reference annotations.</li>
            <li>Edit and validate detected findings interactively.</li>
            <li>Generate structured radiology reports based on detected and edited anomalies.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
