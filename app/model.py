from ultralytics import YOLO
import streamlit as st

def load_model():
    try:
        return YOLO("../runs/detect/train_yolo11s/weights/best.pt")
    except Exception as e:
        st.error(f"Error in loading model: {e}")
        return None
