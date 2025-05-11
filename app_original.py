import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import pydicom
import logging
#logging.getLogger("streamlit").setLevel(logging.ERROR)

class_colors = {
    "Aortic enlargement": "blue",
    "Atelectasis": "green",
    "Calcification": "orange",
    "Cardiomegaly": "#8300d5",          #purple
    "Consolidation": "#df04a8",         #dark pink
    "ILD": "brown",
    "Infiltration": "#5b5b5b",          #dark gray
    "Lung Opacity": "#008b8b",          #dark cyan
    "Nodule/Mass": "red",
    "Other lesion": "#f1c40f",          #dark yellow
    "Pleural effusion": "#333300",      #olive
    "Pleural thickening": "#40597b",    #blue gray
    "Pneumothorax": "#9b0074",          #dark magenta
    "Pulmonary fibrosis": "maroon"
}

def load_model():
    try:
        return YOLO("runs/detect/train_yolo11s/weights/best.pt")
    except Exception as e:
        st.error(f"Greška pri učitavanju modela: {e}")
        return None

def dicom_to_image(dicom_path):
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack((pixel_array,) * 3, axis=-1)
        return Image.fromarray(pixel_array)
    except Exception as e:
        st.error(f"Greška pri konverziji DICOM-a: {e}")
        return None

def draw_bounding_boxes(image, boxes, font_size=45):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for _, row in boxes.iterrows():
        if pd.isnull(row[['x_min', 'y_min', 'x_max', 'y_max']]).any():
            continue
        x_min, y_min, x_max, y_max = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
        class_name = row.get('class_name', 'Unknown')
        color = class_colors.get(class_name, "black")

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        label = f"{class_name}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        draw.rectangle([(x_min, y_min - text_height - 5), (x_min + text_width + 10, y_min)], fill=color)
        draw.text((x_min + 5, y_min - text_height - 2), label, fill="white", font=font)
    return image

def predict_dicom(image, model):
    try:
        results = model.predict(np.array(image))
        predictions = [
            {"x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], 
             "confidence": box[4], "class_name": model.names[int(box[5])]}
            for result in results for box in result.boxes.data.cpu().numpy()
        ]
        return pd.DataFrame(predictions)
    except Exception as e:
        st.error(f"Greška pri predikciji: {e}")
        return None

st.sidebar.title("DICOM Viewer")
uploaded_file = st.sidebar.file_uploader("Upload a DICOM file", type=["dicom"])
real_boxes_df = pd.read_csv("data/train.csv") if pd.io.common.file_exists("data/train.csv") else pd.DataFrame()

if uploaded_file:
    dicom_image = dicom_to_image(uploaded_file)
    if dicom_image:
        image_id = uploaded_file.name.split('.')[0]
        real_boxes = real_boxes_df[real_boxes_df['image_id'] == image_id]

        if not real_boxes.empty and not real_boxes['class_name'].eq("No finding").all():
            real_boxes[['x_min', 'y_min', 'x_max', 'y_max']] = real_boxes[['x_min', 'y_min', 'x_max', 'y_max']].fillna(0).astype(int)
            st.write("Stvarne vrijednosti:")
            st.table(real_boxes[['class_name', 'x_min', 'y_min', 'x_max', 'y_max']])
            original_with_real_boxes = draw_bounding_boxes(dicom_image.copy(), real_boxes)
        else:
            original_with_real_boxes = dicom_image.copy()
            if not real_boxes.empty:
                st.info("Stvarne vrijednosti su samo 'No finding'.")

        model = load_model()
        if model:
            predicted_boxes_df = predict_dicom(dicom_image, model)
            if predicted_boxes_df is not None and not predicted_boxes_df.empty:
                if not predicted_boxes_df['class_name'].eq("No finding").all():
                    st.write("Rezultati predikcije:")
                    st.table(predicted_boxes_df[['class_name', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']])
                    annotated_image_with_predictions = draw_bounding_boxes(dicom_image.copy(), predicted_boxes_df)

                    col1, col2 = st.columns(2)
                    col1.image(original_with_real_boxes, caption="Originalna slika sa stvarnim bounding boxovima", clamp=True)
                    col2.image(annotated_image_with_predictions, caption="Slika s predviđenim bounding boxovima", clamp=True)
                else:
                    st.info("Model je detektirao samo 'No finding'.")
            else:
                st.warning("Model nije pronašao nikakve objekte.")