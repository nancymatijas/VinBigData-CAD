import pydicom
import numpy as np
from PIL import Image
import streamlit as st
import io
from typing import Optional

def dicom_to_image(uploaded_file) -> Optional[Image.Image]:
    try:
        dicom_bytes = uploaded_file.read()
        dicom_stream = io.BytesIO(dicom_bytes)
        dicom_data = pydicom.dcmread(dicom_stream)
        
        pixel_array = dicom_data.pixel_array
        
        if 'WindowWidth' in dicom_data:
            center = dicom_data.WindowCenter
            width = dicom_data.WindowWidth
            pixel_array = np.clip(pixel_array, center - width/2, center + width/2)
        
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack((pixel_array,) * 3, axis=-1)
            
        return Image.fromarray(pixel_array)
    
    except Exception as e:
        st.error(f"Error in conversion to DICOM: {str(e)}")
        return None

