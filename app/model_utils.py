import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
from typing import Optional

def predict_dicom(image: Image.Image, model) -> Optional[pd.DataFrame]:
    try:
        img_array = np.array(image)
        
        results = model.predict(img_array)
        
        predictions = []
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes.data.cpu().numpy():
                if len(box) < 6:
                    continue
                    
                predictions.append({
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                    "confidence": box[4],
                    "class_name": model.names[int(box[5])]
                })
        
        if not predictions:
            #st.warning("No clinical findings detected.")
            return None
            
        return pd.DataFrame(predictions)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None
