# VinBigData-CAD

This application is developed as a test project using **Streamlit** and **Python**. It leverages the **YOLOv8** model from Ultralytics for AI-powered object detection on chest X-ray DICOM images. Interactive annotations and PDF report generation enhance usability for medical professionals and researchers.

---

## Features

- **DICOM Image Handling**  
  - Upload and visualize posteroanterior chest X-ray DICOM images  
  - Automatic windowing and image normalization for better visualization  

- **AI-Powered Detection**  
  - Detect clinical findings using a custom-trained YOLOv8 model  
  - Confidence scoring and class prediction for radiological anomalies  

- **Interactive Annotations**  
  - Draw and adjust bounding boxes on images  
  - Select classification labels with predefined color codes  
  - Patient and exam information form for contextual metadata  

- **PDF Report Generation**  
  - Generate detailed reports with annotated images, findings, and patient data  
  - Customizable and professional PDF layout  

- **User Guidance**  
  - Clear annotation guidelines included in the UI  

---

## Installation

1. Clone the repository:
```
git clone <repository_url>
cd <repository_directory>
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit application:
```
streamlit run main.py
```

---

## Usage

1. Upload a DICOM chest X-ray file via the sidebar uploader.
2. The image is displayed with AI-detected bounding boxes and labels.
3. Use the interactive annotation tools to adjust or add new bounding boxes.
4. Enter patient and ordering provider details in the provided form.
5. Generate and download a comprehensive PDF report of the findings and annotations.

---