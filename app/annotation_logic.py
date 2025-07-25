import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_img_label import st_img_label
from display import resize_image, convert_bboxes_to_display_format, draw_bounding_boxes_with_labels
from ui_components import render_classification_selector, display_annotation_guide
import os
from pdf_writer import generate_pdf_bytes

def interactive_annotation(
    image: Image.Image,
    labels: list[str],
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    bboxes: pd.DataFrame,
    class_colors: dict[str, str]
) -> None:
    bboxes = bboxes if bboxes is not None else pd.DataFrame()
    original_size = image.size
    resized_img = resize_image(image)
    display_annotation_guide()
    session_rects_key = f"rects_{uploaded_file.name}"
    session_labels_key = f"labels_{uploaded_file.name}"
    rects_input = convert_bboxes_to_display_format(bboxes, *original_size) if not bboxes.empty else []
    st.session_state.setdefault(session_rects_key, rects_input)
    st.session_state.setdefault(session_labels_key, [
        rect.get("label", labels[0]) for rect in st.session_state[session_rects_key]
    ])

    with st.form("patient_and_exam_info"):
        st.markdown("### Patient & Exam Information")
        col1, col2 = st.columns(2)
        with col1:
            doctor_name = st.text_input("Ordering Provider", "")
        with col2:
            patient_name = st.text_input("Patient Name", "")
        history = st.text_input("Medical History", "")
        form_submitted = st.form_submit_button("Apply")

    col_preview, col_draw, col_class = st.columns(3)
    with col_draw:
        st.markdown('<div class="section-header">Interactive Validation</div>', unsafe_allow_html=True)
        updated_rects = st_img_label(
            resized_img,
            rects=st.session_state[session_rects_key],
            key=f"annotator_{uploaded_file.name}",
        )
        st.session_state[session_rects_key] = updated_rects
    current_labels = st.session_state[session_labels_key]
    current_labels = current_labels[:len(updated_rects)] + [labels[0]]*(len(updated_rects)-len(current_labels))
    with col_class:
        st.markdown('<div class="section-header">Region Classification</div>', unsafe_allow_html=True)
        if updated_rects:
            current_labels[:] = [
                render_classification_selector(idx, labels, uploaded_file.name, label)
                for idx, label in enumerate(current_labels)
            ]
            if current_labels != st.session_state[session_labels_key]:
                st.session_state[session_labels_key] = current_labels
                st.rerun()
    with col_preview:
        st.markdown('<div class="section-header">Labels Preview</div>', unsafe_allow_html=True)
        if updated_rects:
            annotated_img = draw_bounding_boxes_with_labels(resized_img, updated_rects, current_labels, class_colors)
            st.image(annotated_img, width=450)

    pdf_bytes = generate_pdf_bytes(
        uploaded_file,
        updated_rects,
        current_labels,
        *original_size,
        image=resized_img,
        doctor_name=doctor_name,
        patient_name=patient_name,
        indication=history
    )
    if st.download_button(
        label="💾 Save Annotations (PDF)",
        data=pdf_bytes,
        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.pdf",
        mime="application/pdf",
        use_container_width=True
    ):
        st.success("PDF annotations saved successfully!")
    st.caption(f"Original resolution: {original_size[0]}x{original_size[1]} | Annotations: {len(updated_rects)}")
