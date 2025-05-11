import streamlit as st

def render_classification_selector(index, labels, upload_key, current_label):
    default_index = labels.index(current_label) if current_label in labels else 0
    return st.selectbox(
        f"Class for region {index + 1}",
        labels,
        index=default_index,
        key=f"label_{index}_{upload_key}",
    )

def display_annotation_guide():
    with st.expander("📖 Annotation Guidelines"):
        st.markdown("""
        1. Click and drag to draw a box
        2. Select the appropriate class for each region
        3. Drag edges to adjust precisely
        4. Save annotations when finished
        """)
