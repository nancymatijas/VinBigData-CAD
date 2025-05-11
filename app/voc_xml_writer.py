import xml.etree.ElementTree as ET
import os

def save_annotations(uploaded_file, rects, labels, original_width, original_height):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = uploaded_file.name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(original_width)
    ET.SubElement(size, "height").text = str(original_height)
    ET.SubElement(size, "depth").text = "3"

    for rect, label in zip(rects, labels):
        try:
            x_min = int(rect['left'] * original_width / 450)
            y_min = int(rect['top'] * original_height / 450)
            x_max = int((rect['left'] + rect['width']) * original_width / 450)
            y_max = int((rect['top'] + rect['height']) * original_height / 450)

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = label
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(x_min)
            ET.SubElement(bbox, "ymin").text = str(y_min)
            ET.SubElement(bbox, "xmax").text = str(x_max)
            ET.SubElement(bbox, "ymax").text = str(y_max)
        except Exception as e:
            print(f"Greška pri spremanju: {e}")

    xml_path = os.path.join("img", f"{os.path.splitext(uploaded_file.name)[0]}.xml")
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    ET.indent(annotation)
    ET.ElementTree(annotation).write(xml_path)
