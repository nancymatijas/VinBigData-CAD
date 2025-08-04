from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import datetime

def wrap_text_to_width(canvas, text, max_width, fontname="Courier", fontsize=12):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = current + (' ' if current else '') + word
        if canvas.stringWidth(test, fontname, fontsize) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def generate_pdf_bytes(
    uploaded_file, rects, labels, original_width, original_height, image=None,
    indication="", doctor_name="", patient_name=""
):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    mono_bold = "Courier-Bold"
    mono = "Courier"

    c.setFont(mono_bold, 14)
    c.drawString(40, height - 40, "Report Status: Final")
    c.drawString(40, height - 60, "Type: Chest View")
    c.setFont(mono, 12)
    c.drawString(40, height - 80, f"Date/Time: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(40, height - 120, f"Ordering Provider: {doctor_name}")
    c.drawString(40, height - 140, f"Patient Name: {patient_name}")

    y = height - 170
    c.setFont(mono_bold, 12)
    c.drawString(40, y, "HISTORY:")
    c.setFont(mono, 12)
    c.drawString(130, y, indication)

    y -= 30
    c.setFont(mono_bold, 14)
    c.drawString(40, y, "REPORT: Posteroanterior chest X-ray view")

    y -= 20
    c.setFont(mono_bold, 12)
    c.drawString(40, y, "FINDINGS:")
    c.setFont(mono, 12)
    findings = [l for l in labels if l != "No finding"]
    if not findings:
        findings_text = (
            "The lungs are well inflated and clear. No evidence of abnormalities is present."
        )
    elif len(findings) == 1:
        findings_text = f"There is radiographic evidence of: {findings[0]}."
    else:
        findings_text = (
            "Radiographic findings include: " + ", ".join(sorted(set(findings))) + "."
        )
    max_content_width = 500
    y_findings = y - 17
    lines = wrap_text_to_width(c, findings_text, max_content_width, fontname=mono, fontsize=12)
    for line in lines:
        c.drawString(55, y_findings, line)
        y_findings -= 16
    y = y_findings

    c.setFont(mono_bold, 12)
    c.drawString(40, y, "IMPRESSIONS:")
    c.setFont(mono, 12)
    if not findings:
        impression_text = "No radiographic signs of disease."
    elif len(findings) == 1:
        impression_text = f"Imaging findings are consistent with: {findings[0]}."
    else:
        impression_text = (
            "Imaging findings are consistent with the following: " +
            ", ".join(sorted(set(findings))) + "."
        )
    y_impressions = y - 17
    lines = wrap_text_to_width(c, impression_text, max_content_width, fontname=mono, fontsize=12)
    for line in lines:
        c.drawString(55, y_impressions, line)
        y_impressions -= 16
    y = y_impressions


    if image is not None:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img_width, img_height = 250, 250
        x_img = int((width - img_width) / 2)
        y_img = y - img_height - 15
        c.drawImage(ImageReader(img_buffer), x_img, y_img, width=img_width, height=img_height, preserveAspectRatio=True, anchor='n')
        y = y_img - 30

    if rects and labels:
        c.setFont(mono_bold, 12)
        c.drawString(40, y, "ANNOTATIONS:")
        c.setFont(mono, 10)
        box_y = y - 20
        for idx, (rect, label) in enumerate(zip(rects, labels), 1):
            x_min = int(rect['left'] * original_width / 450)
            y_min = int(rect['top'] * original_height / 450)
            x_max = int((rect['left'] + rect['width']) * original_width / 450)
            y_max = int((rect['top'] + rect['height']) * original_height / 450)
            region_line = f"{idx}. {label}: ({x_min}, {y_min}) – ({x_max}, {y_max})"
            region_lines = wrap_text_to_width(c, region_line, max_content_width, fontname=mono, fontsize=10)
            for rline in region_lines:
                if box_y < 50:
                    c.showPage()
                    box_y = height - 70
                c.drawString(55, box_y, rline)
                box_y -= 14

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
