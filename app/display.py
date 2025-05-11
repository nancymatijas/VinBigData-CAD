from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes_with_labels(image, rects, labels, class_colors):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, rect in enumerate(rects):
        left = rect['left']
        top = rect['top']
        width = rect['width']
        height = rect['height']
        label = labels[i] if i < len(labels) else ""
        color = class_colors.get(label, "red")

        draw.rectangle([left, top, left + width, top + height], outline=color, width=2)
        draw.text(
            (left + 2, top + 2),
            f"{i+1}: {label}",
            fill=color,
            font=font
        )
    return img


def resize_image(image: Image.Image, target_size: tuple = (450, 450)) -> Image.Image:
    return image.resize(target_size)


def convert_bboxes_to_display_format(bboxes, original_width, original_height, target_size=(450, 450)):
    return [
        {
            "left": row["x_min"] * target_size[0] / original_width,
            "top": row["y_min"] * target_size[1] / original_height,
            "width": (row["x_max"] - row["x_min"]) * target_size[0] / original_width,
            "height": (row["y_max"] - row["y_min"]) * target_size[1] / original_height,
            "label": row["class_name"],
        }
        for _, row in bboxes.iterrows()
    ]


