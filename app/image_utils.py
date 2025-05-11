from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from constants import class_colors

def draw_bounding_boxes(image: Image.Image, boxes: pd.DataFrame, font_size: int = 45) -> Image.Image:
    if boxes.empty:
        return image

    draw = ImageDraw.Draw(image)
    font = _load_font(font_size)
    
    for _, row in boxes.iterrows():
        if _is_valid_bbox(row, image.size):
            _draw_single_box(draw, row, font)
    
    return image

def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        return ImageFont.load_default()

def _is_valid_bbox(row: pd.Series, img_size: tuple) -> bool:
    coords = ['x_min', 'y_min', 'x_max', 'y_max']
    if row[coords].isnull().any():
        return False
    x_min, y_min, x_max, y_max = (row[c] for c in coords)
    width, height = img_size
    if not (0 <= x_min < x_max <= width):
        return False
    if not (0 <= y_min < y_max <= height):
        return False
    return True

def _draw_single_box(draw: ImageDraw, row: pd.Series, font: ImageFont):
    x_min, y_min, x_max, y_max = map(int, row[['x_min', 'y_min', 'x_max', 'y_max']])
    class_name = row.get('class_name', 'Unknown')
    color = class_colors.get(class_name, "black")
    
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
    
    label = class_name
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    bg_position = [
        (x_min, y_min - text_height - 5),
        (x_min + text_width + 10, y_min)
    ]
    
    draw.rectangle(bg_position, fill=color)
    draw.text((x_min + 5, y_min - text_height - 2), label, fill="white", font=font)
