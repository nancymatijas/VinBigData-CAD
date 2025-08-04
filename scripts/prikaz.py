import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from PIL import Image

# Učitavanje slike (JPEG, a ne DICOM!)
image_path = "../data/images/val/0005e8e3701dfb1dd93d53e2ff537b6e.jpg"
image = Image.open(image_path)
image_np = np.array(image)

# Učitavanje anotacija za tu sliku iz CSV-a
df = pd.read_csv("../data/train.csv")
bboxes = df[df["image_id"] == "0005e8e3701dfb1dd93d53e2ff537b6e"]

# Rječnik boja za klase
class_colors = {
    "Aortic enlargement": "#FF5B33",
    "Atelectasis": "#FFA233",
    "Calcification": "#AABF3B",
    "Cardiomegaly": "#3BB349",
    "Consolidation": "#2EA39E",
    "ILD": "#347EEA",
    "Infiltration": "#8134EA",
    "Lung Opacity": "#B634EA",
    "Nodule/Mass": "#EA359E",
    "Other lesion": "#EA6A35",
    "Pleural effusion": "#33B3B5",
    "Pleural thickening": "#854C2B",
    "Pneumothorax": "#FBC4FF",
    "Pulmonary fibrosis": "#6F6F6E",
    "No finding": "#BFE5C3",
}

# Prikaz slike i pripadnih bounding boxova
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_np, cmap='gray')

for _, row in bboxes.iterrows():
    x_min = int(row['x_min'])
    y_min = int(row['y_min'])
    x_max = int(row['x_max'])
    y_max = int(row['y_max'])
    label = row['class_name']
    color = class_colors.get(label, 'red')
    width = x_max - x_min
    height = y_max - y_min

    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x_min, y_min - 5, label, color=color, fontsize=12,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

ax.axis('off')
plt.tight_layout()
plt.show()
