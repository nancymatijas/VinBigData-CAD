import os
import pandas as pd
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')

# Podijeli slike na train (80%) i val (20%) skupove
unique_images = df['image_id'].unique()
train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Kreiraj train i val dataframe-ove
train_df = df[df['image_id'].isin(train_images)]
val_df = df[df['image_id'].isin(val_images)]

# Spremi rezultate u zasebne CSV datoteke
os.makedirs('data/splits', exist_ok=True)
train_df.to_csv('data/splits/train_split.csv', index=False)
val_df.to_csv('data/splits/val_split.csv', index=False)

# Kreiraj direktorije za oznake i slike ako ne postoje
os.makedirs('data/labels/train', exist_ok=True)
os.makedirs('data/labels/val', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)

def prepare_data(df, image_source_dir, image_dest_dir, label_dest_dir):
    for image_id, group in df.groupby('image_id'):
        source_image_path = os.path.join(image_source_dir, f"{image_id}.jpg")
        dest_image_path = os.path.join(image_dest_dir, f"{image_id}.jpg")

        if not os.path.exists(source_image_path):
            print(f"Slika {source_image_path} nije pronađena. Preskačem...")
            continue

        if image_source_dir != image_dest_dir:
            shutil.move(source_image_path, dest_image_path)

        with Image.open(dest_image_path) as img:
            img_width, img_height = img.size

        label_path = os.path.join(label_dest_dir, f"{image_id}.txt")
        if group.empty:  # Ako nema bounding boxova za ovu sliku
            open(label_path, 'w').close()  # Kreiraj praznu .txt datoteku
            continue

        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                class_id = row['class_id']
                x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']

                # Provjera bounding boxova
                if pd.isna(x_min) or pd.isna(y_min) or pd.isna(x_max) or pd.isna(y_max):
                    print(f"Preskačem bounding box za sliku {image_id} zbog nedostajućih koordinata.")
                    continue

                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)

# Pripremi skupove (generiraj oznake i premjesti slike)
prepare_data(train_df, 'data/images/train', 'data/images/train', 'data/labels/train')
prepare_data(val_df, 'data/images/train', 'data/images/val', 'data/labels/val')
print("Priprema podataka završena!")
