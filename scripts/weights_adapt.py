import pandas as pd
import json
import os
import shutil

train_csv_path = "data/splits/train_split.csv"
train_df = pd.read_csv(train_csv_path)

class_counts = train_df['class_id'].value_counts()
total_instances = len(train_df[train_df['class_id'] != 14])

class_weights = {int(class_id): max(0.1, total_instances / count) for class_id, count in class_counts.items() if class_id != 14}
class_weights[14] = 0.1

max_samples_per_class = 500
filtered_df = train_df.groupby('class_id').apply(lambda x: x.sample(min(len(x), max_samples_per_class), random_state=42)).reset_index(drop=True)

filtered_train_dir = "data/images/train_filtered"
filtered_labels_dir = "data/labels/train_filtered"
os.makedirs(filtered_train_dir, exist_ok=True)
os.makedirs(filtered_labels_dir, exist_ok=True)

for image_id in filtered_df['image_id'].unique():
    # Kopiranje slike
    src_image = f"data/images/train/{image_id}.jpg"
    dst_image = f"{filtered_train_dir}/{image_id}.jpg"
    if os.path.exists(src_image):
        shutil.copy(src_image, dst_image)
    else:
        print(f"Slika {src_image} nije pronađena. Preskačem...")

    # Kopiranje oznake
    src_label = f"data/labels/train/{image_id}.txt"
    dst_label = f"{filtered_labels_dir}/{image_id}.txt"
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
    else:
        print(f"Oznaka {src_label} nije pronađena. Kreiram praznu...")
        open(dst_label, 'w').close()  # Prazna datoteka ako nema oznaka

weights_path = "data/class_weights.json"
with open(weights_path, 'w') as f: 
    json.dump(class_weights, f, indent=4)

print("Filtrirani dataset spremljen u:")
print(f"Slike: {filtered_train_dir}")
print(f"Oznake: {filtered_labels_dir}")
print("Težine za klase:", class_weights)