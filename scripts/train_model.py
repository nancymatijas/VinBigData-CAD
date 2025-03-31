from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        device = 0  
        print(f"GPU je dostupan: {torch.cuda.get_device_name(device)}")
    else:
        device = 'cpu'
        print("GPU nije dostupan. Trening će se izvršavati na CPU-u.")

    yaml_path = "dataset.yaml"
    model = YOLO("yolo11s.pt")

    results = model.train(
        data=yaml_path,
        device=device,  
        epochs=100,
        imgsz=896,  
        batch=16,  
        freeze=5,   
        lr0=0.0005,
        lrf=0.2,
        optimizer='AdamW',  
        augment=True,  
        workers=4,                 
        save_period=10, 
        patience=5,   
        name="train _yolo11"
    )

    print("Trening završen!")

if __name__ == '__main__':
    main()
