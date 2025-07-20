from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        device = 0  
        print(f"GPU is available: {torch.cuda.get_device_name(device)}")
    else:
        device = 'cpu'
        print("GPU is not available. Training will be performed on the CPU.")

    yaml_path = "../dataset.yaml"
    model = YOLO("yolo11s.pt")

    results = model.train(
        data=yaml_path,
        device=device,  
        epochs=150,
        imgsz=896,  
        batch=16,  
        freeze=5,   
        lr0=0.0005,
        lrf=0.2,
        optimizer='AdamW',  
        augment=True,  
        workers=4,                 
        save_period=10, 
        patience=10,   
        name="train_yolo11"
    )

    print("Training completed!")

if __name__ == '__main__':
    main()
