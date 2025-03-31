from ultralytics import YOLO
import torch
from torchvision import models
from torch import nn

class CustomYOLOWithResNet(nn.Module):
    def __init__(self, yolov8_model, resnet_backbone):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])  
        self.yolo_layers = yolov8_model.model.model  
        self.backbone_end_idx = 9
        self.adapter = nn.Conv2d(2048, 1024, kernel_size=1)  # Usklađivanje s yolov8s
        self.yaml = yolov8_model.model.yaml  

    def forward(self, x):
        x = self.backbone(x)  # [batch, 2048, H/32, W/32]
        x = self.adapter(x)   # [batch, 1024, H/32, W/32]
        
        for i, layer in enumerate(self.yolo_layers):
            if i >= self.backbone_end_idx:
                x = layer(x)
        return x

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Koristi se: {torch.cuda.get_device_name(device) if device == 0 else 'CPU'}")

    resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
    
    yaml_path = "dataset.yaml"
    yolov8 = YOLO("yolov8s.pt")
    model = CustomYOLOWithResNet(yolov8, resnet)
    yolov8.model = model

    results = yolov8.train(
        data=yaml_path,
        device=device,
        epochs=150,
        imgsz=640,
        batch=32,
        freeze=5,    
        lr0=0.005,
        lrf=0.2,
        optimizer='AdamW',
        augment=True,
        mosaic=True,
        mixup=True,
        save_period=10,
        patience=5,
        name="vinbigdata_yolo8s_resnet"
    )

    print("Trening završen!")

if __name__ == '__main__':
    main()
