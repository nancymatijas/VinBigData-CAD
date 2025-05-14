from ultralytics import YOLO

def evaluate_model():
    model = YOLO("../runs/detect/train_yolo11s/weights/best.pt") 

    results = model.val(
        data="../dataset.yaml",  
        imgsz=896,  
        batch=16,   
        conf=0.15,
        iou=0.4,
        name="val"
    )

    precision = results.box.p.mean() if results.box.p.size > 1 else results.box.p.item()
    recall = results.box.r.mean() if results.box.r.size > 1 else results.box.r.item()
    print(f"Precision: {precision:.4f}") 
    print(f"Recall: {recall:.4f}")         
    print(f"mAP50: {results.box.map50:.4f}") 
    print(f"mAP50-95: {results.box.map:.4f}")  

if __name__ == "__main__":
    evaluate_model()
