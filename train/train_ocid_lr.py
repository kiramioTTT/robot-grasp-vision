from ultralytics import YOLO
import os

os.environ['YOLO_DATASETS_DIR'] = '/home/emanon/datasets'

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data='/home/emanon/datasets/ocid_yolo/ocid.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    project='/home/emanon/projects/runs',
    name='exp5_ocid_lr01',
    workers=0,
    cache=True,
    pretrained=True,
    lr0=0.01,      # 学习率从0.001改成0.01，大10倍
)

print(f"训练完成，结果保存在: {results.save_dir}")