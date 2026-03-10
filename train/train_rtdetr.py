from ultralytics import YOLO
import os

os.environ['YOLO_DATASETS_DIR'] = '/home/emanon/datasets'

# RT-DETR用YOLO库加载
model = YOLO('rtdetr-l.pt')

results = model.train(
    data='coco128.yaml',
    epochs=50,
    imgsz=640,
    batch=4,               # RT-DETR更吃显存，batch调小到4
    device=0,
    project='/home/emanon/projects/runs',
    name='exp2_rtdetr',
    workers=0,             # WSL2共享内存问题，保持0
    cache=True,
)

print(f"训练完成，结果保存在: {results.save_dir}")