from ultralytics import YOLO
import os

os.environ['YOLO_DATASETS_DIR'] = '/home/emanon/datasets'

# 加载YOLOv8-seg预训练模型（seg=分割版本）
model = YOLO('yolov8n-seg.pt')

results = model.train(
    data='/home/emanon/datasets/ocid_yolo/ocid.yaml',
    epochs=50,
    imgsz=640,
    batch=8,               # seg比detect更吃显存，batch调小
    device=0,
    project='/home/emanon/projects/runs',
    name='exp4_ocid_fulltune',
    workers=0,             # WSL2共享内存问题
    cache=True,
    # 以下是迁移学习关键参数
    pretrained=True,       # 使用预训练权重
    lr0=0.001,             # 学习率比默认小10倍，微调用小学习率
    # freeze=10,             # 冻结前10层backbone，只训练后面的层
)

print(f"训练完成，结果保存在: {results.save_dir}")