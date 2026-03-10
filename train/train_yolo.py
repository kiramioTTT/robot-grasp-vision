from ultralytics import YOLO
import os

os.environ['YOLO_DATASETS_DIR'] = '/home/emanon/datasets'

model = YOLO('yolov8n.pt')

results = model.train(
    data='coco128.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project='/home/emanon/projects/runs',
    name='exp1_yolov8n',
    workers=0,       # 关闭多进程加载，解决WSL2共享内存问题
    cache=True,      # 把数据缓存到内存，不依赖多进程
)

print(f"训练完成，结果保存在: {results.save_dir}")