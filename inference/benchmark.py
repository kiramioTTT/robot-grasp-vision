import sys
import cv2
import torch
import numpy as np
import time

sys.path.append('/home/emanon/projects/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

yolo = YOLO('/home/emanon/projects/runs/exp4_ocid_fulltune/weights/best.pt')

cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
depth_model = DepthAnythingV2(**cfg)
depth_model.load_state_dict(torch.load(
    '/home/emanon/models/depth_anything/depth_anything_v2_vits.pth',
    map_location='cpu', weights_only=False
))
depth_model = depth_model.to('cuda').eval()

img_path = '/home/emanon/datasets/ocid_yolo/images/val/result_2018-08-21-14-41-35.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 预热GPU
for _ in range(3):
    yolo(img, verbose=False)
    with torch.no_grad():
        depth_model.infer_image(img_rgb)

# 测量各部分耗时（跑20次取平均）
N = 20
yolo_times, depth_times, total_times = [], [], []

for _ in range(N):
    start = time.time()
    yolo(img, verbose=False)
    yolo_times.append(time.time() - start)

    start = time.time()
    with torch.no_grad():
        depth_model.infer_image(img_rgb)
    depth_times.append(time.time() - start)

    start = time.time()
    yolo(img, verbose=False)
    with torch.no_grad():
        depth_model.infer_image(img_rgb)
    total_times.append(time.time() - start)

print(f"YOLOv8-seg推理:     {np.mean(yolo_times)*1000:.1f}ms")
print(f"Depth Anything推理: {np.mean(depth_times)*1000:.1f}ms")
print(f"端到端总耗时:        {np.mean(total_times)*1000:.1f}ms")
print(f"端到端FPS:           {1/np.mean(total_times):.1f}")
