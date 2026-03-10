import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.append('/home/emanon/projects/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# 加载YOLOv8-seg模型
yolo = YOLO('/home/emanon/projects/runs/exp4_ocid_fulltune/weights/best.pt')

# 加载Depth Anything V2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load(
    '/home/emanon/models/depth_anything/depth_anything_v2_vits.pth',
    map_location='cpu'
))
depth_model = depth_model.to('cuda').eval()

# 相机内参（OCID数据集标准估计值）
fx, fy = 570.0, 570.0
cx, cy = 320.0, 240.0

def pixel_to_3d(u, v, depth_map):
    """像素坐标 + 深度图 → 3D坐标"""
    z = float(depth_map[int(v), int(u)])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

# 测试图片
img_path = '/home/emanon/datasets/ocid_yolo/images/val/result_2018-08-21-14-41-35.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 步骤1：YOLOv8检测
results = yolo(img_path, conf=0.6, verbose=False)
boxes = results[0].boxes
masks = results[0].masks

# 步骤2：深度估计
with torch.no_grad():
    depth = depth_model.infer_image(img_rgb)

# 步骤3：对每个检测物体计算3D坐标
print(f"检测到 {len(boxes)} 个物体\n")
positions = []
for i in range(len(boxes)):
    box = boxes[i].xyxy[0].cpu().numpy()
    # mask中心点
    u = (box[0] + box[2]) / 2
    v = (box[1] + box[3]) / 2
    x, y, z = pixel_to_3d(u, v, depth)
    positions.append((u, v, x, y, z))
    print(f"物体{i+1}: 像素({u:.0f},{v:.0f}) → 3D坐标({x:.3f}, {y:.3f}, {z:.3f})")

# 步骤4：可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原图+检测框
annotated = results[0].plot(masks=True)
axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
axes[0].set_title('YOLOv8-seg检测结果')
axes[0].axis('off')

# 深度图
depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
axes[1].imshow(depth_vis, cmap='plasma')
axes[1].set_title('Depth Anything V2深度图')
axes[1].axis('off')

# 叠加3D坐标标注
axes[2].imshow(img_rgb)
for u, v, x, y, z in positions:
    axes[2].plot(u, v, 'r+', markersize=12, markeredgewidth=2)
    axes[2].annotate(
        f'Z={z:.2f}',
        (u, v), textcoords='offset points', xytext=(5, 5),
        fontsize=8, color='yellow',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
    )
axes[2].set_title('3D位置估计结果')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('/home/emanon/projects/depth_3d_result.png', dpi=150, bbox_inches='tight')
print("\n保存完成: depth_3d_result.png")