import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/home/emanon/projects/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# 加载模型
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

model = DepthAnythingV2(**model_configs['vits'])
model.load_state_dict(torch.load(
    '/home/emanon/models/depth_anything/depth_anything_v2_vits.pth',
    map_location='cpu'
))
model = model.to('cuda').eval()
print("模型加载成功")

# 选一张OCID图片测试
img_path = '/home/emanon/datasets/ocid_yolo/images/val/result_2018-08-21-14-41-35.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 推理深度图
with torch.no_grad():
    depth = model.infer_image(img_rgb)

print(f"深度图shape: {depth.shape}")
print(f"深度值范围: min={depth.min():.3f}, max={depth.max():.3f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('原始RGB图')
axes[0].axis('off')

depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
axes[1].imshow(depth_vis, cmap='plasma')
axes[1].set_title('深度图（亮=近，暗=远）')
axes[1].axis('off')
plt.colorbar(axes[1].images[0], ax=axes[1], label='相对深度')

plt.tight_layout()
plt.savefig('/home/emanon/projects/depth_result.png', dpi=150, bbox_inches='tight')
print("保存完成: depth_result.png")