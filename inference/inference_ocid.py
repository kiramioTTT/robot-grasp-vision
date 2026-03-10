from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

model = YOLO('/home/emanon/projects/runs/exp4_ocid_fulltune/weights/best.pt')

img_dir = '/home/emanon/datasets/ocid_yolo/images/val'
imgs = os.listdir(img_dir)[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, img_name in enumerate(imgs):
    img_path = os.path.join(img_dir, img_name)
    results = model(img_path, conf=0.6, verbose=False)

    # 读取原图
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 叠加mask
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for j, mask in enumerate(masks):
            # 每个物体用不同颜色
            color = np.array([
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            ])
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            img_rgb[mask_resized > 0.5] = img_rgb[mask_resized > 0.5] * 0.5 + color * 0.5

    axes[i].imshow(img_rgb)
    axes[i].set_title(f'{img_name[:20]}... 检测到{len(results[0].boxes)}个物体')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/emanon/projects/mask_visualization.png', dpi=150, bbox_inches='tight')
print("保存完成: mask_visualization.png")