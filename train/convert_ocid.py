import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# 路径配置
OCID_ROOT = Path('/home/emanon/datasets/OCID-dataset/ARID20')
OUTPUT_ROOT = Path('/home/emanon/datasets/ocid_yolo')

def convert_label(label_path):
    """把OCID的label PNG转换为YOLO分割格式"""
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        return []

    h, w = label.shape
    yolo_labels = []

    # 获取所有物体ID（0是背景，跳过）
    obj_ids = np.unique(label)
    obj_ids = obj_ids[obj_ids != 0]

    for obj_id in obj_ids:
        # 提取该物体的mask
        mask = (label == obj_id).astype(np.uint8)

        # 找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # 取最大轮廓
        contour = max(contours, key=cv2.contourArea)

        # 轮廓面积太小跳过（噪声）
        if cv2.contourArea(contour) < 100:
            continue

        # 简化轮廓点数（最多32个点）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 归一化坐标
        points = approx.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.extend([x / w, y / h])

        if len(normalized) >= 6:  # 至少3个点才是有效多边形
            # 所有物体统一用class 0（可抓取物体）
            label_str = '0 ' + ' '.join([f'{v:.6f}' for v in normalized])
            yolo_labels.append(label_str)

    return yolo_labels


def main():
    # 创建输出目录
    for split in ['train', 'val']:
        (OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 收集所有图片路径
    all_rgb = sorted(OCID_ROOT.rglob('rgb/*.png'))
    print(f"找到 {len(all_rgb)} 张图片")

    # 划分训练集/验证集 8:2
    split_idx = int(len(all_rgb) * 0.8)
    train_imgs = all_rgb[:split_idx]
    val_imgs = all_rgb[split_idx:]

    print(f"训练集: {len(train_imgs)} 张，验证集: {len(val_imgs)} 张")

    # 转换并保存
    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        print(f"\n处理 {split} 集...")
        success, skip = 0, 0

        for rgb_path in tqdm(imgs):
            # 找对应label
            label_path = rgb_path.parent.parent / 'label' / rgb_path.name
            if not label_path.exists():
                skip += 1
                continue

            # 转换label
            yolo_labels = convert_label(label_path)
            if not yolo_labels:
                skip += 1
                continue

            # 保存图片
            img_name = rgb_path.stem + '.jpg'
            dst_img = OUTPUT_ROOT / 'images' / split / img_name
            img = cv2.imread(str(rgb_path))
            cv2.imwrite(str(dst_img), img)

            # 保存label
            label_name = rgb_path.stem + '.txt'
            dst_label = OUTPUT_ROOT / 'labels' / split / label_name
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_labels))

            success += 1

        print(f"{split}: 成功{success}张，跳过{skip}张")

    # 生成yaml配置文件
    yaml_content = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val
nc: 1
names:
  0: graspable_object
"""
    with open(OUTPUT_ROOT / 'ocid.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n完成！数据集保存在: {OUTPUT_ROOT}")
    print(f"yaml配置文件: {OUTPUT_ROOT / 'ocid.yaml'}")


if __name__ == '__main__':
    main()