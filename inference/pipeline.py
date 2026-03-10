import sys
import cv2
import torch
import numpy as np
import time
import argparse

sys.path.append('/home/emanon/projects/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# ─── 模型加载 ───────────────────────────────────────────
def load_models():
    print("加载YOLOv8-seg...")
    yolo = YOLO('/home/emanon/projects/runs/exp4_ocid_fulltune/weights/best.pt')

    print("加载Depth Anything V2...")
    cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    depth_model = DepthAnythingV2(**cfg)
    depth_model.load_state_dict(torch.load(
        '/home/emanon/models/depth_anything/depth_anything_v2_vits.pth',
        map_location='cpu', weights_only=False
    ))
    depth_model = depth_model.to('cuda').eval()
    print("模型加载完成\n")
    return yolo, depth_model

# ─── 相机内参 ────────────────────────────────────────────
FX, FY = 570.0, 570.0
CX, CY = 320.0, 240.0

def pixel_to_3d(u, v, depth_map):
    z = float(depth_map[int(v), int(u)])
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return x, y, z

# ─── 单帧处理 ────────────────────────────────────────────
def process_frame(frame, yolo, depth_model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8检测
    results = yolo(frame, conf=0.6, verbose=False)
    boxes = results[0].boxes

    # 深度估计
    with torch.no_grad():
        depth = depth_model.infer_image(img_rgb)

    # 计算每个物体的3D坐标
    objects = []
    for i in range(len(boxes)):
        box = boxes[i].xyxy[0].cpu().numpy()
        u = (box[0] + box[2]) / 2
        v = (box[1] + box[3]) / 2
        conf = float(boxes[i].conf[0])
        x, y, z = pixel_to_3d(u, v, depth)
        objects.append({'u': u, 'v': v, 'x': x, 'y': y, 'z': z, 'conf': conf})

    # 可视化
    output = results[0].plot(masks=True)
    for obj in objects:
        label = f"({obj['x']:.2f},{obj['y']:.2f},{obj['z']:.2f})"
        cv2.putText(output, label,
                    (int(obj['u'])-40, int(obj['v'])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return output, objects

# ─── 三种输入模式 ─────────────────────────────────────────
def run_image(img_path, yolo, depth_model):
    frame = cv2.imread(img_path)
    start = time.time()
    output, objects = process_frame(frame, yolo, depth_model)
    elapsed = time.time() - start

    print(f"处理时间: {elapsed*1000:.1f}ms  FPS: {1/elapsed:.1f}")
    print(f"检测到 {len(objects)} 个物体:")
    for i, obj in enumerate(objects):
        print(f"  物体{i+1}: 3D坐标=({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f})  置信度={obj['conf']:.2f}")

    cv2.imwrite('/home/emanon/projects/pipeline_result.jpg', output)
    print("结果保存: pipeline_result.jpg")

def run_video(video_path, yolo, depth_model):
    cap = cv2.VideoCapture(video_path)
    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        output, objects = process_frame(frame, yolo, depth_model)
        fps = 1 / (time.time() - start)
        fps_list.append(fps)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pipeline', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"平均FPS: {np.mean(fps_list):.1f}")

def run_camera(yolo, depth_model):
    cap = cv2.VideoCapture(0)
    fps_list = []
    print("按q退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        output, objects = process_frame(frame, yolo, depth_model)
        fps = 1 / (time.time() - start)
        fps_list.append(fps)
        cv2.putText(output, f"FPS: {fps:.1f}  Objects: {len(objects)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pipeline', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"平均FPS: {np.mean(fps_list):.1f}")

# ─── 主程序 ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], default='image')
    parser.add_argument('--input', type=str, default='')
    args = parser.parse_args()

    yolo, depth_model = load_models()

    if args.mode == 'image':
        img = args.input or '/home/emanon/datasets/ocid_yolo/images/val/result_2018-08-21-14-41-35.jpg'
        run_image(img, yolo, depth_model)
    elif args.mode == 'video':
        run_video(args.input, yolo, depth_model)
    elif args.mode == 'camera':
        run_camera(yolo, depth_model)