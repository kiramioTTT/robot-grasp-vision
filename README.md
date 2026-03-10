# 机器人抓取视觉感知系统

基于YOLOv8-seg与单目深度估计的机器人抓取视觉感知系统，通过ROS2话题实时发布目标3D位姿。

## 性能指标

| 指标 | 数值 |
|------|------|
| mask mAP50 | 99% |
| mask mAP50-95 | 78% |
| 端到端FPS | 14.2 |
| YOLOv8推理 | 9.8ms |
| 深度估计 | 60.6ms |

## 系统架构
```
RGB图片 → YOLOv8-seg（实例分割）→ 3D坐标计算 → ROS2话题发布 → RViz2可视化
                ↑
        Depth Anything V2（深度估计）
```

## 快速运行

### 推理单张图片
```bash
conda activate robot
python inference/pipeline.py --mode image --input 图片路径
```

### 运行ROS2完整链路
```bash
# 终端1
cd ~/ros2_ws && colcon build && source install/setup.bash
ros2 run vision_publisher vision_node

# 终端2
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run vision_publisher subscriber_demo

# 终端3（可视化）
source /opt/ros/jazzy/setup.bash && rviz2
```

## 环境要求

- Ubuntu 24.04 (WSL2)
- Python 3.10 (conda robot环境)
- CUDA 12.x
- ROS2 Jazzy
- RTX 3060 6GB（或同等显存GPU）

## 数据集

OCID (Object Cluttered Indoors Dataset) ARID20场景
- 训练集：852张
- 验证集：214张
