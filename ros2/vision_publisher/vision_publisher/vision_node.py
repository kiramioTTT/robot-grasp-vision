#!/home/emanon/miniconda3/envs/robot/bin/python

import sys
import cv2
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration

sys.path.append('/home/emanon/projects/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

class VisionPublisher(Node):
    def __init__(self):
        super().__init__('vision_publisher')
        self.get_logger().info('初始化视觉发布节点...')

        # 发布者
        self.pose_pub = self.create_publisher(PoseArray, '/object_poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)

        # 加载模型
        self.yolo = YOLO('/home/emanon/projects/runs/exp4_ocid_fulltune/weights/best.pt')
        cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        self.depth_model = DepthAnythingV2(**cfg)
        self.depth_model.load_state_dict(torch.load(
            '/home/emanon/models/depth_anything/depth_anything_v2_vits.pth',
            map_location='cpu', weights_only=False
        ))
        self.depth_model = self.depth_model.to('cuda').eval()

        # 相机内参
        self.fx, self.fy = 570.0, 570.0
        self.cx, self.cy = 320.0, 240.0

        # 定时器：每500ms处理一张图
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.img_path = '/home/emanon/datasets/ocid_yolo/images/val/result_2018-08-21-14-41-35.jpg'
        self.get_logger().info('节点初始化完成，开始发布...')

    def pixel_to_3d(self, u, v, depth_map):
        z = float(depth_map[int(v), int(u)])
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    def timer_callback(self):
        img = cv2.imread(self.img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测
        results = self.yolo(img, conf=0.6, verbose=False)
        boxes = results[0].boxes

        # 深度估计
        with torch.no_grad():
            depth = self.depth_model.infer_image(img_rgb)

        # 构建PoseArray消息
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'camera_frame'

        # 构建MarkerArray消息
        marker_array = MarkerArray()

        for i in range(len(boxes)):
            box = boxes[i].xyxy[0].cpu().numpy()
            u = (box[0] + box[2]) / 2
            v = (box[1] + box[3]) / 2
            x, y, z = self.pixel_to_3d(u, v, depth)

            # Pose消息
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            # Marker消息（RViz2显示球体）
            marker = Marker()
            marker.header = pose_array.header
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = Duration(sec=1)
            marker_array.markers.append(marker)

        # 发布
        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'发布 {len(boxes)} 个物体位姿')


def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
