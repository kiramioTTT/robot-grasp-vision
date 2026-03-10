import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

class ObjectSubscriber(Node):
    def __init__(self):
        super().__init__('object_subscriber')
        self.subscription = self.create_subscription(
            PoseArray,
            '/object_poses',
            self.callback,
            10
        )
        self.get_logger().info('订阅节点启动，等待数据...')

    def callback(self, msg):
        self.get_logger().info(f'收到 {len(msg.poses)} 个物体位姿:')
        for i, pose in enumerate(msg.poses):
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            self.get_logger().info(f'  物体{i+1}: ({x:.3f}, {y:.3f}, {z:.3f})')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
