import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shisen_interfaces.msg import CompressedImage
from shisen_interfaces.msg import RawImage
import sys
import functools
import collections

# _to_numpy = {}

class Viewer (Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        self.raw_image_subscription = self.create_subscription(
            RawImage,
            topic_name,
            self.listener_callback_raw,
            10)
        self.get_logger().info("subscribe raw image on " + self.raw_image_subscription.topic_name)
        self.compressed_image_subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback_compressed,
            10)
        self.get_logger().info("subscribe compressed image on " + self.compressed_image_subscription.topic_name)
        
    def listener_callback_raw(self, message):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)
        received_frame = received_frame.reshape(480,640,3)

        if (received_frame.size != 0):
            cv2.imshow(self.raw_image_subscription.topic_name, received_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received raw image and display it")
        else:
            self.get_logger().warn("once, received empty raw image")

    def listener_callback_compressed(self, message):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)
        received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)

        if (received_frame.size != 0):
            cv2.imshow(self.compressed_image_subscription.topic_name, received_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received compressed image and display it")
        else:
            self.get_logger().warn("once, received empty compressed image")


def main(args=None):
    if (len(sys.argv) < 2):
        print("Usage: ros2 run shisen viewer <topic_name>")
    topic_name = sys.argv[1]
    rclpy.init(args=args)
    viewer = Viewer("viewer", topic_name)

    rclpy.spin(viewer)

    viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()