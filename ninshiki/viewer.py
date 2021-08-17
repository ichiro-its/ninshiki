# Copyright (c) 2021 Ichiro ITS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shisen_interfaces.msg import Image
import sys


class Viewer (Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)

        self.image_subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10)

        self.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

    def listener_callback(self, message):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)

        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(480, 640, 3)
            print("Raw Image")
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)
            print("Compressed Image")

        if (received_frame.size != 0):
            cv2.imshow(self.image_subscription.topic_name, received_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received image and display it")
        else:
            self.get_logger().warn("once, received empty image")


def main(args=None):
    try:
        topic_name = sys.argv[1]

        rclpy.init(args=args)
        viewer = Viewer("viewer", topic_name)

        rclpy.spin(viewer)

        viewer.destroy_node()
        rclpy.shutdown()
    except (IndexError):
        print("Usage: ros2 run ninshiki viewer <topic_name>")


if __name__ == '__main__':
    main()
