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
from shisen_interfaces.msg import CompressedImage
from shisen_interfaces.msg import RawImage
import sys
import tensorflow as tf
from object_detection.utils import ops as utils_ops


class Viewer (Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        self.detection_model = tf.saved_model.load("ninshiki/ninshiki/saved_model")

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
        self.get_logger().info(
            "subscribe compressed image on "
            + self.compressed_image_subscription.topic_name)

    def listener_callback_raw(self, message):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)
        received_frame = received_frame.reshape(480, 640, 3)

        output_dict = self.run_inference_for_single_image(self.detection_model, received_frame)
        print(output_dict)

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

        output_dict = self.run_inference_for_single_image(self.detection_model, received_frame)
        print(output_dict)

        if (received_frame.size != 0):
            cv2.imshow(self.compressed_image_subscription.topic_name, received_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received compressed image and display it")
        else:
            self.get_logger().warn("once, received empty compressed image")

    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {
            key: value[0, : num_detections].numpy()
            for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict


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
