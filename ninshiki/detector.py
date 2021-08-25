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
from rclpy.node import MsgType
from rclpy.node import Node
from shisen_interfaces.msg import Image
import sys
import tensorflow as tf
from types import ModuleType
from object_detection.utils import ops as utils_ops
from .detection import Detection
from ninshiki_interfaces.msg import DetectedObject, DetectedObjects


class Detector (Node):
    def __init__(self, node_name: str, topic_name: str, model_path: str):
        super().__init__(node_name)
        self.rows = 0
        self.cols = 0

        self.detection_model = tf.saved_model.load(model_path)

        self.image_subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10)
        self.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

        self.detected_object_publisher = self.create_publisher(
            DetectedObjects, node_name + "/detections", 10)
        self.get_logger().info(
            "publish detected images on "
            + self.detected_object_publisher.topic_name)

    def listener_callback(self, message: MsgType):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)
        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(message.rows, message.cols, 3)
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)

        if (received_frame.size != 0):
            output_dict = self.run_inference_for_single_image(self.detection_model, received_frame)
            self.publishers_detection(output_dict)

            cv2.imshow(self.image_subscription.topic_name, received_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received image and display it")

        else:
            self.get_logger().warn("once, received empty image")

    def publishers_detection(self, output_dict: list):
        messages = DetectedObjects()
        for i in range(len(output_dict)):
            message = DetectedObject()
            message.label = str(output_dict[i].label)
            message.score = output_dict[i].score
            message.left = output_dict[i].left
            message.right = output_dict[i].right
            message.top = output_dict[i].top
            message.bottom = output_dict[i].bottom
            messages.detected_objects.append(message)

        self.detected_object_publisher.publish(messages)

    def run_inference_for_single_image(self, model: ModuleType, image: np.array) -> list:
        image = np.asarray(image)
        # conver input needs to be a tensor
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
        # print(output_dict)
        object_list = list()
        for i in range(output_dict['num_detections']):
            detect_object = Detection(
                output_dict['detection_classes'][i].tolist(),
                output_dict['detection_scores'][i].tolist(),
                output_dict['detection_boxes'][i][0].tolist(),
                output_dict['detection_boxes'][i][1].tolist(),
                output_dict['detection_boxes'][i][2].tolist(),
                output_dict['detection_boxes'][i][3].tolist())
            object_list.append(detect_object)

        return object_list


def main(args=None):
    try:
        topic_name = sys.argv[1]
        model_path = sys.argv[2]

        rclpy.init(args=args)
        detector = Detector("detector", topic_name, model_path)

        rclpy.spin(detector)

        detector.destroy_node()
        rclpy.shutdown()
    except (IndexError):
        print("Usage: ros2 run ninshiki detector <topic_name> <model_file>")


if __name__ == '__main__':
    main()
