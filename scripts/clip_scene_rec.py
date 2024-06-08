#!/usr/bin/env python3

import torch
import clip
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import String


class ClipSceneRecNode(Node):

    def __init__(self):

        super().__init__('clip_scene_rec')

        # TODO - convert these to ROS parameters, declare and import
        self.callback_period = 5
        self.scene_labels = ['indoor', 'outdoor', 'transportation']
        self.scene_descriptions = ['inside a building', 'outdoors', 'inside a vehicle']

        self.image_sub = self.create_subscription(Image, 'clip_scene_image', self.image_callback, 10)
        self.scene_pub = self.create_publisher(String, 'clip_scene', 10)
        self.timer = self.create_timer(10, self.timer_callback)

        self.image_msg = Image()
        self.image_received = False
        self.scene_probs = [.33, .33, .33]

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def image_callback(self,msg):
        self.image_received = True

        self.image_msg = msg


    def timer_callback(self):
        
        if self.image_received:

            # Convert to CV image type
            self.cv_image = self.bridge.imgmsg_to_cv2(self.image_msg)
            cv2.imwrite('last_image.png',self.cv_image)


            msg = String()
            msg.data = "Timer callback triggered."
            self.scene_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    clip_scene_rec_node = ClipSceneRecNode()
    rclpy.spin(clip_scene_rec_node)

    clip_scene_rec_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()