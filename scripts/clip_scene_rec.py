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
from situated_hri_interfaces.msg import CategoricalDistribution


class ClipSceneRecNode(Node):

    def __init__(self):

        super().__init__('clip_scene_rec')

        self.declare_parameter('callback_period_sec', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('scene_labels', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_descriptions', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('clip_model', rclpy.Parameter.Type.STRING)
        self.callback_period_sec = self.get_parameter('callback_period_sec').get_parameter_value().double_value
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_descriptions = self.get_parameter('scene_descriptions').get_parameter_value().string_array_value
        self.clip_model = self.get_parameter('clip_model').get_parameter_value().string_value

        self.image_sub = self.create_subscription(Image, 'clip_scene_image', self.image_callback, 10)
        self.scene_pub = self.create_publisher(String, 'clip_scene', 10)
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, 'clip_scene_category', 10)
        self.timer = self.create_timer(10, self.timer_callback)

        self.image_msg = Image()
        self.image_received = False

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

        self.text_tokens = clip.tokenize(self.scene_descriptions).to(self.device)
        self.text_features = self.model.encode_text(self.text_tokens)

    def image_callback(self,msg):
        self.image_received = True

        self.image_msg = msg


    def timer_callback(self):

        if self.image_received:

            with torch.no_grad():

                # Convert to CV image type
                self.cv_image = self.bridge.imgmsg_to_cv2(self.image_msg)

                # Compute CLIP image embeddings
                self.clip_image = self.preprocess(PILImage.fromarray(self.cv_image)).unsqueeze(0).to(self.device)
                self.image_deatures = self.model.encode_image(self.clip_image)

                # Compute scene probabilities
                self.logits_per_image, self.logits_per_text = self.model(self.clip_image, self.text_tokens)
                probs = self.logits_per_image.softmax(dim=-1).cpu().numpy()

                # Output
                msg = String()
                msg.data = 'Classes: %s \n Probabilities: %s' % (self.scene_labels, str(probs))
                self.scene_pub.publish(msg)

                scene_category_msg = CategoricalDistribution()
                scene_category_msg.categories = self.scene_labels
                scene_category_msg.probabilities = probs[0].tolist()
                self.scene_category_pub.publish(scene_category_msg)

def main(args=None):
    rclpy.init(args=args)

    clip_scene_rec_node = ClipSceneRecNode()
    rclpy.spin(clip_scene_rec_node)

    clip_scene_rec_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()