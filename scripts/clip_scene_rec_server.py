#!/usr/bin/env python3

import torch
import clip
from PIL import Image as PILImage
import time

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from situated_hri_interfaces.srv import SceneVisRec
from situated_hri_interfaces.msg import CategoricalDistribution

class ClipSceneRecServer(Node):

    def __init__(self):

        super().__init__('clip_scene_rec_server')

        self.declare_parameter('scene_labels', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_descriptions', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('clip_model', rclpy.Parameter.Type.STRING)
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_descriptions = self.get_parameter('scene_descriptions').get_parameter_value().string_array_value
        self.clip_model = self.get_parameter('clip_model').get_parameter_value().string_value

        self.clip_service = self.create_service(SceneVisRec, 'scene_vis_rec_service', self.scene_rec_callback)

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

        self.text_tokens = clip.tokenize(self.scene_descriptions).to(self.device)
        self.text_features = self.model.encode_text(self.text_tokens)


    def scene_rec_callback(self, request, response):
        start_time = time.time()

        with torch.no_grad():

            # Convert to CV image type
            self.cv_image = self.bridge.imgmsg_to_cv2(request.scene_image)

            # Compute CLIP image embeddings
            self.clip_image = self.preprocess(PILImage.fromarray(self.cv_image)).unsqueeze(0).to(self.device)
            self.image_features = self.model.encode_image(self.clip_image)

            # Compute scene probabilities
            self.logits_per_image, self.logits_per_text = self.model(self.clip_image, self.text_tokens)
            probs = self.logits_per_image.softmax(dim=-1).cpu().numpy()

            # Output
            response.scene_class.categories = self.scene_labels
            response.scene_class.probabilities = probs[0].tolist()
            
            self.get_logger().info("Scene recognition inference time (s): %s" % (time.time() - start_time))
            return response

def main(args=None):
    rclpy.init(args=args)

    clip_scene_rec_server = ClipSceneRecServer()
    rclpy.spin(clip_scene_rec_server)

    clip_scene_rec_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()