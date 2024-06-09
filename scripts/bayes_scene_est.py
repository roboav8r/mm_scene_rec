#!/usr/bin/env python3

import numpy as np
import gtsam

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from situated_hri_interfaces.msg import CategoricalDistribution


def pmf_to_spec(pmf):

    spec = ''
    for row_idx in range(pmf.shape[0]):
        row = pmf[row_idx,:]
        row_spec = ''
        
        for col_idx in range(len(row)):
            if col_idx == 0: # If empty spec
                row_spec += str(row[col_idx])
            else:
                row_spec += '/' +  str(row[col_idx]) 
        
        if row_idx==0:
            spec += row_spec
        else:
            spec += ' ' + row_spec
        
    return spec

class BayesSceneEstNode(Node):

    def __init__(self):
        super().__init__('bayes_scene_est')
    
        # TODO - put these in param file
        self.callback_period_sec = 1.
        self.scene_labels = ['indoor','outdoor','transportation']
        self.scene_probs = [.4, .4, .2]

        # Initialize scene estimate
        self.scene_symbol = scene_sym = gtsam.symbol('s',0)
        self.scene_prob_est = gtsam.DiscreteDistribution([scene_sym,len(self.scene_labels)],self.scene_probs)

        # TODO - get these for each detector
        self.obs_sym = gtsam.symbol('o',0)
        self.obs_labels = ['indoor','outdoor','transportation']
        self.sensor_model_array = np.array([[.7, .1, .2],[.1, .8, .1],[.25, .25, .5]])
        self.sensor_model = gtsam.DiscreteConditional([self.obs_sym,len(self.obs_labels)],[[scene_sym,len(self.scene_labels)]],pmf_to_spec(self.sensor_model_array))

        # TODO - set up multiple subscribers
        self.clip_category_sub = self.create_subscription(CategoricalDistribution, 'clip_scene_category', self.scene_update, 10)
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, 'fused_scene_category', 10)
        self.timer = self.create_timer(self.callback_period_sec, self.publish_fused_scene)

    def publish_fused_scene(self):
        scene_category_msg = CategoricalDistribution()
        scene_category_msg.categories = self.scene_labels
        scene_category_msg.probabilities = self.scene_prob_est.pmf()
        self.scene_category_pub.publish(scene_category_msg)

    def scene_update(self,scene_msg):

        # TODO - get these for each detector
        obs = gtsam.DiscreteDistribution([self.obs_sym,len(self.obs_labels)],scene_msg.probabilities)
        obs_factor = gtsam.DecisionTreeFactor(obs)
        sensor_model_factor = gtsam.DecisionTreeFactor(self.sensor_model)
        likelihood = (obs_factor*sensor_model_factor).sum(1)

        self.scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.scene_prob_est)


def main(args=None):
    rclpy.init(args=args)

    bayes_scene_est_node = BayesSceneEstNode()
    rclpy.spin(bayes_scene_est_node)

    bayes_scene_est_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()