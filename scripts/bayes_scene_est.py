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
    
        # Get scene/estimator parameters
        self.declare_parameter('callback_period_sec',rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('scene_labels',rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_prior',rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.callback_period_sec = self.get_parameter('callback_period_sec').get_parameter_value().double_value
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_probs = self.get_parameter('scene_prior').get_parameter_value().double_array_value

        # Initialize scene estimate
        self.scene_symbol = scene_sym = gtsam.symbol('s',0)
        self.scene_prob_est = gtsam.DiscreteDistribution([scene_sym,len(self.scene_labels)],self.scene_probs)

        # Setup scene publisher and timer
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, 'fused_scene_category', 10)
        self.timer = self.create_timer(self.callback_period_sec, self.publish_fused_scene)

        # Get sensor parameters, form sensor param dictionary, setup subs
        sensor_params = dict()
        self.declare_parameter('sensor_names',rclpy.Parameter.Type.STRING_ARRAY)
        self.sensor_names = self.get_parameter('sensor_names').get_parameter_value().string_array_value

        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_params[sensor_name] = dict()
            
            self.declare_parameter('%s.obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter('%s.topic' % sensor_name, rclpy.Parameter.Type.STRING)
            self.declare_parameter('%s.sensor_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            sensor_params[sensor_name]['symbol'] = gtsam.symbol('o',sensor_idx)
            sensor_params[sensor_name]['obs_labels'] = self.get_parameter('%s.obs_labels' % sensor_name).get_parameter_value().string_array_value
            sensor_params[sensor_name]['sensor_model_coeffs'] = self.get_parameter('%s.obs_labels' % sensor_name).get_parameter_value().string_array_value
            sensor_params[sensor_name]['sensor_model_array'] = np.array(sensor_params[sensor_name]['sensor_model_coeffs']).reshape(len(sensor_params[sensor_name]['obs_labels']),-1)
            sensor_params[sensor_name]['sensor_model'] = gtsam.DiscreteConditional([sensor_params[sensor_name]['symbol'],len(sensor_params[sensor_name]['obs_labels'])],[[self.scene_symbol,len(self.scene_labels)]],pmf_to_spec(sensor_params[sensor_name]['sensor_model_array']))

            exec('self.clip_category_sub = self.create_subscription(CategoricalDistribution, \'clip_scene_category\', self.scene_update, 10)')

            # self.obs_sym = gtsam.symbol('o',0)
            # self.obs_labels = ['indoor','outdoor','transportation']
            # self.sensor_model_array = np.array([[.7, .1, .2],[.1, .8, .1],[.25, .25, .5]])
            # self.sensor_model = gtsam.DiscreteConditional([self.obs_sym,len(self.obs_labels)],[[scene_sym,len(self.scene_labels)]],pmf_to_spec(self.sensor_model_array))
        

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