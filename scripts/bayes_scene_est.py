#!/usr/bin/env python3

import numpy as np
import gtsam

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from std_srvs.srv import Empty

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
        self.declare_parameter('scene_labels',rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_prior',rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_probs = self.get_parameter('scene_prior').get_parameter_value().double_array_value

        # Initialize scene estimate
        self.scene_symbol = gtsam.symbol('s',0)
        self.scene_prob_est = gtsam.DiscreteDistribution([self.scene_symbol,len(self.scene_labels)],self.scene_probs)

        # Setup scene publisher
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, '~/fused_scene_category', 10)

        # Setup services
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback)

        # Get sensor parameters, form sensor param dictionary, setup subs
        self.sensor_params = dict()
        self.declare_parameter('sensor_names',rclpy.Parameter.Type.STRING_ARRAY)
        self.sensor_names = self.get_parameter('sensor_names').get_parameter_value().string_array_value

        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            self.subscribers = []
            self.sensor_params[sensor_name] = dict()
            
            self.declare_parameter('%s.obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter('%s.topic' % sensor_name, rclpy.Parameter.Type.STRING)
            self.declare_parameter('%s.sensor_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            self.sensor_params[sensor_name]['symbol'] = gtsam.symbol('o',sensor_idx)
            self.sensor_params[sensor_name]['obs_labels'] = self.get_parameter('%s.obs_labels' % sensor_name).get_parameter_value().string_array_value
            self.sensor_params[sensor_name]['sensor_model_coeffs'] = self.get_parameter('%s.sensor_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            self.sensor_params[sensor_name]['sensor_model_array'] = np.array(self.sensor_params[sensor_name]['sensor_model_coeffs']).reshape(-1,len(self.sensor_params[sensor_name]['obs_labels']))
            self.sensor_params[sensor_name]['sensor_model'] = gtsam.DiscreteConditional([self.sensor_params[sensor_name]['symbol'],len(self.sensor_params[sensor_name]['obs_labels'])],[[self.scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.sensor_params[sensor_name]['sensor_model_array']))

            # self.get_logger().info(f'SENSOR PARAMS: {self.sensor_params[sensor_name]}')

            self.subscribers.append(self.create_subscription(CategoricalDistribution,self.get_parameter('%s.topic' % sensor_name).get_parameter_value().string_value, eval("lambda msg: self.scene_update(msg, \"" + sensor_name + "\")",locals()), 10))

    def publish_fused_scene(self):
        scene_category_msg = CategoricalDistribution()
        scene_category_msg.categories = self.scene_labels
        scene_category_msg.probabilities = self.scene_prob_est.pmf()
        self.scene_category_pub.publish(scene_category_msg)

    def scene_update(self,scene_msg, sensor_name):

        # TODO - compute these and store them in sensor params beforehand to reduce unnecessary computation
        obs = gtsam.DiscreteDistribution([self.sensor_params[sensor_name]['symbol'],len(self.sensor_params[sensor_name]['obs_labels'])],scene_msg.probabilities)
        # likelihood = self.sensor_params[sensor_name]['sensor_model'].likelihood(obs.argmax())

        obs_factor = gtsam.DecisionTreeFactor(obs)
        sensor_model_factor = gtsam.DecisionTreeFactor(self.sensor_params[sensor_name]['sensor_model'])
        likelihood = (obs_factor*sensor_model_factor).sum(1)

        self.scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.scene_prob_est)

        self.publish_fused_scene()

    def reset_callback(self, request, response):
        self.get_logger().info('Resetting...')
        self.scene_prob_est = gtsam.DiscreteDistribution([self.scene_symbol,len(self.scene_labels)],self.scene_probs)
        return response


def main(args=None):
    rclpy.init(args=args)

    bayes_scene_est_node = BayesSceneEstNode()
    rclpy.spin(bayes_scene_est_node)

    bayes_scene_est_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()