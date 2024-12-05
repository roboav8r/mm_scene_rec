#!/usr/bin/env python3

import numpy as np
import gtsam

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

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

class JointSceneEstNode(Node):

    def __init__(self):
        super().__init__('joint_scene_est')
    
        # Get scene/estimator parameters
        self.declare_parameter('scene_labels',rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_prior',rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('loop_time_sec',rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('min_prob',rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('max_prob',rclpy.Parameter.Type.DOUBLE)
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_probs = self.get_parameter('scene_prior').get_parameter_value().double_array_value
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value
        self.min_prob = self.get_parameter('min_prob').get_parameter_value().double_value
        self.max_prob = self.get_parameter('max_prob').get_parameter_value().double_value

        # Initialize scene estimate
        self.scene_symbol = gtsam.symbol('s',0)
        self.scene_prob_est = gtsam.DiscreteDistribution([self.scene_symbol,len(self.scene_labels)],self.scene_probs)

        # Create callback groups
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_srv_cb_group = MutuallyExclusiveCallbackGroup()

        # Setup scene publisher
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, '~/joint_scene_category', 10)

        # Setup services
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback, callback_group=self.sub_srv_cb_group)
        self.reconf_srv = self.create_service(Empty, '~/reconfigure', self.reconf_callback, callback_group=self.sub_srv_cb_group)

        # Set up main timer
        self.update_timer = self.create_timer(self.loop_time_sec, self.update_callback, callback_group=self.timer_cb_group)

        # Get sensor parameters, form sensor param dictionary, setup subs
        # self.last_sensor_update_idx = None
        # self.next_sensor_update_idx = None
        self.last_sensor_msg = dict()
        self.sensor_params = dict()
        self.declare_parameter('sensor_names',rclpy.Parameter.Type.STRING_ARRAY)
        self.sensor_names = self.get_parameter('sensor_names').get_parameter_value().string_array_value
        self.msg_is_new = [False]*len(self.sensor_names)

        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            self.subscribers = []
            self.sensor_params[sensor_name] = dict()
            
            self.declare_parameter('%s.obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter('%s.topic' % sensor_name, rclpy.Parameter.Type.STRING)
            self.declare_parameter('%s.sensor_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            # TODO (improvement) - make sensor class with all these in it
            self.sensor_params[sensor_name]['symbol'] = gtsam.symbol('o',sensor_idx)
            self.sensor_params[sensor_name]['obs_labels'] = self.get_parameter('%s.obs_labels' % sensor_name).get_parameter_value().string_array_value
            self.sensor_params[sensor_name]['sensor_model_coeffs'] = self.get_parameter('%s.sensor_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            self.sensor_params[sensor_name]['sensor_model_array'] = np.array(self.sensor_params[sensor_name]['sensor_model_coeffs']).reshape(-1,len(self.sensor_params[sensor_name]['obs_labels']))
            self.sensor_params[sensor_name]['sensor_model'] = gtsam.DiscreteConditional([self.sensor_params[sensor_name]['symbol'],len(self.sensor_params[sensor_name]['obs_labels'])],[[self.scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.sensor_params[sensor_name]['sensor_model_array']))

            # self.get_logger().info(f'SENSOR PARAMS: {self.sensor_params[sensor_name]}')

            self.subscribers.append(self.create_subscription(CategoricalDistribution,self.get_parameter('%s.topic' % sensor_name).get_parameter_value().string_value, eval("lambda msg: self.save_msg(msg, \"" + sensor_name + "\")",locals()), 10, callback_group=self.sub_srv_cb_group))

    def normalize_probs(self):
        # self.scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.scene_prob_est)

        pmf = self.scene_prob_est.pmf()
        # self.get_logger().info(f"Raw PMF: {pmf}")

        for ii, prob in enumerate(pmf):
            if prob > self.max_prob:
                pmf[ii] = self.max_prob
            elif prob < self.min_prob:
                pmf[ii] = self.min_prob

        self.scene_prob_est = gtsam.DiscreteDistribution([self.scene_symbol,len(self.scene_labels)],pmf)

        # self.get_logger().info(f"Normalized PMF: {self.scene_prob_est.pmf()}")

    def publish_fused_scene(self):
        scene_category_msg = CategoricalDistribution()
        scene_category_msg.categories = self.scene_labels
        scene_category_msg.probabilities = self.scene_prob_est.pmf()
        self.scene_category_pub.publish(scene_category_msg)

    def save_msg(self, msg, sensor_name):

        sensor_idx = self.sensor_names.index(sensor_name)

        self.last_sensor_msg[sensor_name] = msg
        self.msg_is_new[sensor_idx] = True

        # self.get_logger().info(f"Got message from {sensor_name}, msg_is_new: {self.msg_is_new}")

    def update_callback(self):


        # If all sensors have a new observation
        if all(self.msg_is_new):
            # self.get_logger().info(f"All new messages")

            # Compute the joint scene estimate with likelihood factors from each sensor
            temp_factor = self.scene_prob_est

            for sensor in self.sensor_params.keys():
                obs_msg = self.last_sensor_msg[sensor]

                obs = gtsam.DiscreteDistribution([self.sensor_params[sensor]['symbol'],len(self.sensor_params[sensor]['obs_labels'])],obs_msg.probabilities)
                obs_factor = gtsam.DecisionTreeFactor(obs)
                sensor_model_factor = gtsam.DecisionTreeFactor(self.sensor_params[sensor]['sensor_model'])
                likelihood = (obs_factor*sensor_model_factor).sum(1)

                temp_factor = likelihood*temp_factor

                # self.get_logger().info(f"temp_factor after {sensor} update: {temp_factor}")

            self.scene_prob_est = gtsam.DiscreteDistribution(temp_factor)
            # self.get_logger().info(f"Scene prob after all updates: {self.scene_prob_est}")
        
            # Normalize
            self.normalize_probs()

            # self.get_logger().info(f"Scene prob after normalization: {self.scene_prob_est}")

            # Publish
            self.publish_fused_scene()

            # Reset
            self.msg_is_new = [False]*len(self.sensor_names)        

    def reset_callback(self, request, response):
        self.get_logger().info('Resetting...')
        self.scene_prob_est = gtsam.DiscreteDistribution([self.scene_symbol,len(self.scene_labels)],self.scene_probs)
        return response

    def reconf_callback(self, request, response):
        self.get_logger().info('Reconfiguring...')

        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            self.sensor_params[sensor_name]['sensor_model_coeffs'] = self.get_parameter('%s.sensor_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            self.sensor_params[sensor_name]['sensor_model_array'] = np.array(self.sensor_params[sensor_name]['sensor_model_coeffs']).reshape(-1,len(self.sensor_params[sensor_name]['obs_labels']))
            self.sensor_params[sensor_name]['sensor_model'] = gtsam.DiscreteConditional([self.sensor_params[sensor_name]['symbol'],len(self.sensor_params[sensor_name]['obs_labels'])],[[self.scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.sensor_params[sensor_name]['sensor_model_array']))

        return response


def main(args=None):
    rclpy.init(args=args)

    joint_scene_est_node = JointSceneEstNode()
    rclpy.spin(joint_scene_est_node)

    joint_scene_est_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()