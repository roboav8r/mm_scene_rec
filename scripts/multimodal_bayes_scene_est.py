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

class BayesSceneEstNode(Node):

    def __init__(self):
        super().__init__('bayes_scene_est')
    
        # Get scene/estimator parameters
        self.declare_parameter('scene_labels',rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scene_prior',rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('loop_time_sec',rclpy.Parameter.Type.DOUBLE)
        self.scene_labels = self.get_parameter('scene_labels').get_parameter_value().string_array_value
        self.scene_probs = self.get_parameter('scene_prior').get_parameter_value().double_array_value
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value

        # Initialize scene estimate
        self.audiofirst_scene_symbol = gtsam.symbol('s',0)
        self.audiofirst_scene_prob_est = gtsam.DiscreteDistribution([self.audiofirst_scene_symbol,len(self.scene_labels)],self.scene_probs)
        self.audiofirst_initialized = False

        self.visionfirst_scene_symbol = gtsam.symbol('s',1)
        self.visionfirst_scene_prob_est = gtsam.DiscreteDistribution([self.visionfirst_scene_symbol,len(self.scene_labels)],self.scene_probs)
        self.visionfirst_initialized = False

        # Create callback groups
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_srv_cb_group = MutuallyExclusiveCallbackGroup()

        # Setup scene publisher
        self.scene_audiofirst_category_pub = self.create_publisher(CategoricalDistribution, '~/fused_scene_category_audiofirst', 10)
        self.scene_visionfirst_category_pub = self.create_publisher(CategoricalDistribution, '~/fused_scene_category_visionfirst', 10)

        # Setup services
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback, callback_group=self.sub_srv_cb_group)
        self.reconf_srv = self.create_service(Empty, '~/reconfigure', self.reconf_callback, callback_group=self.sub_srv_cb_group)

        # Set up main timer
        self.update_timer = self.create_timer(self.loop_time_sec, self.update_callback, callback_group=self.timer_cb_group)

        # Get sensor parameters, form sensor param dictionary, setup subs
        self.last_audio_msg = None
        self.next_audiofirst_sensor = 'audio'
        self.last_vision_msg = None
        self.next_visionfirst_sensor = 'vision'
            
        self.declare_parameter('audio_obs_labels', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('audio_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('audio_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.audio_sensor_symbol = gtsam.symbol('o',0)
        self.audio_obs_labels = self.get_parameter('audio_obs_labels').get_parameter_value().string_array_value
        self.audio_sensor_model_coeffs = self.get_parameter('audio_sensor_model_coeffs').get_parameter_value().double_array_value
        self.audio_sensor_model_array = np.array(self.audio_sensor_model_coeffs).reshape(-1,len(self.audio_obs_labels))
        self.audio_sub = self.create_subscription(CategoricalDistribution,self.get_parameter('audio_topic').get_parameter_value().string_value, lambda msg: self.save_msg(msg, "audio"), 10, callback_group=self.sub_srv_cb_group)

        self.declare_parameter('vision_obs_labels', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('vision_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('vision_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.vision_sensor_symbol = gtsam.symbol('o',1)
        self.vision_obs_labels = self.get_parameter('vision_obs_labels').get_parameter_value().string_array_value
        self.vision_sensor_model_coeffs = self.get_parameter('vision_sensor_model_coeffs').get_parameter_value().double_array_value
        self.vision_sensor_model_array = np.array(self.vision_sensor_model_coeffs).reshape(-1,len(self.vision_obs_labels))
        self.vision_sub = self.create_subscription(CategoricalDistribution,self.get_parameter('vision_topic').get_parameter_value().string_value, lambda msg: self.save_msg(msg, "vision"), 10, callback_group=self.sub_srv_cb_group)

    def save_msg(self, msg, sensor_name):

        if sensor_name=='audio':
            self.last_audio_msg = msg

        if sensor_name=='vision':
            self.last_vision_msg = msg

    def update_callback(self):

        # Update and publish audiofirst estimate
        if (self.next_audiofirst_sensor == 'audio') & (self.last_audio_msg is not None):

            obs = gtsam.DiscreteDistribution([self.audio_sensor_symbol,len(self.audio_obs_labels)],self.last_audio_msg.probabilities)
            obs_factor = gtsam.DecisionTreeFactor(obs)
            self.audio_sensor_model = gtsam.DiscreteConditional([self.audio_sensor_symbol,len(self.audio_obs_labels)],[[self.audiofirst_scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.audio_sensor_model_array))
            sensor_model_factor = gtsam.DecisionTreeFactor(self.audio_sensor_model)
            likelihood = (obs_factor*sensor_model_factor).sum(1)

            self.audiofirst_scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.audiofirst_scene_prob_est)

            scene_category_msg = CategoricalDistribution()
            scene_category_msg.categories = self.scene_labels
            scene_category_msg.probabilities = self.audiofirst_scene_prob_est.pmf()
            self.scene_audiofirst_category_pub.publish(scene_category_msg)

            self.next_audiofirst_sensor = 'vision'

        elif (self.next_audiofirst_sensor == 'vision') & (self.last_vision_msg is not None):

            obs = gtsam.DiscreteDistribution([self.vision_sensor_symbol,len(self.vision_obs_labels)],self.last_vision_msg.probabilities)
            obs_factor = gtsam.DecisionTreeFactor(obs)
            self.vision_sensor_model = gtsam.DiscreteConditional([self.vision_sensor_symbol,len(self.vision_obs_labels)],[[self.audiofirst_scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.vision_sensor_model_array))
            sensor_model_factor = gtsam.DecisionTreeFactor(self.vision_sensor_model)
            likelihood = (obs_factor*sensor_model_factor).sum(1)

            self.audiofirst_scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.audiofirst_scene_prob_est)

            scene_category_msg = CategoricalDistribution()
            scene_category_msg.categories = self.scene_labels
            scene_category_msg.probabilities = self.audiofirst_scene_prob_est.pmf()
            self.scene_audiofirst_category_pub.publish(scene_category_msg)

            self.next_audiofirst_sensor = 'audio'


        # Update and publish visionfirst estimate
        if (self.next_visionfirst_sensor == 'audio') & (self.last_audio_msg is not None):

            obs = gtsam.DiscreteDistribution([self.audio_sensor_symbol,len(self.audio_obs_labels)],self.last_audio_msg.probabilities)
            obs_factor = gtsam.DecisionTreeFactor(obs)
            self.audio_sensor_model = gtsam.DiscreteConditional([self.audio_sensor_symbol,len(self.audio_obs_labels)],[[self.visionfirst_scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.audio_sensor_model_array))
            sensor_model_factor = gtsam.DecisionTreeFactor(self.audio_sensor_model)
            likelihood = (obs_factor*sensor_model_factor).sum(1)

            self.visionfirst_scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.visionfirst_scene_prob_est)

            scene_category_msg = CategoricalDistribution()
            scene_category_msg.categories = self.scene_labels
            scene_category_msg.probabilities = self.visionfirst_scene_prob_est.pmf()
            self.scene_visionfirst_category_pub.publish(scene_category_msg)

            self.next_visionfirst_sensor = 'vision'

        elif (self.next_visionfirst_sensor == 'vision') & (self.last_vision_msg is not None):

            obs = gtsam.DiscreteDistribution([self.vision_sensor_symbol,len(self.vision_obs_labels)],self.last_vision_msg.probabilities)
            obs_factor = gtsam.DecisionTreeFactor(obs)
            self.vision_sensor_model = gtsam.DiscreteConditional([self.vision_sensor_symbol,len(self.vision_obs_labels)],[[self.visionfirst_scene_symbol,len(self.scene_labels)]],pmf_to_spec(self.vision_sensor_model_array))
            sensor_model_factor = gtsam.DecisionTreeFactor(self.vision_sensor_model)
            likelihood = (obs_factor*sensor_model_factor).sum(1)

            self.visionfirst_scene_prob_est = gtsam.DiscreteDistribution(likelihood*self.visionfirst_scene_prob_est)

            scene_category_msg = CategoricalDistribution()
            scene_category_msg.categories = self.scene_labels
            scene_category_msg.probabilities = self.visionfirst_scene_prob_est.pmf()
            self.scene_visionfirst_category_pub.publish(scene_category_msg)

            self.next_visionfirst_sensor = 'audio'


    def reset_callback(self, request, response):
        self.get_logger().info('Resetting...')

        self.audiofirst_scene_prob_est = gtsam.DiscreteDistribution([self.audiofirst_scene_symbol,len(self.scene_labels)],self.scene_probs)
        self.audiofirst_initialized = False
        self.last_audio_msg = None
        self.next_audiofirst_sensor = 'audio'

        self.visionfirst_scene_prob_est = gtsam.DiscreteDistribution([self.visionfirst_scene_symbol,len(self.scene_labels)],self.scene_probs)
        self.visionfirst_initialized = False
        self.last_vision_msg = None
        self.next_visionfirst_sensor = 'vision'

        return response

    def reconf_callback(self, request, response):
        self.get_logger().info('Reconfiguring...')

        self.audio_sensor_model_coeffs = self.get_parameter('audio_sensor_model_coeffs').get_parameter_value().double_array_value
        self.audio_sensor_model_array = np.array(self.audio_sensor_model_coeffs).reshape(-1,len(self.audio_obs_labels))

        self.vision_sensor_model_coeffs = self.get_parameter('vision_sensor_model_coeffs').get_parameter_value().double_array_value
        self.vision_sensor_model_array = np.array(self.vision_sensor_model_coeffs).reshape(-1,len(self.vision_obs_labels))
        
        return response


def main(args=None):
    rclpy.init(args=args)

    bayes_scene_est_node = BayesSceneEstNode()
    rclpy.spin(bayes_scene_est_node)

    bayes_scene_est_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()