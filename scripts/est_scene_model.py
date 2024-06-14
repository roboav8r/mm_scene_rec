import json

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from situated_hri_interfaces.msg import CategoricalDistribution


class MySubscriber(Node):
    def __init__(self, n_obs, out_filename, scene_label, topic):
        super().__init__('my_subscriber_node')
        self.subscription = self.create_subscription(
            CategoricalDistribution,
            topic,
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.counter = 0
        self.n_obs = n_obs
        self.out_filename = out_filename
        self.scene_label = scene_label
        self.obs_counts = [0]

    def listener_callback(self, msg):
        self.get_logger().info(f'Received message: "{msg}"')

        # Initialize scene count
        if self.counter == 0:
            self.obs_counts = [0]*len(msg.categories)

        # Increment scene count
        self.obs_counts[msg.probabilities.index(max(msg.probabilities))] += 1

        self.counter += 1


        if self.counter >= self.n_obs:
            self.get_logger().info(f'Received {self.n_obs} messages, exiting.')

            # Write to file
            data_out = {}
            data_out['true_scene'] = self.scene_label
            data_out['observed_scenes'] = msg.categories
            data_out['observed_counts'] = self.obs_counts

            with open('./config/models/' + self.out_filename + '.json', 'w', encoding='utf-8') as f:
                json.dump(data_out, f, ensure_ascii=False, indent=4)

            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='ROS2 subscriber node')
    parser.add_argument('n_obs', type=int, help='Number of messages to receive before exiting')
    parser.add_argument('out_filename', type=str, help='Output filename')
    parser.add_argument('scene_label', type=str, help='Second string argument')
    parser.add_argument('topic', type=str, help='Topic to subscribe to')
    args = parser.parse_args()

    # Create subscriber node
    node = MySubscriber(args.n_obs, args.out_filename, args.scene_label, args.topic)
    
    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
