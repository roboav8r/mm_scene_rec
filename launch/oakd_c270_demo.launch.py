import os

from ament_index_python import get_package_share_directory

from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node, LoadComposableNodes, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    ld = LaunchDescription()

    # Config files
    cam_config = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'oakd_cam.yaml'
    )
    mic_config = os.path.join(
        get_package_share_directory('ros_audition'),
        'config',
        'c270_config.yaml'
    )
    clip_config = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'clip_params.yaml'
    )
    audio_scene_rec_config = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'audio_scene_rec_params.yaml'
    )

    bayes_est_config = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'bayes_estimator_params.yaml'
    )

    # Static TF node
    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = ["0", "0", "1.0", "0", "0", "0", "map", "oak-d-base-frame"]
    )
    ld.add_action(tf_node)

    # Sensor nodes
    cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'rgbd_pcl.launch.py'
            ])
        ]),
        launch_arguments={'params_file': cam_config }.items()
    )
    ld.add_action(cam_node)

    acq_node = Node(
        package='ros_audition',
        executable='audio_acq_node.py',
        name='audio_acq_node',
        output='screen',
        parameters=[mic_config]
    )
    ld.add_action(acq_node)

    # Scene recognition nodes
    clip_rec_node = Node(package = "mm_scene_rec", 
                    executable = "clip_scene_rec.py",
                    name = "clip_scene_rec",
                    remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[clip_config]
    )
    ld.add_action(clip_rec_node)
    
    audio_rec_node = Node(
        package='mm_scene_rec',
        executable='audio_scene_rec.py',
        name='audio_scene_rec',
        output='screen',
        parameters=[audio_scene_rec_config]
    )
    ld.add_action(audio_rec_node)


    # Fusion node
    scene_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_scene_category_est",
                    output = 'screen',
                    parameters=[bayes_est_config]
    )
    ld.add_action(scene_rec_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('mm_scene_rec'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld