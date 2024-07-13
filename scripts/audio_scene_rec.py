#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import importlib
import librosa
import pytorch_lightning as pl
import torchaudio.transforms as T
import pathlib
import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioDataStamped
from std_msgs.msg import String

from situated_hri_interfaces.msg import CategoricalDistribution

from mm_scene_rec.cpjku_dcase23.models.cp_mobile_clean import get_model
from mm_scene_rec.cpjku_dcase23.models.mel import AugmentMelSTFT

# Helper functions
def fuse_model(module):
    # fuse layers
    module.model.eval()  # only works in eval mode
    module.model.cpu()
    module.model.fuse_model()

    # put original net back on cuda
    module.model.cuda()

def load_pretrained_from_id(module):
    ckpt_path = os.path.join(get_package_share_directory('mm_scene_rec'),'config', "checkpoints")
    assert os.path.exists(ckpt_path), f"No checkpoint path '{ckpt_path}' found."
    ckpt_files = [file for file in pathlib.Path(os.path.expanduser(ckpt_path)).rglob('*.ckpt')]
    assert len(ckpt_files) > 0, f"No checkpoint files found in path {ckpt_path}."
    latest_ckpt = sorted(ckpt_files)[-1]
    state_dict = torch.load(latest_ckpt)['state_dict']
    # remove "model" prefix
    state_dict = {k[len("model."):]: state_dict[k] for k in state_dict.keys()}
    module.model.load_state_dict(state_dict)

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config['n_mels'],
                                  sr=config['resample_rate'],
                                  win_length=config['window_size'],
                                  hopsize=config['hop_size'],
                                  n_fft=config['n_fft'],
                                  freqm=config['freqm'],
                                  timem=config['timem'],
                                  fmin=config['fmin'],
                                  fmax=None,
                                  fmin_aug_range=config['fmin_aug_range'],
                                  fmax_aug_range=config['fmax_aug_range']
                                  )

        # CP-Mobile
        self.model = get_model(n_classes=config['n_classes'],
                               in_channels=config['in_channels'],
                               base_channels=config['base_channels'],
                               channels_multiplier=config['channels_multiplier'],
                               expansion_rate=config['expansion_rate']
                               )

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating log mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x):
        """
        :param x: batch of spectrograms
        :return: final model predictions
        """
        x = self.model(x)
        return x

class SceneRecNode(Node):

    def __init__(self):
        super().__init__('scene_rec_node')
        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.scene_pub = self.create_publisher(String, 'audio_scene', 10)
        self.scene_category_pub = self.create_publisher(CategoricalDistribution, 'audio_scene_category', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('downsample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('scene_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('scene_index', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('scene_est_interval', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('labels', rclpy.Parameter.Type.STRING_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.downsample_rate = self.get_parameter('downsample_rate').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.scene_size = self.get_parameter('scene_size').get_parameter_value().integer_value
        self.scene_idx = self.get_parameter('scene_index').get_parameter_value().integer_value
        self.scene_est_interval = self.get_parameter('scene_est_interval').get_parameter_value().double_value
        self.audio_scene_labels = self.get_parameter('labels').get_parameter_value().string_array_value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Audio data storage
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.scene_audio = torch.zeros([self.scene_size, len(self.scene_idx)],dtype=torch.float16)
        self.scene_audio = self.scene_audio.to(self.device)

        # Module configuration
        self.module_config = {}

        self.declare_parameter('project_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('wandb_id', rclpy.Parameter.Type.STRING)
        self.project_name = self.get_parameter('project_name').get_parameter_value().string_value
        self.wandb_id = self.get_parameter('wandb_id').get_parameter_value().string_value

        self.declare_parameter('n_classes', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('in_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('base_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('channels_multiplier', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('expansion_rate', rclpy.Parameter.Type.INTEGER)
        self.module_config['n_classes'] = self.get_parameter('n_classes').get_parameter_value().integer_value
        self.module_config['in_channels'] = self.get_parameter('in_channels').get_parameter_value().integer_value
        self.module_config['base_channels'] = self.get_parameter('base_channels').get_parameter_value().integer_value
        self.module_config['channels_multiplier'] = self.get_parameter('channels_multiplier').get_parameter_value().double_value
        self.module_config['expansion_rate'] = self.get_parameter('expansion_rate').get_parameter_value().integer_value

        self.declare_parameter('resample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('window_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('hop_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('n_fft', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('n_mels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('freqm', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('timem', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('fmin', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('fmax', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('fmin_aug_range', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('fmax_aug_range', rclpy.Parameter.Type.INTEGER)    
        self.module_config['resample_rate'] = self.get_parameter('resample_rate').get_parameter_value().integer_value
        self.module_config['window_size'] = self.get_parameter('window_size').get_parameter_value().integer_value
        self.module_config['hop_size'] = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.module_config['n_fft'] = self.get_parameter('n_fft').get_parameter_value().integer_value
        self.module_config['n_mels'] = self.get_parameter('n_mels').get_parameter_value().integer_value
        self.module_config['freqm'] = self.get_parameter('freqm').get_parameter_value().integer_value
        self.module_config['timem'] = self.get_parameter('timem').get_parameter_value().integer_value
        self.module_config['fmin'] = self.get_parameter('fmin').get_parameter_value().integer_value
        self.module_config['fmax'] = self.get_parameter('fmax').get_parameter_value().integer_value
        self.module_config['fmin_aug_range'] = self.get_parameter('fmin_aug_range').get_parameter_value().integer_value
        self.module_config['fmax_aug_range'] = self.get_parameter('fmax_aug_range').get_parameter_value().integer_value

        # Load scene rec model and resampler object
        self.init_model()
        self.resampler = T.Resample(self.sample_rate, self.downsample_rate, dtype=torch.float16)
        self.resampler = self.resampler.to(self.device)

    def init_model(self):

        # create pytorch lightening module
        self.pl_module = PLModule(self.module_config)

        # load model
        load_pretrained_from_id(self.pl_module)

        # fuse layers
        fuse_model(self.pl_module)

        self.pl_module.mel.cuda()


    def audio_data_callback(self, msg):

        start_time = time.time()

        chunk = torch.frombuffer(msg.audio.data,dtype=torch.float16).view(-1,self.n_channels)

        # Roll the frame, and replace oldest contents with new chunk
        self.frame = torch.roll(self.frame, -chunk.size(0), 0)
        self.frame[-chunk.size(0):,:] = -chunk

        torch.save(self.frame,'frame_data_recovered.pt')

        # Reformat data as needed
        self.scene_audio = self.frame[:,self.scene_idx]
        self.scene_audio = self.scene_audio.to(self.device)
        torch.save(self.scene_audio,'scene_data_recovered.pt')

        resampled_sig = self.resampler(self.scene_audio.T)
        torch.save(resampled_sig,'scene_data_resampled.pt')

        unsqueezed_sig = resampled_sig.unsqueeze(0)
        unsqueezed_sig_float = unsqueezed_sig.float()

        # Run inference
        mel_sig = self.pl_module.mel_forward(unsqueezed_sig_float)
        torch.save(mel_sig,'scene_data_mel.pt')

        scene_est = self.pl_module.forward(mel_sig)
        torch.save(scene_est,'scene_data_est.pt')

        scene_probs = scene_est.softmax(dim=-1).half().detach().cpu().numpy()

        # Output
        msg = String()
        msg.data = 'Classes: %s \n Probabilities: %s' % (self.audio_scene_labels, str(scene_probs))
        self.scene_pub.publish(msg)

        scene_category_msg = CategoricalDistribution()
        scene_category_msg.categories = self.audio_scene_labels
        scene_category_msg.probabilities = scene_probs[0].tolist()
        self.scene_category_pub.publish(scene_category_msg)

        self.get_logger().debug("Inference time (s): %s" % (time.time() - start_time))


def main(args=None):
    rclpy.init(args=args)
    audio_proc_node = SceneRecNode()
    rclpy.spin(audio_proc_node)

    audio_proc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
