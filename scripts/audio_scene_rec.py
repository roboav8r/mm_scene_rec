#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import importlib
import librosa
import torchaudio.transforms as T

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioDataStamped
from std_msgs.msg import String

class SceneRecNode(Node):

    def __init__(self):
        super().__init__('scene_rec_node')
        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.audio_scene_publisher = self.create_publisher(String, 'audio_scene', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('downsample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('scene_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('scene_index', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter('scene_est_interval', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('model_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('mean_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('std_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('labels', rclpy.Parameter.Type.STRING_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.downsample_rate = self.get_parameter('downsample_rate').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.scene_size = self.get_parameter('scene_size').get_parameter_value().integer_value
        self.scene_idx = self.get_parameter('scene_index').get_parameter_value().integer_array_value
        self.scene_est_interval = self.get_parameter('scene_est_interval').get_parameter_value().double_value
        self.model_path = os.path.join(get_package_share_directory('situated_interaction'),self.get_parameter('model_path').get_parameter_value().string_value)
        self.mean_path = os.path.join(get_package_share_directory('situated_interaction'),self.get_parameter('mean_path').get_parameter_value().string_value)
        self.std_path = os.path.join(get_package_share_directory('situated_interaction'),self.get_parameter('std_path').get_parameter_value().string_value)
        self.audio_scene_labels = self.get_parameter('labels').get_parameter_value().string_array_value

        # Audio data storage
        # TODO - get datatype from config file
        # self.frame = torch.zeros([self.frame_size*self.n_channels],dtype=torch.float16)
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.scene_audio = torch.zeros([self.scene_size, len(self.scene_idx)],dtype=torch.float16)
        self.scene_audio = self.scene_audio.to('cuda')

        # Load scene rec model and dataset norm weights
        self.init_model()
        self.load_mean_std()
        self.resampler = T.Resample(self.sample_rate, self.downsample_rate, dtype=torch.float16)
        self.resampler = self.resampler.to('cuda')

    def init_model(self):

        with open(self.model_path+'model_config.json',"r")as f:
            model_config=json.load(f)
       
        module = importlib.import_module('situated_interaction.models.{}'.format(model_config['arch']))
        Network = getattr(module, 'Network')
        self.model=Network(model_config)

        # Load pretrained weights
        state_dict=torch.load(self.model_path+"/model_state_dict.pth")

        # if the weights are float16 cast the model
        if [v for k,v in state_dict.items()][0].dtype==torch.float16:
            self.model=self.model.half()
        self.model.load_state_dict(state_dict)

        self.model.cuda()
        self.model.eval()
        self.model = self.model.to('cuda')
    
    def load_mean_std(self):
        self.tr_mean = torch.load(self.mean_path)
        self.tr_std = torch.load(self.std_path)

        self.tr_mean = self.tr_mean.to('cuda')
        self.tr_std = self.tr_std.to('cuda')

        self.tr_mean.half()
        self.tr_std.half()        

    def processor_d18_stereo_tensor(self, tensor):
        n_fft = 2048  # 2048
        sr = 22050  # 22050  # 44100  # 32000
        n_mels = 256

        hop_length = 512
        fmax = None

        self.spectrograms = []

        for y in tensor:

            y = y.to('cpu').numpy()

            # compute stft
            stft = librosa.stft(np.asfortranarray(y), n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            # keep only amplitures
            stft = np.abs(stft)

            # spectrogram weighting
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

            # keep spectrogram
            self.spectrograms.append(np.asarray(spectrogram))

        self.spectrograms = np.asarray(self.spectrograms, dtype=np.float32)

    def audio_data_callback(self, msg):

        # self.get_logger().info('Got audio data with size %s' % (str(len(msg.audio.data))))

        chunk = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        # self.get_logger().info('Got chunk with size %s' % (str(chunk.size())))

        # Roll the frame, and replace oldest contents with new chunk
        self.frame = torch.roll(self.frame, -chunk.size(0), 0)
        self.frame[-chunk.size(0):,:] = -chunk

        # self.get_logger().info('Computed frame with size %s' % (str(self.frame.size())))

        torch.save(self.frame,'frame_data_recovered.pt')

        self.scene_audio = self.frame[:,self.scene_idx]
        # self.get_logger().info('Computed scene audio with size %s' % (str(self.scene_audio.size())))
        self.scene_audio = self.scene_audio.to('cuda')
        torch.save(self.scene_audio,'scene_data_recovered.pt')

        resampled_sig = self.resampler(self.scene_audio.T)

        torch.save(resampled_sig,'scene_data_resampled.pt')

        self.processor_d18_stereo_tensor(resampled_sig)

        # classify
        x = torch.from_numpy(self.spectrograms).to('cuda')
        x = x.half()

        x.unsqueeze_(0)

        # print(x.shape)
        x = (x - self.tr_mean) / self.tr_std
        x = x.half()

        out=self.model(x)

        # Normalize logits
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, class_idx = torch.max(probs, 1)

        scene_msg = String()
        scene_msg.data = "Class: %s, %s%%; probs: %s" % (self.audio_scene_labels[class_idx], conf.item(), str(probs))
        self.audio_scene_publisher.publish(scene_msg)

def main(args=None):
    rclpy.init(args=args)
    audio_proc_node = SceneRecNode()
    rclpy.spin(audio_proc_node)
    audio_proc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
