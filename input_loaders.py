##################################################
# Imports
##################################################

import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import lmdb
import os

# Configs
_FPS = 30.0


##################################################
# Samplers
##################################################

class ActionRecognitionSampler(object):
    def __init__(self, fps=5.0, sample_mode='center', num_frames_per_action=16):
        self.fps = fps
        self.sample_mode = sample_mode
        self.num_frames_per_action = num_frames_per_action
    
    def __call__(self, action):
        times, frames_idxs = sample_action_recognition_frames(action.start_time, action.stop_time, 
                                                              action.video_duration, fps=self.fps, 
                                                              sample_mode=self.sample_mode, 
                                                              num_frames_per_action=self.num_frames_per_action, 
                                                              fps_init=_FPS)
        return times, frames_idxs, None
    
class ActionAnticipationSampler(object):
    def __init__(self, t_buffer, t_ant, fps=5.0):
        self.t_buffer = t_buffer
        self.t_ant = t_ant
        self.fps = fps
    
    def __call__(self, action):
        times, frames_idxs, mask = sample_action_anticipation_frames(action.start_time, self.t_buffer, 
                                                                      self.t_ant, fps=self.fps, fps_init=_FPS)
        return times, frames_idxs, mask

class ActionAnticipationAndRecognitionSampler(object):
    def __init__(self, t_buffer, t_ant, fps=5.0, sample_mode='center', num_frames_per_action=16):
        self.t_buffer = t_buffer
        self.t_ant = t_ant
        self.fps = fps
        self.sample_mode = sample_mode
        self.num_frames_per_action = num_frames_per_action
    
    def __call__(self, action):
        times_ant, frames_idxs_ant, mask_ant = sample_action_anticipation_frames(action.start_time, self.t_buffer, 
                                                                     self.t_ant, fps=self.fps, fps_init=_FPS)
        times_rec, frames_idxs_rec = sample_action_recognition_frames(action.start_time, action.stop_time,
                                                              action.video_duration, fps=self.fps,
                                                              sample_mode=self.sample_mode,
                                                              num_frames_per_action=self.num_frames_per_action,
                                                              fps_init=_FPS)
        times = np.concatenate([times_ant, times_rec])
        frames_idxs = np.concatenate([frames_idxs_ant, frames_idxs_rec])
        #mask = np.concatenate([mask_ant, np.zeros([len(frames_idxs_rec)])])
        return times, frames_idxs, mask_ant

    
def get_sampler(args):
    if args.task in ['recognition', 'all_frames']:
        sampler = ActionRecognitionSampler(fps=args.fps, sample_mode=args.sample_mode, 
                                           num_frames_per_action=args.num_frames_per_action)
        
    elif args.task == 'anticipation': 
        sampler = ActionAnticipationSampler(t_buffer=args.t_buffer, t_ant=args.t_ant, fps=args.fps)

    elif args.task == 'anticipation_recognition':
        sampler = ActionAnticipationAndRecognitionSampler(t_buffer=args.t_buffer, t_ant=args.t_ant, fps=args.fps, 
                                                          sample_mode=args.sample_mode, 
                                                          num_frames_per_action=args.num_frames_per_action)

    else:
        raise Exception(f'Error. Task "{args.task}" not supported.')
    return sampler


def sample_action_recognition_frames(time_start, time_stop, video_duration, fps=5.0, sample_mode='center', 
                                     num_frames_per_action=16, fps_init=30.0):
    if sample_mode == 'center':
        if num_frames_per_action is None:
            num_frames_per_action = int((time_stop - time_start) * fps) + 1
        t_c = (time_start + time_stop) / 2.0
        times = np.arange(num_frames_per_action) / fps
        times = times - times.max() / 2 + t_c
        times = np.clip(times, 0.0, video_duration)
        frames_idxs = np.round(times * fps_init).astype(np.int32) + 1 # first frame is 1 not 0
        
    elif sample_mode == 'center_hard':
        t_c = (time_start + time_stop) / 2.0
        times = np.arange(num_frames_per_action) / fps
        times = times - times.max() / 2 + t_c
        times = np.clip(times, time_start, time_stop)
        frames_idxs = np.round(times * fps_init).astype(np.int32) + 1 # first frame is 1 not 0
        
    elif sample_mode == 'uniform':
        times = np.linspace(time_start, time_stop, num_frames_per_action)
        frames_idxs = np.round(times * fps_init).astype(np.int32) + 1 # first frame is 1 not 0
        
    else:
        raise Exception(f'Error. Mode "{sample_mode}" not supported.')
    return times, frames_idxs

def sample_action_anticipation_frames(time_start, t_buffer, t_ant, fps=5.0, fps_init=30.0):
    num_frames = int(np.floor(t_buffer * fps))
    times = (np.arange(num_frames) - num_frames) / fps + time_start
    times = np.clip(times, 0, np.inf)
    times = times.astype(np.float32)
    mask = 1.0 * ((time_start - times) >= t_ant)
    if (fps_init / fps) < 1e-5:
        frames_idxs = np.round(times * fps_init).astype(np.int32)
    else:
        frames_idxs = np.floor(times * fps_init).astype(np.int32) + 1# first frame is 1 not 0
    times = (frames_idxs- 1) / fps_init
    return times, frames_idxs, mask


##################################################
# Loaders
##################################################

class FramesLoader(object):
    """
    Load intput frames or optical flow.
    """
    def __init__(self, sampler, frames_base_path, fps, input_name='frame', 
                 frame_tmpl='{}_frame_{:010d}.jpg', sample_mode='center', 
                 num_frames_per_action=16, transform_frame=None, 
                 transform_video=None, task='recognition'):
        self.frames_base_path = frames_base_path
        self.fps = fps
        self.input_name = input_name
        self.frame_tmpl = frame_tmpl
        self.sample_mode = sample_mode # ['center', 'center_hard', 'uniform']
        self.num_frames_per_action = num_frames_per_action
        self.transform_frame = transform_frame
        self.transform_video = transform_video
        self.sampler = sampler
        self.task = task
    
    def __call__(self, action):
        
        # Sample frames
        times, frames_idxs, mask = self.sampler(action)
        frames_names = [self.frame_tmpl.format(action.video_id, i) for i in frames_idxs]
        partition_folder = 'train' if action.partition in ['train', 'validation'] else 'test'
        frames = []
        for f_name in frames_names:
            f_path = os.path.join(self.frames_base_path, partition_folder, action.participant_id, 
                                  action.video_id, f_name)
            frame = Image.open(f_path)
            
            # Transform frame
            if self.transform_frame is not None:
                frame = self.transform_frame(frame)
            frames += [frame]
            
        # Transform video
        if self.transform_video is not None:
            frames = self.transform_video(frames)
        out = {self.input_name: frames}
        if self.task == 'anticipation':
            out['mask'] = mask
        return out
    

class FeaturesLoader(object):
    """
    Load intput feature.
    """
    def __init__(self, sampler, feature_base_path, fps, input_name='rgb', 
                 frame_tmpl='{}_frame_{:010d}.jpg', sample_mode='center', 
                 num_frames_per_action=16, transform_feat=None, 
                 transform_video=None, task='recognition'):
        self.feature_base_path = feature_base_path
        self.env = lmdb.open(os.path.join(self.feature_base_path, input_name), readonly=True, lock=False)
        self.fps = fps
        self.input_name = input_name
        self.frame_tmpl = frame_tmpl
        self.sample_mode = sample_mode # ['center', 'center_hard', 'uniform']
        self.num_frames_per_action = num_frames_per_action
        self.transform_feat = transform_feat
        self.transform_video = transform_video
        self.sampler = sampler
        self.task = task
    
    def __call__(self, action):
        times, frames_idxs, mask = self.sampler(action)
        frames_names = [self.frame_tmpl.format(action.video_id, i) for i in frames_idxs]
        feats = []
        with self.env.begin() as env:
            for f_name in frames_names:
                feat = env.get(f_name.strip().encode('utf-8'))
                if feat is None:
                    print(f_name)
                feat = np.frombuffer(feat, 'float32')
                
                # Transform frame
                if self.transform_feat is not None:
                    feat = self.transform_feat(feat)
                feats += [feat]
            
        # Transform video
        if self.transform_video is not None:
            feats = self.transform_video(feats)
        out = {self.input_name: feats}
        if 'anticipation' in self.task:
            out['mask'] = mask
        out['times'] = times
        out['start_time'] = action.start_time
        out['frames_idxs'] = frames_idxs
        return out
    

class PipeLoaders(object):
    """
    Chain loaders.
    """
    def __init__(self, loader_list):
        self.loader_list = loader_list
        
    def __call__(self, action):
        out = {}
        for loader in self.loader_list:
            out.update(loader(action))
        return out

def get_frames_loader(args):
    sampler = get_sampler(args)
    modalities = args.modalities.split()
    transform_frame = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])
    transform_video = lambda x: torch.stack(x, 1)
    loader_args = {
        'frames_base_path': args.frames_paths[args.ek_version], 
        'fps': args.fps, 
        'input_name': 'frame', 
        'frame_tmpl': '{}_frame_{:010d}.jpg', 
        'sample_mode': args.sample_mode, 
        'num_frames_per_action': args.num_frames_per_action,
        'transform_frame': transform_frame,
        'transform_video': transform_video,
        'sampler': sampler,
        'task': args.task,
    }
    frame_loaders = {
        'train': FramesLoader(**loader_args) if 'frames' in modalities else None,
        'train_aug': FramesLoader(**loader_args) if 'frames' in modalities else None,
        'validation': FramesLoader(**loader_args) if 'frames' in modalities else None,
        'test': FramesLoader(**loader_args) if 'frames' in modalities else None,
    }
    return frame_loaders

def get_features_loader(args):
    sampler = get_sampler(args)
    modalities = args.modalities.split()
    feat_in_modalities = list({'rgb', 'flow', 'obj'}.intersection(set(modalities)))
    transform_feat = lambda x: torch.tensor(x.copy())
    transform_video = lambda x: torch.stack(x, 0) # [T, D]
    loader_args = {
        'feature_base_path': args.features_paths[args.ek_version], 
        'fps': args.fps,  
        'frame_tmpl': '{}_frame_{:010d}.jpg', 
        'sample_mode': args.sample_mode if args.task == 'recognition' else None, 
        'num_frames_per_action': args.num_frames_per_action if args.task == 'recognition' else None,
        'transform_feat': transform_feat,
        'transform_video': transform_video,
        'sampler': sampler,
        'task': args.task,
    }
    feat_loader_list = []
    for modality in feat_in_modalities:
        feat_loader = FeaturesLoader(input_name=modality, **loader_args)
        feat_loader_list += [feat_loader]
    feat_loaders = {
        'train': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
        'train_aug': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
        'validation': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
        'test': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
    }
    return feat_loaders

def get_loaders(args):
    
    loaders = {
        'train': [],
        'train_aug': [],
        'validation': [],
        'test': [],
    }
    
    # Frames loader
    if 'frame' in args.modalities:
        frame_loaders = get_frames_loader(args)
        for k, l in frame_loaders.items():
            if l is not None:
                loaders[k] += [l]
            
    # Features
    feat_loaders = get_features_loader(args)
    for k, l in feat_loaders.items():
        if l is not None:
            loaders[k] += [l]
        
    for k, l in loaders.items():
        loaders[k] = PipeLoaders(l)
    return loaders
