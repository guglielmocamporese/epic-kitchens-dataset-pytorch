import json
import torch
from dataloaders import get_dataloaders

class Args(object):
    def __init__(self):
        self.features_paths = {
            'ek55': './annotations/epic-kitchens-55-annotations/features',
            'ek100': './annotations/epic-kitchens-100-annotations/features',
        }
        self.frames_paths = {
            'ek55': './annotations/epic-kitchens-55-annotations/frames',
            'ek100': './annotations/epic-kitchens-100-annotations/frames',
        }
        self.ek_version = 'ek55' # 'ek55' or 'ek100'
        self.fps = 4.0
        
        self.batch_size = 32
        self.num_workers = 0
        
        self.num_frames_per_action = 8
        self.sample_mode = 'uniform'
        
        self.modalities = 'frames rgb flow obj'
        
        self.validation_ratio = 0.2
        self.use_rulstm_splits = False
        
        self.mode = 'train'
        
        self.height = 224
        self.width = 224
        
        self.task = 'anticipation'
        
        self.t_buffer = 3.5
        self.t_ant = 1.0
        
    def __repr__(self):
        return 'Input Args: ' + json.dumps(self.__dict__, indent=4)

if __name__ == '__main__':

    # Get args
    args = Args()
    print(args)

    # Dataloaders
    dls = get_dataloaders(args)

    # Get sample
    sample = next(iter(dls['train'].dataset))
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f'{k}: {v.shape}')
        else:
            print(f'{k}: {v}')

