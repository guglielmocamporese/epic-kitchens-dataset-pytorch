# EPIC-Kitchens PyTorch Dataset

A PyTorch dataset that reads the EPIC-Kitchens-55 and the EPIC-Kitchens-100.
In particular, it handles frames and features from RULSTM for both the *action recignition* and the *action anticipation* tasks.

# Simple Usage Example

```python

# Arguments
class Args:

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

# Get args
args = Args()


# EPIC-Kitchens Dataloaders
from dataloaders import get_dataloaders
dataloaders = get_dataloaders(args)
```
# Action Recognition

# Action Anticipation
