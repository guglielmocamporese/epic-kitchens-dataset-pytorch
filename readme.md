# EPIC-Kitchens PyTorch Dataset

A PyTorch dataset that reads the EPIC-Kitchens-55 and the EPIC-Kitchens-100.
In particular, it handles frames and features from RULSTM for both the *action recognition* and the *action anticipation* tasks.

# Simple Usage Example

```python
# Imports
from dataloaders import get_dataloaders

# Arguments
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
        self.fps = 4.0 # sample rate to consider
        self.batch_size = 32
        self.num_workers = 0
        self.num_frames_per_action = 8
        self.sample_mode = 'uniform' # 'uniform', 'center' or  'center_hard'
        self.modalities = 'frames rgb flow obj' # choose a combination of the inputs
        self.validation_ratio = 0.2
        self.use_rulstm_splits = False
        self.mode = 'train' # 'train', 'validation' or 'test'
        self.height = 224 # resized height
        self.width = 224 # resized width
        self.task = 'anticipation' # 'recognition' or 'anticipation'
        self.t_buffer = 3.5 # anticipation buffer before the action, in sec
        self.t_ant = 1.0 # anticipation time, in sec

# Get args
args = Args()

# EPIC-Kitchens Dataloaders
dataloaders = get_dataloaders(args)
```
# Action Recognition

```python
# Imports
from torchvision import transforms
from input_loaders import ActionRecognitionSampler, FramesLoader, FeaturesLoader, PipeLoaders
from utils import get_ek55_annotation

# Create clip samples and frame loader
sampler = ActionRecognitionSampler(sample_mode='center', num_frames_per_action=16)
loader = PipeLoaders([
    FramesLoader(sampler, 'path/to/frames', fps=5.0, transform_frame=transforms.ToTensor()),
    FeaturesLoader(sampler, 'path/to/features', fps=5.0, input_name='obj'),
])
ann = get_ek55_annotation(partition='train')
ds = EpicDataset(ann, partition='train', loader=loader, task='recognition')

sample = next(iter(ds))

"""
sample['uid'] -> int
sample['frame'] -> tensor of shape [C, T, H, W]
sample['obj'] -> tensor of shape [T, D]
sample['noun_class'] -> int
sample['verb_class'] -> int
sample['action_class'] -> int
"""

```

# Action Anticipation

# Install

For the installation, you have to clone this repo and download the annotations as follows:

```sh
# Clone the project
$ git clone https://github.com/guglielmocamporese/epic-kitchens-dataset-pytorch.git ek_datasets

# Go to the project folder
$ cd ek_datasets

# Download the annotations
$ ./setup_annotations.sh
```

