# EPIC-Kitchens PyTorch Dataset

A PyTorch dataset that reads the EPIC-Kitchens-55 and the EPIC-Kitchens-100.
In particular, it handles frames and features from RULSTM for both the *action recognition* and the *action anticipation* tasks.

If you use the code of this repo and you find this project useful, please consider to give a star ⭐!

# Action Recognition Usage Example

```python
# Imports
from torchvision import transforms
from input_loaders import ActionRecognitionSampler, FramesLoader, FeaturesLoader, PipeLoaders
from utils import get_ek55_annotation

# Create clip samples and clip loader
sampler = ActionRecognitionSampler(sample_mode='center', num_frames_per_action=16)
loader = PipeLoaders([
    FramesLoader(sampler, 'path/to/frames', fps=5.0, transform_frame=transforms.ToTensor()),
    FeaturesLoader(sampler, 'path/to/features', fps=5.0, input_name='obj'),
])
csv = get_ek55_annotation(partition='train') # Load annotations (dataframe)
ds = EpicDataset(csv, partition='train', loader=loader, task='recognition') # Create the EK dataset

# Get sample
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

# Action Anticipation Usage Example

```python
# Imports
from torchvision import transforms
from input_loaders import ActionAnticipationSampler, FramesLoader, FeaturesLoader, PipeLoaders
from utils import get_ek55_annotation

# Create clip samples and clip loader
sampler = ActionAnticipationSampler(t_buffer=3.5, t_ant=1.0, fps=5.0)
loader = PipeLoaders([
    FramesLoader(sampler, 'path/to/frames', fps=5.0, transform_frame=transforms.ToTensor()),
    FeaturesLoader(sampler, 'path/to/features', fps=5.0, input_name='obj'),
])
csv = get_ek55_annotation(partition='train') # Load annotations (dataframe)
ds = EpicDataset(csv, partition='train', loader=loader, task='recognition') # Create the EK dataset

# Get sample
sample = next(iter(ds))

"""
sample['uid'] -> int
sample['frame'] -> tensor of shape [C, T, H, W]
sample['obj'] -> tensor of shape [T, D]
sample['mask'] -> tensor of shape [T]
sample['noun_class'] -> int
sample['verb_class'] -> int
sample['action_class'] -> int
"""

```

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

