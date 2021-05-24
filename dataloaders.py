##################################################
# Imports
##################################################

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Custom
from .utils import get_ek55_annotation, get_ek100_annotation
from .input_loaders import get_loaders


##################################################
# Action on the EK Dataset
##################################################

class EpicAction(object):
    """
    Definition of an action in the EK dataset.
    """
    def __init__(self, uid, participant_id, video_id, narration,
                 start_timestamp, stop_timestamp, verb, verb_class,
                 noun, noun_class, all_nouns, all_noun_classes, start_time,
                 stop_time, fps, partition, video_duration, action, action_class):
        self.uid = uid
        self.participant_id = participant_id
        self.video_id = video_id
        self.narration = narration
        self.start_timestamp = start_timestamp
        self.stop_timestamp = stop_timestamp
        self.verb = verb
        self.verb_class = verb_class
        self.noun = noun
        self.noun_class = noun_class
        self.all_nouns = all_nouns
        self.all_noun_classes = all_noun_classes
        self.start_time = start_time
        self.stop_time = stop_time
        self.fps = fps
        self.partition = partition
        self.video_duration = video_duration
        self.action = action
        self.action_class = action_class

        self.duration = self.stop_time - self.start_time

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)


##################################################
# Video on the EK Dataset
##################################################

class EpicVideo(object):
    """
    Definition of a video in the EK dataset.
    """
    def __init__(self, df_video, fps, partition, task='recognition', t_ant=None):
        self.df = df_video
        self.fps = fps
        self.partition = partition
        self.task = task
        self.t_ant = t_ant

        self.actions, self.actions_invalid = self._get_actions()
        self.duration = max([a.stop_time for a in self.actions])

    def _get_actions(self):
        actions = []
        actions_invalid = []
        video_duration = self.df['stop_time'].values.max()
        for _, row in self.df.iterrows():
            action_args = {
                'uid': row.uid,
                'participant_id': row.participant_id,
                'video_id': row.video_id,
                'narration': row.narration if 'test' not in self.partition else None,
                'start_timestamp': row.start_timestamp,
                'stop_timestamp': row.stop_timestamp,
                'verb': row.verb if 'test' not in self.partition else None,
                'verb_class': row.verb_class if 'test' not in self.partition else None,
                'noun': row.noun if 'test' not in self.partition else None,
                'noun_class': row.noun_class if 'test' not in self.partition else None,
                'all_nouns': row.all_nouns if 'test' not in self.partition else None,
                'all_noun_classes': row.all_noun_classes if 'test' not in self.partition else None,
                'start_time': row.start_time,
                'stop_time': row.stop_time,
                'fps': self.fps,
                'partition': self.partition,
                'video_duration': video_duration,
                'action': row.action,
                'action_class': row.action_class,
            }
            action = EpicAction(**action_args)
            if self.task == 'recognition':
                actions += [action]
            elif self.task == 'anticipation':
                assert self.t_ant is not None
                assert self.t_ant > 0.0
                if action.start_time - self.t_ant >= 0:
                    actions += [action]
                else:
                    actions_invalid += [action]
        return actions, actions_invalid


##################################################
# EpicKitchens Dataset
##################################################

class EpicDataset(Dataset):
    def __init__(self, df, partition, fps=5.0, loader=None, task='recognition', t_ant=None):
        super().__init__()
        self.partition = partition
        self.fps = fps
        self.df = df
        self.loader = loader
        self.task = task
        self.t_ant = t_ant

        self.videos = self._get_videos()
        self.actions, self.actions_invalid = self._get_actions()

    def _get_videos(self):
        video_ids = sorted(list(set(self.df['video_id'].values.tolist())))
        videos = []
        pbar = tqdm(desc=f'Loading {self.partition} samples', total=len(self.df))
        for video_id in video_ids:
            video_args = {
                'df_video': self.df[self.df['video_id'] == video_id].copy(),
                'fps': self.fps,
                'partition': self.partition,
                'task': self.task,
                't_ant': self.t_ant if self.task == 'anticipation' else None,
            }
            video = EpicVideo(**video_args)
            videos += [video]
            pbar.update(len(video.actions))
        pbar.close()
        return videos

    def _get_actions(self):
        actions = []
        actions_invalid = []
        for video in self.videos:
            actions += video.actions
            actions_invalid += video.actions_invalid
        return actions, actions_invalid

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        a = self.actions[idx]
        sample = {'uid': a.uid}

        # Load frames of feat
        inputs = self.loader(a)
        sample.update(inputs)

        if 'test' not in self.partition:
            sample['verb_class'] = a.verb_class
            sample['noun_class'] = a.noun_class
            sample['action_class'] = a.action_class
        return sample

def get_datasets(args):
    
    # Loaders
    loaders = get_loaders(args)
    
    # Annotations
    annotation_args = {
        'validation_ratio': args.validation_ratio,
        'use_rulstm_splits': args.use_rulstm_splits,
    }

    if args.ek_version == 'ek55':
        dfs = {
            'train': get_ek55_annotation(partition='train', **annotation_args),
            'validation': get_ek55_annotation(partition='validation', **annotation_args),
            'test_s1': get_ek55_annotation(partition='test_s1', **annotation_args),
            'test_s2': get_ek55_annotation(partition='test_s2', **annotation_args),
        }
    elif args.ek_version == 'ek100':
        dfs = {
            'train': get_ek100_annotation(partition='train', **annotation_args),
            'validation': get_ek100_annotation(partition='validation', **annotation_args),
            'test': get_ek100_annotation(partition='test', **annotation_args),
        }
    else:
        raise Exception(f'Error. EPIC-Kitchens Version "{args.ek_version}" not supported.')

    # Datasets
    ds_args = {
        'fps': args.fps,
        'task': args.task,
        't_ant': args.t_ant,
    }
    if args.mode == 'train':
        dss = {
            'train': EpicDataset(df=dfs['train'], partition='train', loader=loaders['train'], **ds_args),
            'train_aug': EpicDataset(df=dfs['train'], partition='train', loader=loaders['train_aug'], **ds_args),
            'validation': EpicDataset(df=dfs['validation'], partition='validation', 
                                      loader=loaders['validation'], **ds_args),
        }
    elif args.mode == 'validation': 
        dss = {
            'validation': EpicDataset(df=dfs['validation'], partition='validation', 
                                      loader=loaders['validation'], **ds_args),
        }
    elif args.mode == 'test':
        
        if args.ek_version == 55:
            dss = {
                'test_s1': EpicDataset(df=dfs['test_s1'], partition='test_s1', loader=loaders['test'], **ds_args),
                'test_s2': EpicDataset(df=dfs['test_s2'], partition='test_s2', loader=loaders['test'], **ds_args),
            }
        elif args.ek_version == 100:
            dss = {
                'test': EpicDataset(df=dfs['test'], partition='test', loader=loaders['test'], **ds_args),
            }
    else:
        raise Exception(f'Error. Mode "{args.mode}" not supported.')
        
    return dss

def get_dataloaders(args):
    
    # Datasets
    dss = get_datasets(args)
       
    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    if args.mode in ['train', 'training']:
        dls = {
            'train': DataLoader(dss['train'], shuffle=False, **dl_args),
            'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
            'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        }
    elif args.mode in ['validate', 'validation', 'validating']:
        dls = {
            'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        }
    elif args.mode == 'test':
        if args.ek_version == 55:
            dls = {
                'test_s1': DataLoader(dss['test_s1'], shuffle=False, **dl_args),
                'test_s2': DataLoader(dss['test_s2'], shuffle=False, **dl_args),
            }
        elif args.ek_version == 100:
            dls = {
                'test': DataLoader(dss['test'], shuffle=False, **dl_args),
            }
    else:
        raise Exception(f'Error. Mode "{args.mode}" not supported.')
    return dls
