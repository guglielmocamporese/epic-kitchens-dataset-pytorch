##################################################
# Imports
##################################################

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Configs
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_PATH = {
    'ek55': os.path.join(SCRIPT_PATH, 'annotations/epic-kitchens-55-annotations'),
    'ek100': os.path.join(SCRIPT_PATH, 'annotations/epic-kitchens-100-annotations'),
}
RULSTM_ANNOTATIONS_PATH = {
    'ek55': os.path.join(SCRIPT_PATH, 'annotations/rulstm/RULSTM/data/ek55'),
    'ek100': os.path.join(SCRIPT_PATH, 'annotations/rulstm/RULSTM/data/ek100'),
}

# Utils
def timestr2sec(t_str):
    """
    Convert the hh:mm:ss.SSS time format to seconds (in float) format.
    """
    hh, mm, ss = [float(x) for x in t_str.split(':')]
    t_sec = hh * 3600.0 + mm * 60.0 + ss
    return t_sec

def read_rulstm_splits(rulstm_annotation_path):
    """
    Read the RULSTM dataset splits.
    """
    header = ['uid', 'video_id', 'start_frame', 'stop_frame', 'verb_class', 'noun_class', 'action_class']
    df_train = pd.read_csv(os.path.join(rulstm_annotation_path, 'training.csv'), names=header)
    df_validation = pd.read_csv(os.path.join(rulstm_annotation_path, 'validation.csv'), names=header)
    return df_train, df_validation

def str2list(s, out_type=None):
    """
    Convert a string "[i1, i2, ...]" of items into a list [i1, i2, ...] of items.
    """
    s = s.replace('[', '').replace(']', '')
    s = s.replace('\'', '')
    s = s.split(', ')
    if out_type is not None:
        s = [out_type(ss) for ss in s]
    return s

def split_train_val(df, validation_ratio=0.2, use_rulstm_splits=False, rulstm_annotation_path=None):
    """
    Split the train dataset into train and validation.
    """
    if use_rulstm_splits:
        assert rulstm_annotation_path is not None
        df_train_rulstm, df_validation_rulstm = read_rulstm_splits(rulstm_annotation_path)
        uids_train = df_train_rulstm['uid'].values.tolist()
        uids_validation = df_validation_rulstm['uid'].values.tolist()
        df_train = df.loc[df['uid'].isin(uids_train)]
        df_validation = df.loc[df['uid'].isin(uids_validation)]
    else:
        if validation_ratio == 0.0:
            df_train = df
            df_validation = pd.DataFrame(columns=df.columns)
        elif validation_ratio == 1.0:
            df_train = pd.DataFrame(columns=df.columns)
            df_validation = df
        elif 0.0 < validation_ratio and validation_ratio < 1.0:
            df_train = df
            df_train, df_validation = train_test_split(df, test_size=validation_ratio,
                                                       random_state=3577,
                                                       shuffle=True, stratify=df['participant_id'])
        else:
            raise Exception(f'Error. Validation "{validation_ratio}" not supported.')
    return df_train, df_validation

def create_actions_df(ek_version, out_path='actions.csv', use_rulstm_splits=True):
    """
    Save actions.csv with actions labels.
    """
    if use_rulstm_splits:
        if ek_version == 'ek55':
            df_actions = pd.read_csv(os.path.join(RULSTM_ANNOTATIONS_PATH['ek55'], 'actions.csv'))
        elif ek_version == 'ek100':
            df_actions = pd.read_csv(os.path.join(RULSTM_ANNOTATIONS_PATH['ek100'], 'actions.csv'))
            df_actions['action'] = df_actions.action.map(lambda x: x.replace(' ', '_'))

        df_actions['verb_class'] = df_actions.verb
        df_actions['noun_class'] = df_actions.noun
        df_actions['verb'] = df_actions.action.map(lambda x: x.split('_')[0])
        df_actions['noun'] = df_actions.action.map(lambda x: x.split('_')[1])
        df_actions['action'] = df_actions.action
        df_actions['action_class'] = df_actions.id
        del df_actions['id']

    else:
        if ek_version == 'ek55':
            df_train = get_ek55_annotation('train', raw=True)
            df_validation = get_ek55_annotation('validation', raw=True)
            df = pd.concat([df_train, df_validation])
            df.sort_values(by=['uid'], inplace=True)

        elif ek_version == 'ek100':
            df_train = get_ek100_annotation('train', raw=True)
            df_validation = get_ek100_annotation('validation', raw=True)
            df = pd.concat([df_train, df_validation])
            df.sort_values(by=['narration_id'], inplace=True)

        noun_classes = df.noun_class.values
        nouns = df.noun.values
        verb_classes = df.verb_class.values
        verbs = df.verb.values

        actions_combinations = [f'{v}_{n}' for v, n in zip(verb_classes, noun_classes)]
        actions = [f'{v}_{n}' for v, n in zip(verbs, nouns)]

        df_actions = {'verb_class': [], 'noun_class': [], 'verb': [], 'noun': [], 'action': []}
        vn_combinations = []
        for i, a in enumerate(actions_combinations):
            if a in vn_combinations:
                continue

            v, n = a.split('_')
            v = int(v)
            n = int(n)
            df_actions['verb_class'] += [v]
            df_actions['noun_class'] += [n]
            df_actions['action'] += [actions[i]]
            df_actions['verb'] += [verbs[i]]
            df_actions['noun'] += [nouns[i]]
            vn_combinations += [a]
        df_actions = pd.DataFrame(df_actions)
        df_actions.sort_values(by=['verb_class', 'noun_class'], inplace=True)
        df_actions['action_class'] = range(len(df_actions))

    df_actions.to_csv(out_path, index=False)
    print(f'Saved file at "{out_path}".')

def show_sample(action, loader):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    out = loader(action)
    assert 'frames' not in out.keys()

    x_grid = make_grid(out['frame'].transpose(0, 1), nrow=4)

    print(action.action)
    plt.figure(figsize=(12, 7))
    plt.imshow(x_grid.permute(1, 2, 0), aspect='auto')
    plt.show()

def get_ek55_annotation(partition, validation_ratio=0.2, use_rulstm_splits=False, raw=False):

    # Load action labels
    if partition in ['train', 'validation']:

        # Here we load the train, and we have to split into train and validation later
        csv_path = os.path.join(ANNOTATIONS_PATH['ek55'], 'EPIC_train_action_labels.csv')
        df = pd.read_csv(csv_path)
        df_train, df_validation = split_train_val(df, validation_ratio=validation_ratio,
                                                  use_rulstm_splits=use_rulstm_splits,
                                                  rulstm_annotation_path=RULSTM_ANNOTATIONS_PATH['ek55'])
        df = df_train if partition == 'train' else df_validation
        if not use_rulstm_splits:
            df.sort_values(by=['uid'], inplace=True)

    elif partition == 'test_s1':
        csv_path = os.path.join(ANNOTATIONS_PATH['ek55'], 'EPIC_test_s1_timestamps.csv')
        df = pd.read_csv(csv_path)

    elif partition == 'test_s2':
        csv_path = os.path.join(ANNOTATIONS_PATH['ek55'], 'EPIC_test_s2_timestamps.csv')
        df = pd.read_csv(csv_path)
    else:
        raise Exception(f'Error. Partition "{partition}" not supported.')

    if raw:
        return df

    # Load labels csv
    df_verbs = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek55'], 'EPIC_verb_classes.csv'))
    df_nouns = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek55'], 'EPIC_noun_classes.csv'))
    actions_df_path = os.path.join(ANNOTATIONS_PATH['ek55'], 'actions.csv')
    if not os.path.exists(actions_df_path):
        create_actions_df('ek55', out_path=actions_df_path, use_rulstm_splits=True)
    df_actions = pd.read_csv(actions_df_path)

    # Process dataframe
    df['start_time'] = df['start_timestamp'].map(lambda t: timestr2sec(t))
    df['stop_time'] = df['stop_timestamp'].map(lambda t: timestr2sec(t))
    if 'test' not in partition:
        action_classes = []
        actions = []
        for _, row in df.iterrows():
            v, n = row.verb_class, row.noun_class
            df_a_sub = df_actions[(df_actions['verb_class'] == v) & (df_actions['noun_class'] == n)]
            a_cl = df_a_sub['action_class'].values
            a = df_a_sub['action'].values
            if len(a_cl) > 1:
                print(a_cl)
            action_classes += [a_cl[0]]
            actions += [a[0]]
        df['action_class'] = action_classes
        df['action'] = actions
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x))
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int))

    # Remove this for avoiding wrong [time - frame] correspondance (different fps for different videos...)
    del df['stop_frame']
    del df['start_frame']
    return df

def get_ek100_annotation(partition, validation_ratio=0.2, use_rulstm_splits=False, raw=False):
    # Load action labels
    if partition in 'train':
        df = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_train.csv'))
        uids = np.arange(len(df))

    elif partition in 'validation':
        df_train = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_train.csv'))
        df = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_validation.csv'))
        uids = np.arange(len(df)) + len(df_train)

    elif partition == 'test':
        df_train = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_train.csv'))
        df_validation = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_validation.csv'))
        df = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_test_timestamps.csv'))
        uids = np.arange(len(df)) + len(df_train) + len(df_validation)

    else:
        raise Exception(f'Error. Partition "{partition}" not supported.')
    if raw:
        return df

    # Load labels csv
    df_verbs = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_verb_classes.csv'))
    df_nouns = pd.read_csv(os.path.join(ANNOTATIONS_PATH['ek100'], 'EPIC_100_noun_classes.csv'))
    actions_df_path = os.path.join(ANNOTATIONS_PATH['ek100'], 'actions.csv')
    if not os.path.exists(actions_df_path):
        create_actions_df('ek100', actions_df_path)
    df_actions = pd.read_csv(actions_df_path)

    # Process dataframe
    df['start_time'] = df['start_timestamp'].map(lambda t: timestr2sec(t))
    df['stop_time'] = df['stop_timestamp'].map(lambda t: timestr2sec(t))
    df['uid'] = uids
    if 'test' not in partition:
        action_classes = []
        actions = []
        for _, row in df.iterrows():
            v, n = row.verb_class, row.noun_class
            df_a_sub = df_actions[(df_actions['verb_class'] == v) & (df_actions['noun_class'] == n)]
            a_cl = df_a_sub['action_class'].values
            a = df_a_sub['action'].values
            if len(a_cl) > 1:
                print(a_cl)
            action_classes += [a_cl[0]]
            actions += [a[0]]
        df['action_class'] = action_classes
        df['action'] = actions
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x))
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int))

    # Remove this for avoiding wrong [time - frame] correspondance (different fps for different videos...)
    del df['stop_frame']
    del df['start_frame']
    return df
