import os
import torch
import numpy as np
import random
import gym
import commentjson
import copy


def load_config(config_fname):
    """ Load in a config JSON file and return as a dictionary """
    json_config = commentjson.loads(open(config_fname, 'r').read())
    dict_config = DotDic(json_config)

    # Make inner dictionaries indexable like a class
    for key, value in dict_config.items():
        if isinstance(value, dict):
            dict_config[key] = DotDic(value)
    return dict_config


class DotDic(dict):
    """
    Return dict that supports dot notation
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))

    def __init__(self, dct):
        """ Recursively nest the DotDic. """
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDic(value)
            self[key] = value


def set_random_seeds(seed_id, verbose=False):
    """ Set random seed (random, npy, torch, gym) for reproduction """
    os.environ['PYTHONHASHSEED'] = str(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_id)
    random.seed(seed_id)
    np.random.seed(seed_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_id)
        torch.cuda.manual_seed(seed_id)
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed_id)

    if verbose:
        print("-- Random seeds (random, numpy, torch) were set to {}".format(seed_id))


def normalize_time(t, T, min_lim=-1, max_lim=1, horizon=False):
    """ Normalize time input to lie relative in [min_lim, max_lim]. """
    # Decay down from max_lim to min_lim as t -> T
    if horizon:
        return (min_lim - max_lim) * (t - T) / T + min_lim
    # Decay up from min_lim to max_lim as t -> T
    else:
        return (max_lim - min_lim) * t/ T + min_lim


def one_hot_encode(vec_dim, index):
    """ Encode index as one hot - set dim to 1 all others to 0. """
    out = torch.zeros(vec_dim)
    out[index] = 1
    return out


def linearly_anneal(counter, start, final, in_steps):
    """ Linearly anneal quantity down/up in a set of steps """
    if final > start:
        out = min(final, start + counter * (final - start) / in_steps)
    else:
        out = max(final, start + counter * (final - start) / in_steps)
    return out
