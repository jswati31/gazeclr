
import numpy as np
import os
import torch
from collections import OrderedDict
from torch.nn import functional as F
import yaml
import logging


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) if len(v) > 0 else (k, v) for k, v in self.losses.items()
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []

    def __str__(self):
        fmtstr = ', '.join(['%s: %.6f (%.6f)' % (k, v[-1], self.means()[k]) for k, v in self.losses.items()])
        return fmtstr


def recover_images(x):
    x = x.cpu().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
        x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    else:
        x = np.transpose(x, [1, 2, 0])  # CHW to HWC
        x = x[:, :, ::-1]  # RGB to BGR for OpenCV
    return x


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def np_pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def np_vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = np_pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = np_pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)

    return np.arccos(similarity) * radians_to_degrees


def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.view(-1, 3)
        norm_a = torch.div(a, torch.norm(a, dim=1).view(-1, 1) + 1e-7)
        return torch.stack([
            torch.asin(norm_a[:, 1]),
            torch.atan2(norm_a[:, 0], norm_a[:, 2]),
        ], dim=1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def save_configs(save_path, config):
    target_dir = save_path + '/configs'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    fpath = os.path.relpath(target_dir + '/params.yaml')
    with open(fpath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def set_logger(save_path):

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(save_path, 'training.log'))
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging
