

import logging
import os
import torch

ckpt_extension = '.pth.tar'
ckpt_fmtstring = 'at_step_%07d' + ckpt_extension


def step_number_from_fname(fpath):
    fname = fpath.split('/')[-1]
    stem = fname.split('.')[0]
    return int(stem.split('_')[-1])


class CheckpointsManager(object):

    def __init__(self, network, output_dir):
        self.network = network
        self.output_dir = os.path.join(os.path.realpath(output_dir), 'checkpoints')

    @property
    def all_available_checkpoint_files(self):
        if not os.path.isdir(self.output_dir):
            return []
        fpaths = [
            (step_number_from_fname(p), self.output_dir + '/' + p)
            for p in os.listdir(self.output_dir)
            if os.path.isfile(self.output_dir + '/' + p)
            and p.endswith(ckpt_extension)
        ]
        fpaths = sorted(fpaths)  # sort by step number
        return fpaths

    def load_last_checkpoint(self):
        available_fpaths = self.all_available_checkpoint_files
        if len(available_fpaths) > 0:
            step_number, fpath = available_fpaths[-1]
            logging.info('Found weights file: %s' % fpath)
            loaded_step_number = self.load_checkpoint(step_number)
            return loaded_step_number
        return 0

    def load_checkpoint(self, step_number):
        checkpoint_fpath = os.path.join(self.output_dir, ckpt_fmtstring%step_number)
        assert os.path.isfile(checkpoint_fpath)
        weights = torch.load(checkpoint_fpath)

        self.network.load_state_dict(weights)
        logging.info('Loaded known model weights at step %d' % step_number)
        return step_number

    def save_checkpoint(self, step_number):
        assert os.path.isdir(os.path.abspath(self.output_dir + '/../'))
        fname = ckpt_fmtstring % step_number
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        ofpath = '%s/%s' % (self.output_dir, fname)
        torch.save(self.network.state_dict(), ofpath)
        torch.cuda.empty_cache()

    def save_best_checkpoint(self, step_number):
        assert os.path.isdir(os.path.abspath(self.output_dir + '/../'))
        fname = 'best_checkpoint_' + ckpt_fmtstring % step_number
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        ofpath = '%s/%s' % (self.output_dir, fname)
        torch.save(self.network.state_dict(), ofpath)
        torch.cuda.empty_cache()
