import os
import json
import torch
import argparse
import numpy as np
from argparse import Namespace
from tensorboardX import SummaryWriter
import trainer as T
from utils.train_utils import init_datasets, get_training_model
from utils.core_utils import set_logger, save_configs
from utils.checkpoints_manager import CheckpointsManager
from datasources.EVEDataset import EVEDatasetTrain, EVEDatasetVal
from datasources.Augmenter import get_transformations_list
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config):

    logger = set_logger(config.save_path)

    ###############
    # load datasets
    logger.info('------- Initializing dataloaders --------')

    train_data_class = EVEDatasetTrain
    val_data_class = EVEDatasetVal

    train_dataset_paths = [
        ('eve_train', train_data_class, config.datasrc_eve, config.train_stimuli, config.train_cameras),
    ]
    validation_dataset_paths = [
        ('eve_val', val_data_class, config.datasrc_eve, config.test_stimuli, config.test_cameras),
    ]

    data_transforms = get_transformations_list(config.eyes_size[0] if config.camera_frame_type == 'eyes'
                                               else config.face_size[0])

    train_data, test_data = init_datasets(train_dataset_paths, validation_dataset_paths, config, logger,
                                          num_positives=config.num_positives,
                                          is_load_label=config.is_load_label,
                                          transforms=data_transforms, batch_sampler=config.same_person)

    ###############
    # initialize network and checkpoint manager

    logger.info('------- Initializing model --------')

    trainer = get_training_model(args=config, device=device)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, trainer.model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    checkpoint_manager = CheckpointsManager(network=trainer.model, output_dir=config.save_path)

    if config.load_step != 0:
        logger.info('Loading available model at step {}'.format(config.load_step))
        checkpoint_manager.load_checkpoint(config.load_step)

    #####################################################
    # initialize optimizer, scheduler and tensorboard

    tensorboard = SummaryWriter(logdir=config.save_path)

    if config.opt == 'adam':
        optimizer = torch.optim.Adam(
                params=trainer.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params=trainer.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config.num_iter, eta_min=0, last_epoch=-1)

    #####################################################
    # call trainer

    T.trainer(trainer, train_data, test_data, logger, config, optimizer,
              scheduler, checkpoint_manager, tensorboard, device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GazeCLR')

    parser.add_argument('--config_json', type=str, default=None, help='Path to config in JSON format')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save files')
    parser.add_argument('--num_positives', type=int, default=3, help='number of +ve samples for multi-view learning')
    parser.add_argument('--is_load_label', action='store_true', help='true for loading rotation matrices')
    parser.add_argument('--same_person', action='store_true', help='true if same person batch')
    parser.add_argument('--camera_frame_type', default="face", type=str, help='type of input')

    # training args
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--train_data_workers', type=int, default=16, help='train_data_workers')
    parser.add_argument('--test_data_workers', type=int, default=2, help='test_data_workers')

    parser.add_argument('--opt', default="sgd", type=str, help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='WD',
                        help='weight_decay (default: 0.0005)')

    # architecture args
    parser.add_argument('--arch', default="resnet18", type=str, help='arch')

    # others
    parser.add_argument('--num_iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--load_step', type=int, default=0, help='load step')
    parser.add_argument('--save_interval', type=int, default=10000, help='save_interval')
    parser.add_argument('--print_freq_train', type=int, default=10, help='print_freq_train')
    parser.add_argument('--print_freq_test', type=int, default=1000, help='print_freq_test')
    parser.add_argument('--skip_training', action='store_true', help='skip_training')

    args = parser.parse_args()

    ###############
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    ###############

    # load default arguments
    default_config = json.load(open('configs/default.json'))

    # load specific configurations
    assert os.path.isfile(args.config_json)
    print('Loading ' + args.config_json)

    model_specific_config = json.load(open(args.config_json))
    args = vars(args)

    # merging configurations
    config = {**default_config, **model_specific_config, **args}
    config = Namespace(**config)

    os.makedirs(config.save_path, exist_ok=True)
    # writing config
    save_configs(config.save_path, config)
    print('Written Config file at %s' % config.save_path)

    ###############

    main(config)

