
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
import numpy as np
from datasources.sampler import BatchSampler


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_training_batches(train_data_dicts, device):
    """Get training batches of data from all training data sources."""
    out = {}
    for tag, data_dict in train_data_dicts.items():
        if 'data_iterator' not in data_dict:
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
        # Try to get data
        while True:
            try:
                out[tag] = next(data_dict['data_iterator'])
                break
            except StopIteration:
                del data_dict['data_iterator']
                torch.cuda.empty_cache()
                data_dict['data_iterator'] = iter(data_dict['dataloader'])

        # Move tensors to GPU
        for k, v in out[tag].items():
            if isinstance(v, torch.Tensor):
                out[tag][k] = v.detach()
                if k != 'screen_full_frame':
                    out[tag][k] = out[tag][k].to(device, non_blocking=True)
            else:
                out[tag][k] = v
    return out


def init_datasets(train_specs, test_specs, config, logger, num_positives=0,
                  transforms=None, is_load_label=False, batch_sampler=False):

    # Initialize training datasets
    train_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in train_specs:
        dataset = dataset_class(path, config=config,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli,
                                num_positives=num_positives,
                                transforms=transforms,
                                is_load_label=is_load_label)
        dataset.original_full_dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                collate_fn=my_collate
                                )
        if batch_sampler:
            bs = BatchSampler(dataset.index_to_id, batch_size=config.batch_size, shuffle=False, drop_last=True)
            dataloader = DataLoader(dataset,
                                    batch_sampler=bs,
                                    num_workers=config.train_data_workers,
                                    pin_memory=True, collate_fn=my_collate)

        train_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use training dataset: %s' % tag)
        logger.info('          with number of videos: %d' % len(dataset))

    # Initialize test datasets
    test_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in test_specs:
        # Get the full dataset
        dataset = dataset_class(path, config=config,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli,
                                num_positives=num_positives,
                                transforms=transforms,
                                is_load_label=is_load_label)

        dataset.original_full_dataset = dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(dataset) > num_subset:
            subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:num_subset]))
            subset.original_full_dataset = dataset
            dataset = subset

        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.test_data_workers,
                                pin_memory=True,
                                collate_fn=my_collate, drop_last=True
                                )

        test_data[tag] = {
            'dataset': dataset,
            'dataset_class': dataset_class,
            'dataset_path': path,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use evaluation dataset: %s' % tag)
        logger.info('           with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('     of which we evaluate on just: %d' % len(dataset))

    return train_data, test_data


def get_training_model(args, device):

    if args.model == 'gazeclr':
        from nets.gazeclr import TrainGazeCLR
        model = TrainGazeCLR(args, device)

    elif args.model == 'gazeclrinvequiv':
        from nets.gazeclr_inv_equiv import TrainGazeCLRInvEq
        model = TrainGazeCLRInvEq(args, device)

    else:
        raise NotImplementedError

    return model
