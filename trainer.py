import torch
import numpy as np
from utils.core_utils import RunningStatistics
from utils.train_utils import get_training_batches


def evaluation(test_data, trainer, curr_step, tensorboard, device, logger):
    test_losses = RunningStatistics()
    trainer.model.eval()
    torch.cuda.empty_cache()

    for tag, data_dict in test_data.items():
        with torch.no_grad():
            for i, input_data in enumerate(data_dict['dataloader']):

                # Move tensors to GPU
                for k, v in input_data.items():
                    if isinstance(v, torch.Tensor):
                        input_data[k] = v.detach().to(device, non_blocking=True)

                v_loss_dict, _ = trainer.compute_losses(input_data)

                for k, v in v_loss_dict.items():
                    test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()
    logger.info('Test Losses at [%7d]: %s' %
                 (curr_step, ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    for k, v in test_loss_means.items():
        tensorboard.add_scalar('test/%s' % k, v, curr_step)


def trainer(trainer, train_data, test_data, logger, config, optimizer, scheduler,
            checkpoint_manager, tensorboard, device):

    logger.info('Training')
    running_losses = RunningStatistics()

    max_dataset_len = np.amax([len(data_dict['dataset']) for data_dict in train_data.values()])

    for current_step in range(config.load_step, config.num_iter):

        torch.cuda.empty_cache()

        if current_step % config.save_interval == 0 and current_step != config.load_step:
            checkpoint_manager.save_checkpoint(current_step)

        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value

        trainer.model.train()
        trainer.model.zero_grad()

        # get input data
        full_input_dict = get_training_batches(train_data, device)
        assert len(full_input_dict) == 1
        full_input_dict = next(iter(full_input_dict.values()))

        loss_dict, output_dict = trainer.compute_losses(full_input_dict)

        loss = loss_dict['total_loss']

        # backward
        loss.backward()
        optimizer.step()

        scheduler.step()

        for k, v in loss_dict.items():
            running_losses.add('%s' % k, v.item())

        if current_step != 0 and (current_step % config.print_freq_train == 0):
            logger.info('Losses at Step [%8d|%8d] Epoch [%.2f]: %s' % (current_step, config.num_iter,
                                                                       current_epoch, running_losses))
            # log to tensorboard
            for k, v in running_losses.means().items():
                tensorboard.add_scalar('train/%s' % k, v, current_step)

            for l in range(full_input_dict['img_a'].shape[1]):
                tensorboard.add_image('train/image_{}'.format(l),
                                      torch.clamp((full_input_dict['img_a'].squeeze()[0, l]) * (255.0 / 1.0), 0,
                                                  255).type(torch.cuda.ByteTensor), current_step)

        if current_step % config.print_freq_test == 0 and current_step > 0:
            torch.cuda.empty_cache()
            # test
            evaluation(test_data, trainer, current_step, tensorboard, device, logger)
            torch.cuda.empty_cache()

    logger.info('Finished Training')
    checkpoint_manager.save_checkpoint(current_step + 1)
