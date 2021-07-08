import argparse
import logging
import os, random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import utils
import model.net as net
from evaluate import evaluate
from datasets import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Deep ARIMA on Time series forecasting')

# input
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--dataset', default='synthetic', help='Name of the dataset')
parser.add_argument('--model-name', default='deepar', help='Directory containing params.json')

# output
parser.add_argument('--out-dir', default='results', type=str, help='The name of saved model')


#train/valid split
parser.add_argument('--v_partition', default=0.1, type=float,
                    help='validation_partition')


parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    for i, (train_seq, gt, series_id, scaling_factor) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_seq.shape[0]
        train_seq = train_seq.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        gt = gt.squeeze().permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        series_id = series_id.unsqueeze(0).to(params.device)
        scaling_factor = scaling_factor.to(torch.float32).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.enc_len+params.dec_len):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_seq[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_seq[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(train_seq[t].unsqueeze_(0).clone(), series_id, hidden, cell)
            mu = mu * scaling_factor
            sigma = sigma * scaling_factor
            loss += loss_fn(mu, sigma, gt[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / (params.enc_len+params.dec_len)  # loss per timestep
        loss_epoch[i] = loss
        if i % 1000 == 0:
            logger.info(f'{epoch}-th epoch, {i}-th batch train_loss: {loss:.5f}')
        if i == 0:
            logger.info(f'{epoch}-th epoch, {i}-th batch train_loss: {loss:.5f}')
    return loss_epoch


def train_and_evaluate(params: utils.Params,
                       train_loader: DataLoader,
                       valid_loader: DataLoader,
                       test_loader: DataLoader,
                       model: nn.Module,
                       save_best: str=None,
                       restore_file: str=None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        params: (Params) hyperparameters
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        valid_loader: load validate data and labels
        test_loader: load test data and labels
        save_best: (string) optional
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    writer = SummaryWriter(params.out_dir)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # fetch loss function
    loss_fn = net.loss_fn

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.out_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)


    logger.info('begin training and evaluation')

    best_valid_ND = float('inf')
    best_test_ND = float('inf')
    best_epoch = -1
    train_len = len(train_loader)
    valid_ND_summary = np.zeros(params.num_epochs)
    test_ND_summary = np.zeros(params.num_epochs)
    train_loss_summary = np.zeros((train_len * params.num_epochs))

    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, params.num_epochs))
        # training
        train_loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, epoch)
        writer.add_scalar('training/train_loss', train_loss_summary[epoch * train_len:(epoch + 1) * train_len].mean(), epoch)

        # validation and test
        valid_metrics = evaluate(model, valid_loader, params, "valid"+str(epoch))
        test_metrics = evaluate(model, test_loader, params, "test"+str(epoch))
        writer.add_scalars('validation/', valid_metrics, epoch)
        writer.add_scalars('test/', test_metrics, epoch)
        #for k,v in valid_metrics.items():
        #    writer.add_scalar('validation/'+k, valid_metrics[k], epoch)
        #for k,v in test_metrics.items():
        #    writer.add_scalar('test /'+k, test_metrics[k], epoch)
        #test_ND_summary[epoch] = test_metrics['ND']

        if valid_metrics['ND'] < best_valid_ND:
            logger.info('- Found new best ND')
            best_valid_ND = valid_metrics['ND']
            best_test_ND = test_metrics['ND']
            logger.info('Current Best ND is: %.5f' % best_valid_ND)
            loss_is_best = True
            best_epoch = epoch
        else:
            loss_is_best = False

        if (epoch - best_epoch >= params.early_stop_ep):
            print("Achieve early_stop_ep and current epoch is", epoch)
            break

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=loss_is_best,
                              checkpoint=params.out_dir)

        utils.plot_all_epoch(test_ND_summary[:epoch + 1], args.dataset + '_ND', params.plot_dir)
        utils.plot_all_epoch(train_loss_summary[:(epoch + 1) * train_len], args.dataset + '_loss', params.plot_dir)

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = args.search_params.split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()
        utils.plot_all_epoch(test_ND_summary, print_params + '_ND', location=params.plot_dir)
        utils.plot_all_epoch(train_loss_summary, print_params + '_loss', location=params.plot_dir)

if __name__ == '__main__':

    # load the parameters from json file
    args = parser.parse_args()
    model_name = args.model_name
    json_path = os.path.join('conf', f'{model_name}.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path, args.dataset)

    # add output dir path, plot dir path, log path
    model_name += "_" + str(int(time.time()))
    params.out_dir = os.path.join(args.out_dir, model_name)
    params.plot_dir = os.path.join(params.out_dir, 'figures')
    params.log_path = os.path.join(params.out_dir, 'train.log')
    params.device = device

    # create missing directories
    os.makedirs(params.out_dir, exist_ok=True)
    os.makedirs(params.plot_dir, exist_ok=True)

    # fix training and validation set
    np.random.seed(0)
    num_train = params.train_ins_num
    indices = list(range(num_train))
    split = int(args.v_partition * num_train)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    # set random seeds for reproducible experiments if necessary
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    logging.basicConfig(level=logging.INFO,
                        filename=params.log_path, filemode='w',
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load the dataset
    logger.info('Loading the datasets...')
    if(args.dataset=='synthetic'):
        train_set = Synthetic(data_dir, num_train, pred_days=params.pred_days, overlap=params.overlap,
                              win_len=params.enc_len+params.dec_len, enc_len=params.enc_len)
        params.seq_num = train_set.points.shape[1]
        test_set = SyntheticTest(train_set.points, train_set.covariates, train_set.withhold_len, params.enc_len, params.dec_len)
    else:
        raise NameError('Currently, we only support synthetic, traffic, ele, m4, solar and wind')
    logger.info('Loading complete.')

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=4, sampler=train_sampler)
    if(params.eval_batch_size ==-1):
        eval_batch_size = params.batch_size
    else:
        eval_batch_size = params.eval_batch_size

    valid_loader = DataLoader(train_set, batch_size=eval_batch_size, num_workers=4, sampler =valid_sampler)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, num_workers=4)

    # specify the model
    model = net.DeepAR(params).to(device)
    # parallel training
    #model = nn.DataParallel(model).to(device)
    logger.info(f'Model: \n{str(model)}')
    logger.info(f'Arguments: \n{str(params)}')
    logger.info(f"use {device}")


    #
    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(params,
                       train_loader,
                       valid_loader,
                       test_loader,
                       model,
                       args.save_best,
                       args.restore_file)

