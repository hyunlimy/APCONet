# -*- coding: utf-8 -*-

# import system functions
import gc
import pickle
import time
from copy import deepcopy

# import data handling functions
import pandas as pd

# import modeling functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

# import utility functions
from .apconet_packages.model_functions_inference import *
from .apconet_packages import ranger_optimizer as ranger

class Datasets(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, aline_wave, ashw, ylabels, chart, train_valid_flag=None):
        '''Initialization
        [train_valid_flag]:: binalized value with same length of each parameters (true will selected)
        '''
        self.flag = train_valid_flag
        if self.flag is None:
            self._wave = aline_wave
            self._ashw = ashw
            self._ylabel = ylabels
            self._chart = chart

        else:
            self._wave = aline_wave[self.flag]
            self._ashw = ashw[self.flag]
            self._ylabel = ylabels[self.flag]
            self._chart = chart[self.flag]

        #self.data_x = build_x_none(self.wave, self.ashw)
        self._wave = np.expand_dims(self._wave, axis=2)
        self.wave_pt = torch.from_numpy(np.transpose(self._wave, (0, 2, 1))).float()
        self.ashw_pt = torch.from_numpy(self._ashw).float()
        self.ylabel_pt = torch.unsqueeze(torch.from_numpy(self._ylabel).float(), 1)
        self.chart_pt = self._chart

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ylabel_pt)

    def __getitem__(self, index):
        'Generates one sample of data'
        # build dataset in specific index
        # index is for train and validation specification
        X = [self.ashw_pt[index], self.wave_pt[index]]
        Y = self.ylabel_pt[index]
        c = self.chart_pt[index]

        return X, Y, c

    def get_yvals(self):
        return self.ylabel_pt


def validate(model_, validation_data_generator_, criteria='mape', onGPU=True, test_mode=False):
    """ Functions for both validation and test the model"""

    # check criteria
    if criteria == 'mape':
        criterion = mape_loss
    elif criteria == 'mae':
        criterion = nn.SmoothL1Loss().cuda()
    elif criteria == 'rmse':
        criterion = RMSELoss
    elif criteria == 'smape':
        criterion = smape_loss
    else:
        raise ValueError('Validation criteria error: not supported criteria, current version only support mape or mae')

    _valid_generator = validation_data_generator_


    p_valid = []
    pred_true = []
    c_valid = []
    # idx = np.array(range(_valid_generator.__len__()))
    #
    # if len(idx) > batch_size:
    #     if int(len(idx) % batch_size):
    #         iterate = range(int(len(idx) / batch_size))
    #     else:
    #         iterate = range(int(len(idx) / batch_size) + 1)
    # else:
    #     iterate = range(1)
    if onGPU:
        for _batch_X, _batch_Y, _batch_c in _valid_generator:
            _batch_X = [x.cuda() for x in _batch_X]
            _batch_Y = _batch_Y.cuda()
            if test_mode:
                print('Shape of test batch')
                print(_batch_X[0].size())
                print(_batch_X[1].size())

            model_.eval().cuda()
            val_output = model_(_batch_X)
            val_output = val_output.cpu().data.numpy().astype(float)

            # reason for deepcopy:
            # To prevent "RuntimeError: received 0 items of ancdata" from dataloader
            # It caused by dangling reference on shared object of pytorch dataloader
            p_valid.extend(deepcopy(val_output))
            pred_true.extend(deepcopy(_batch_Y))
            c_valid.extend(deepcopy(_batch_c))
    else:
        for _batch_X, _batch_Y, _batch_c in _valid_generator:
            if test_mode:
                print('Shape of test batch')
                print(_batch_X.size())
            model_.eval().cpu()
            val_output = model_(_batch_X)
            val_output = val_output.cpu().data.numpy().astype(float)

            p_valid.extend(val_output)
            pred_true.extend(_batch_Y)
            c_valid.extend(_batch_c)

    p_valid_var = torch.from_numpy(np.array(p_valid)).float()
    pred_true_var = torch.from_numpy(np.array(pred_true, dtype=np.float32)).float()
    pred_true_var = torch.unsqueeze(pred_true_var, 1)

    val_loss = criterion(p_valid_var, pred_true_var)

    return val_loss, p_valid_var, pred_true_var, c_valid


def train_with_scheduler(Model_architecture, train_data_generator, validatioin_data_generator, epochs,
             learning_rate, patient, criteria, optimizer_type, weight_decay_rate, sgd_momentum, filewrite, odir, log_df,
             valid_interval, batch_size):

    model_pretrain = Model_architecture

    #model_pretrain.apply(weights_init_conv)
    #model_pretrain.apply(weights_init_linear)

    if torch.cuda.is_available():
        model_pretrain.cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), " GPUs.")
            print("Use", torch.cuda.device_count(), " GPUs.", file=filewrite)
            model_pretrain = nn.DataParallel(model_pretrain)

    best_loss_pretrain = np.Inf


    print('Start model training')
    print('--------------------')

    print('Start model training', file=filewrite)
    print('--------------------', file=filewrite)

    # check criteria
    if criteria == 'mape':
        criterion = mape_loss
    elif criteria == 'mae':
        criterion = nn.SmoothL1Loss().cuda()
    elif criteria == 'rmse':
        criterion = RMSELoss
    elif criteria == 'smape':
        criterion = smape_loss
    else:
        raise ValueError('Pre-train criteria error: not supported criteria, current version only support mape or mae')

    # define early stopping for every epoch and every lr
    early_stopping = EarlyStopping(patience=patient, verbose=True, filewrite=filewrite)

    lr = learning_rate
    epo = epochs

    early_stopping.force_init_counter()

    if optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model_pretrain.parameters(), lr=lr, weight_decay=weight_decay_rate)
    elif optimizer_type == 'adadelta':
        optimizer = optim.Adadelta(model_pretrain.parameters(), lr=lr, weight_decay=weight_decay_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model_pretrain.parameters(), lr=lr, weight_decay=weight_decay_rate,
                              momentum=sgd_momentum)
    elif optimizer_type == 'ranger':
        optimizer = ranger.Ranger(model_pretrain.parameters(), lr=lr, weight_decay=weight_decay_rate)
    else:
        if optimizer_type != 'adam':
            print('[W] {} optimizer is not supported in pretrain: Use defalut Adam optimizer'.format(optimizer_type))
        optimizer = optim.Adam(model_pretrain.parameters(), lr=lr, weight_decay=weight_decay_rate)

    _train_generator = train_data_generator
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)

    for epoch in range(epo):
        start_time_epoch = time.time()
        #num_shuffle = np.random.permutation(range(len(Y_train)))
        scheduler.step()

        # code for lstm model
        if model_pretrain.__class__.__name__ == ('CNN_LSTM_inception' or 'CNN_NL_LSTM_inception'):
            model_pretrain.lstm_part.hidden = model_pretrain.lstm_part.init_hidden(batch_size=batch_size)

        for step, (batch_X, batch_Y, batch_c) in enumerate(_train_generator):
            model_pretrain.zero_grad()
            batch_X = [x.cuda() for x in batch_X]
            batch_Y = batch_Y.cuda()

            model_pretrain.train()
            output = model_pretrain(batch_X)
            #loss = mape_loss(output, target_var)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()

            if step % valid_interval == 0:

                current_lr = optimizer.param_groups[0]['lr']

                valid_loss, val_output, val_true, _ = validate(model_=model_pretrain,
                                                            validation_data_generator_=validatioin_data_generator,
                                                            criteria=criteria)

                print("[Epochs: " + str(epoch + 1) + "  step: " + str(step) + " ]  / lr = " + str(current_lr))
                print("  Train loss: " + str(
                    loss.data.cpu().numpy().astype(float)) + "/  Validation loss: " + str(
                    valid_loss.data.numpy()))

                print("[Epochs: " + str(epoch + 1) + "  step: " + str(step) + " ]  / lr = " + str(current_lr),
                      file=filewrite)
                print("  Train loss: " + str(
                    loss.data.cpu().numpy().astype(float)) + "/  Validation loss: " + str(
                    valid_loss.data.numpy()), file=filewrite)

                log_df = log_df.append({'stage': int(0), 'epoch': (epoch + 1), 'lr': current_lr,
                                        'step': step, 'train_loss': loss.data.cpu().numpy(),
                                        'val_loss': valid_loss.data.numpy()}, ignore_index=True)

                if not os.path.exists(odir):
                    os.mkdir(odir)

                if valid_loss.data < best_loss_pretrain:
                    for delkf in glob.glob('{}/model_best_pretrain_vl-*'.format(odir)):
                        try:
                            os.remove(delkf)
                        except:
                            print('Remove file error: {}'.format(delkf))
                            print('Remove file error: {}'.format(delkf), file=filewrite)
                    best_loss_pretrain = valid_loss.data.cpu().numpy()
                    current_pretrain_loss = loss.data.cpu().numpy()

                if step == 0:
                    # None
                    start_time_step = time.time()
                else:
                    elapsed_time_step = time.time() - start_time_step
                    print("  Elapsed Time for {} step: {}s".format(valid_interval, elapsed_time_step))
                    print("  Elapsed Time for {} step: {}s".format(valid_interval, elapsed_time_step), file=filewrite)
                    start_time_step = time.time()

                bestcache_pretrain = '{}/model_best_pretrain_vl-{}.pt'.format(odir,
                                                                             np.around(valid_loss.data.numpy(), 4))

                early_stopping(val_loss=valid_loss, model=model_pretrain, directory=bestcache_pretrain)
                best_loss_pretrain = early_stopping.val_loss_min

                if early_stopping.early_stop:
                    print("Early Stopping @ epoch : {} / step : {} / min val loss : {}".format(epoch, step,
                                                                                               early_stopping.val_loss_min))
                    model_pretrain = early_stopping.checkpoint
                    bestcache_pretrain = early_stopping.checkpoint_dir
                    break

        else:
            model_pretrain = early_stopping.checkpoint
            bestcache_pretrain = early_stopping.checkpoint_dir
            continue
        break

        elapsed_time_epoch = time.time() - start_time_epoch
        print('End training epoch number {} from {} / at {}s'.format(epoch + 1, epo, elapsed_time_epoch))
        print('Current best validation loss = {}'.format(best_loss_pretrain))
        print('--------------------')
        print(' ')

        print('End training epoch number {} from {} / at {}s'.format(epoch + 1, epo, elapsed_time_epoch),
              file=filewrite)
        print('Current best validation loss = {}'.format(best_loss_pretrain), file=filewrite)
        print('--------------------', file=filewrite)
        print(' ', file=filewrite)

    model_pretrain = early_stopping.checkpoint
    bestcache_pretrain = early_stopping.checkpoint_dir

    del batch_X
    del output
    del batch_Y
    del val_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model_pretrain, bestcache_pretrain, best_loss_pretrain, current_pretrain_loss, log_df