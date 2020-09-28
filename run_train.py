# -*- coding: utf-8 -*-

import os
import gc
import time
import re

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data


from . import trainer
from .trainer import Datasets
from .apconet_packages.model_functions_inference import Inception1DNet_NL_compact_dilated

def load_datasets(dataset_dir, version, custom_set = False):

    if custom_set:

        print("Load version: {} ".format(version))

        with open('{}/np_w_{}'.format(dataset_dir, version), 'rb') as f:
            all_w = pickle.load(f)

        with open('{}/np_sv_{}'.format(dataset_dir, version), 'rb') as f:
            all_svs = pickle.load(f)

        with open('{}/np_a_{}'.format(dataset_dir, version), 'rb') as f:
            all_a = pickle.load(f)

        with open('{}/np_c_{}'.format(dataset_dir, version), 'rb') as f:
            all_c = pickle.load(f)

        return all_w, all_svs, all_a, all_c

    else:
        with open(dataset_dir + '/apconet_packages/sample_wave.np', 'rb') as f:
            sample_wave = pickle.load(f)

        with open(dataset_dir + '/apconet_packages/sample_aswh.np', 'rb') as f:
            sample_aswh = pickle.load(f)

        with open(dataset_dir + '/apconet_packages/sample_svs.np', 'rb') as f:
            sample_svs = pickle.load(f)

        with open(dataset_dir + '/apconet_packages/sample_chart.np', 'rb') as f:
            sample_chart = pickle.load(f)

        return sample_wave, sample_svs, sample_aswh, sample_chart


def get_single_dataset_index_chart_based(unique_chart, data_c, valid_sampling_rate, regex_rule=None):
    """
    """

    random_valid = np.random.choice([True, False], size=len(unique_chart), p=[valid_sampling_rate, 1-valid_sampling_rate])
    train_chart = unique_chart[np.invert(random_valid)]
    valid_chart = unique_chart[random_valid]

    sel_idx_train = np.array([c in train_chart for c in data_c])
    sel_idx_valid = np.array([c in valid_chart for c in data_c])

    if regex_rule is not None:
        r1 = re.compile(regex_rule)
        vmatch1 = np.vectorize(lambda x: bool(r1.match(x)))
        regex_matched_index = np.array(vmatch1(data_c))
    else:
        regex_matched_index = np.array([True] * len(data_c))

    train_index = (sel_idx_train & regex_matched_index)
    valid_index = (sel_idx_valid & regex_matched_index)

    return train_index, valid_index



def train(Model, train_generator, valid_generator, current_kfold_step, epochs,
          batch_size, learning_rate, patience, criteria, optimizer, weight_decay,
          sgd_momentum, valid_interval, root_directory='./', return_directory=False):

    odir_folder = 'inception_temp'

    odir = root_directory + '/' + odir_folder

    if not os.path.exists(odir):
        os.mkdir(odir)

    filewrite = open(odir + '/' + 'log_of_model_training.txt', 'w+')
    training_log_df = pd.DataFrame(columns=['stage', 'epoch', 'lr', 'step', 'train_loss', 'val_loss'])

    start_time_kf = time.time()


    model = Model
    model = model.cuda()
    model = nn.DataParallel(model)

    pretrain_cache = []
    best_loss_pretrain = np.Inf
    train_loss = np.Inf
    training_log_df = training_log_df


    # pre-train with scheduler
    print('Start pretrain with learning rate scheduler (StepLR)')
    model, trained_cache, best_loss, train_loss, training_log_df = trainer.train_with_scheduler(
        Model_architecture=Model,
        train_data_generator=train_generator,
        validatioin_data_generator=valid_generator,
        epochs=epochs,
        learning_rate=learning_rate,
        patient=patience,
        criteria=criteria,
        optimizer_type=optimizer,
        weight_decay_rate=weight_decay,
        sgd_momentum=sgd_momentum,
        filewrite=filewrite, odir=odir,
        log_df=training_log_df,
        valid_interval=valid_interval,
        batch_size=batch_size)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             obj_name = varname(obj)
    #             print(type(obj), obj_name, obj.size(), obj.get_device())
    #     except:
    #         pass

    print(' ')
    print('-----------Train step done-------------------')
    print(' ')

    print(' ', file=filewrite)
    print('-----------Train step done-------------------', file=filewrite)
    print(' ', file=filewrite)


    newdir = "{}/val-{:.3f}_train-{:.3f}".format(os.path.dirname(odir), float(best_loss), float(train_loss))

    elapsed_time_kf = time.time() - start_time_kf
    print('End KFold number {} / at {}s'.format(current_kfold_step, elapsed_time_kf))
    print('Best validation loss: {}.'.format(best_loss))
    print(' ')
    print('===================================================================================')
    print(' ')

    print('End KFold number {} / at {}s'.format(current_kfold_step, elapsed_time_kf), file=filewrite)
    print('Best validation loss: {}.'.format(best_loss), file=filewrite)
    print(' ')
    print('===================================================================================')
    print(' ')

    if return_directory:
        return best_loss, trained_cache
    else:
        return best_loss



def run_train(config):
    # define parameters for each dataset
    train_params = {
        # run on multiple gpus
        'batch_size': config.batch_size,
        'shuffle': True,
        'num_workers': torch.cuda.device_count()
    }

    train_params_non_suffle = {
        # run on multiple gpus
        'batch_size': config.batch_size,
        'shuffle': False,
        'num_workers': torch.cuda.device_count()
    }

    valid_params = {
        # run on multiple gpus
        'batch_size': config.batch_size,
        'shuffle': False,
        'num_workers': torch.cuda.device_count()
    }

    #############
    # load data #
    #############
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Start Loading data")

    # load pretrain data

    all_w, all_svs, all_a, all_c = load_datasets(config.dataset_dir,
                                                 config.version,
                                                 custom_set=False)

    print("Data Loading done")

    print('Length of Data: [train, synced, vig-only]')
    print('w: [{}]'.format(len(all_w)))
    print('svs: [{}]'.format(len(all_svs)))
    print('a: [{}]'.format(len(all_a)))
    print('c: [{}]'.format(len(all_c)))

    all_X = all_w / 100
    # train_all_y = (train_all_svs - 0.75) / (30-0.75)
    all_y = all_svs

    del all_w


    ##############
    # Build dataset without KFold
    ##############
    train_unique_chart_name = np.unique(all_c)

    train_index, valid_index = get_single_dataset_index_chart_based(train_unique_chart_name,
                                                                  data_c=all_c,
                                                                  valid_sampling_rate=config.valid_sampling_rate)

    #############################
    # build dataset and generator
    #############################

    val_losses = []
    final_mapes = []
    final_loas = []
    fold_step = 0

    train_set = Datasets(aline_wave=all_X, ashw=all_a, ylabels=all_y, chart=all_c,
                         train_valid_flag=train_index)
    train_generator = data.DataLoader(train_set, **train_params)

    valid_set = Datasets(aline_wave=all_X, ashw=all_a, ylabels=all_y, chart=all_c,
                         train_valid_flag=valid_index)
    valid_generator = data.DataLoader(valid_set, **valid_params)


    ####################
    # define environment
    ####################

    base_dir = config.base_dir

    rootdir = '{}/Train'.format(base_dir)

    # e.g., 60 sec, 100hz, inception=... 등의 폴더 명을 찾아서 filename에 반환함
    # 해당 폴더명을 predir로 저장해두는 함수
    print("Start root directory making and search")
    # 만약 root directory가 없으면 만듬
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)

    # record used chartname for TRAIN chart name
    train_charts_in_used, train_counts_in_used = np.unique(all_c[train_index], return_counts=True)
    with open(rootdir + '/train_charts_in_used.csv', 'wt') as f:
        f.write('chart_names, counts\n')
        for o1, o2 in zip(train_charts_in_used, train_counts_in_used):
            f.write("{}, {}\n".format(o1, o2))

    # record used chartname for validation chart name and test chart name
    valid_charts_in_used, valid_counts_in_used = np.unique(all_c[valid_index], return_counts=True)
    with open(rootdir + '/validation_charts_in_used.csv', 'wt') as f:
        f.write('chart_names, counts\n')
        for o1, o2 in zip(valid_charts_in_used, valid_counts_in_used):
            f.write("{}, {}\n".format(o1, o2))

    model0 = Inception1DNet_NL_compact_dilated(nlayer=15, nfilter=32)


    val_loss = train(Model=model0, train_generator=train_generator,
                     valid_generator=valid_generator,
                     current_kfold_step=fold_step,
                     epochs=config.epoch,
                     batch_size=config.batch_size,
                     learning_rate=config.lr,
                     patience=config.earlystopping_patience,
                     criteria=config.criteria,
                     optimizer=config.optimizer_type,
                     weight_decay=config.weight_decay,
                     sgd_momentum=config.sgd_momentum,
                     root_directory=rootdir,
                     valid_interval=config.validation_interval)

    val_losses.append(val_loss)

    gc.collect()

    print('Training is finished.')
    print('Model mean loss = {}'.format(np.mean(np.array(val_losses))))
    print("Execution finished and saved at: {}".format(rootdir))

    return rootdir