# -*- coding: utf-8 -*-


import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from .apconet_packages import load_model as load
from . import arr
from . import run_train
import numpy as np
import torch
import argparse
import pickle
import time

getwd = os.path.dirname(os.path.realpath(__file__))
model_binary = '{}/apconet_packages/model.pth'.format(getwd)
global_model = load.load_model(model_binary)

cfg = {
    'name': 'Stroke Volume',
    'group': 'Medical algorithms',
    'desc': 'Calculate stroke volume from arterial blood pressure using deep learning',
    'reference': 'DLAPCO',
    'overlap': 18,
    'interval': 20,
    'inputs': [{'name': 'art1', 'type': 'wav'}],
    'outputs': [
        {'name': 'sv', 'type': 'num', 'min': 0, 'max': 200, 'unit': 'mL'},
        ]
}


def build_x_inference(signal_np, wave_np):
    cnn = torch.Tensor(np.expand_dims(signal_np, axis=(0, 1)))
    fnn = torch.Tensor(np.expand_dims(wave_np, axis=0))
    return [fnn, cnn]


def run_test(inp, opt, cfg):
    """
    calculate SV from DeepLearningAPCO
    :param inp: input waves
    inp['art1']['vals'] must be 1-dimensional, (#,)
    :param opt: demographic information
    :param cfg:
    :return: SV
    """
    global global_model

    #signal_data = arr.interp_undefined(inp['art1']['vals'])
    signal_data = inp['art1']['vals']
    srate = inp['art1']['srate']

    signal_data = arr.resample_hz(signal_data, srate, 100)
    srate = 100

    signal_data = signal_data / 100.

    if len(signal_data) < 20 * srate:
        print('length < 20s')
        return

    # age_data = opt['age']
    # sex_data = opt['sex']
    # wt_data = opt['weight']
    # ht_data = opt['height']
    age_data = 60
    sex_data = 1.
    wt_data = 65.8
    ht_data = 164.9

    if isinstance(sex_data, str):
        if sex_data == 'M':
            sex_data = int(1)
        elif sex_data == 'F':
            sex_data = int(0)
        else:
            raise ValueError('opt_sex must be "M" or "F". current value: {}'.format(sex_data))

    else:
        if not (int(sex_data) == 1) or (int(sex_data) == 0):
            raise ValueError('opt_sex must be 1 or 0 current value: {}'.format(str(sex_data)))

    ashw_data = np.array([age_data, sex_data, wt_data, ht_data])

    # input of 'build_x_none' must be 3-dimensional, (batch_size, channel, length)
    # x_input = build_x_none(torch.Tensor(np.expand_dims(signal_data, axis=(0, 1))),
    #                        torch.Tensor(np.expand_dims(ashw_data, axis=(0, 1))))

    x_input = build_x_inference(signal_data, ashw_data)

    global_model.eval()

    output = global_model(x_input)
    print('done')

    return [
        [{'dt': cfg['interval'], 'val': output.detach().numpy()}]
    ]


def main_train(cfg):
    output_dir = run_train.run_train(cfg)
    print('Training done, Results saved at: {}'.format(output_dir))


def main_test():
    getwd = os.path.dirname(os.path.realpath(__file__))

    with open(getwd + '/apconet_packages/sample_wave.np', 'rb') as f:
        sample_wave = pickle.load(f)

    with open(getwd + '/apconet_packages/sample_aswh.np', 'rb') as f:
        sample_aswh = pickle.load(f)

    sample_data_inp = {}
    sample_data_inp['art1'] = {}
    sample_data_inp['art1']['vals'] = sample_wave
    sample_data_inp['art1']['srate'] = 100

    sample_data_opt = {}
    sample_data_opt['age'] = sample_aswh[0]
    sample_data_opt['sex'] = sample_aswh[1]
    sample_data_opt['weight'] = sample_aswh[2]
    sample_data_opt['height'] = sample_aswh[3]

    start_time = time.time()
    output = run_test(sample_data_inp, sample_data_opt, cfg)
    end_time = time.time()

    print(np.squeeze(output))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Train or Test mode'
    )

    arg_parser.add_argument(
        '-m',
        '--mode',
        dest='mode',
        type=str,
        default='train',
        help='train or test (inference) mode'
    )

    arg_parser.add_argument('--version', type=str, default='200101',
                            help='Dataset version with 6-length digits (yymmdd)')

    arg_parser.add_argument('--base_dir', type=str, default='./',
                            help='Base output directory where results will be saved.')

    arg_parser.add_argument('--dataset_dir', type=str, default='./dataset',
                            help='Dataset directory where np binaries exists')

    arg_parser.add_argument('--valid_sampling_rate', type=float, default=0.3,
                            help='validation rate (by chart)')

    arg_parser.add_argument('--batch_size', type=int, default=512,
                            help='batch size for training')

    arg_parser.add_argument('--epoch', type=int, default=500,
                            help='epoch for training')

    arg_parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate for training')

    arg_parser.add_argument('--earlystopping_patience', type=int, default=50,
                            help='earlystopping patience for training')

    arg_parser.add_argument('--optimizer_type', type=str, default='ranger',
                            help='optimizer for training')

    arg_parser.add_argument('--weight_decay', type=float, default=0.5,
                            help='weight decay for training')

    arg_parser.add_argument('--sgd_momentum', type=float, default=0.9,
                            help='monentum for training if SGD optimizer is selected')

    arg_parser.add_argument('--validation_interval', type=int, default=10,
                            help='validation interval for training')


    args = arg_parser.parse_args()

    if str(args.mode).lower() == 'train':
        main_train(args)
    elif (str(args.mode).lower() == 'test') or (str(args.mode).lower() == 'inference'):
        main_test()









