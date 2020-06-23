import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import dlapco_packages.load_model as load
import arr
import numpy as np
import torch

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


def run(inp, opt, cfg):
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


if __name__ == '__main__':
    import pickle
    import time

    getwd = os.path.dirname(os.path.realpath(__file__))

    with open(getwd+'/apconet_packages/sample_wave.np', 'rb') as f:
        sample_wave = pickle.load(f)

    with open(getwd+'/apconet_packages/sample_aswh.np', 'rb') as f:
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
    output = run(sample_data_inp, sample_data_opt, cfg)
    end_time = time.time()

    print(np.squeeze(output))



