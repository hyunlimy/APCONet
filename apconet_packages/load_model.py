# -*- coding: utf-8 -*-


import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import json
import numpy as np
import torch
from collections import OrderedDict

from .model_functions_inference import *


class SetupConfiguration(object):
    """input: args (Namespace)"""
    def __init__(self):
        self.model_filters = int(32)
        self.model_layers = int(15)
        self.model_type = 'inception-nl-compact-dilated'.lower()

    def view_model_type(self):
        return (self.model_layers, self.model_filters)

    def return_model(self):
        if self.model_type == 'inception-nl-compact-dilated':
            self.model = Inception1DNet_NL_compact_dilated(nlayer=self.model_layers, nfilter=self.model_filters)
            return self.model
        else:
            raise ValueError('Not supporting model type')


def load_model(Model_cache):

    config = SetupConfiguration()

    ft_cache = Model_cache
    Model = config.return_model()

    print('Start best model loading')

    # convert data parallel model to CPU version
    test_model = Model
    # if torch.cuda.is_available():
    #     loaded_model = torch.load(ft_cache)
    # else:
    #     loaded_model = torch.load(ft_cache, map_location='cpu')
    loaded_model = torch.load(ft_cache, map_location='cpu')
    try:
        test_model.load_state_dict(loaded_model.state_dict())
        print('no dataparallel module detected')
    except RuntimeError:
        print('runtime error occured: remove dataparallel module')
        loaded_state_dict = loaded_model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in loaded_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        test_model.load_state_dict(new_state_dict)

    print('Loaded model for test: {}'.format(ft_cache))

    # if torch.cuda.is_available():
    #     test_model = test_model.cuda()
    #     print('model loaded on GPU')
    #     onGPU = True
    # else:
    #     test_model = test_model.cpu()
    #     print('model loaded on CPU')
    #     onGPU = False

    test_model = test_model.cpu()

    test_model.eval()

    return test_model
