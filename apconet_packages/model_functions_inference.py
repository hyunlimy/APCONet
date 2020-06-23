import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import glob
import os
import copy

#############################################
# Utilities
#############################################


def weights_init_conv(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform(m.bias.data)
        m.bias.data.fill_(0.01)


def weights_init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform(m.bias.data)
        m.bias.data.fill_(0.01)


def data_split(np_cases, ntest):
    """
    input: cases, # of cases to select
    return: (remained index, test index) of provided data
    """
    all_cases = np.unique(np_cases)
    np.random.shuffle(all_cases)
    test_cases = all_cases[:ntest]
    remained_cases = all_cases[ntest:]
    nremain = len(remained_cases)
    remained_idx = [c not in test_cases for c in all_cases]
    test_idx = [t in test_cases for t in all_cases]

    return (remained_idx, test_idx)


def data_split_FT(np_cases, dev_name):
    """
    input: cases, device name to split
    return: (remained index, fine-tune index) of provided data
    """

    ft_idx = [d == dev_name for d in np_cases]
    remained_idx = [not i for i in ft_idx]

    return (remained_idx, ft_idx)


def build_x(all_X, all_a, idx):
    cnn = all_X[idx]
    fnn = all_a[idx]
    cnn = np.reshape(cnn, (cnn.shape[0], cnn.shape[1], 1))
    return [fnn, cnn]


def build_x_none(all_X, all_a):
    cnn = all_X
    fnn = all_a
    cnn = np.reshape(cnn, (cnn.shape[0], cnn.shape[1], 1))
    return [fnn, cnn]


def extract_X(x, idx):
    x_fnn = x[0][idx]
    x_cnn = x[1][idx]
    return [x_fnn, x_cnn]


def vstack_X(list1, list2):
    x_fnn = np.vstack([list1[0], list2[0]])
    x_cnn = np.vstack([list1[1], list2[1]])
    return [x_fnn, x_cnn]


def mape_loss(output, target, loss_type ='mape'):
    # numpy to torch tensor
    if type(output) is not torch.Tensor:
        output = torch.from_numpy(output)
        target = torch.from_numpy(target)

    # use smape for handle no upper limit problem
    if loss_type == 'mape':
        # mape
        loss = torch.mean(torch.abs((target - output) / target)) * 100
    elif loss_type == 'smape':
        # sampe
        loss = 100 / len(target) * torch.sum(2 * torch.abs(target - output) / (torch.abs(output) + torch.abs(target)))
    else:
        # default is mape
        loss = torch.mean(torch.abs((target - output) / target)) * 100
    return loss


def smape_loss(output, target):
    if type(output) is not torch.Tensor:
        output = torch.from_numpy(output)
        target = torch.from_numpy(target)
    return 100 / len(target) * torch.sum(2 * torch.abs(target - output) / (torch.abs(output) + torch.abs(target)))


def RMSELoss(output, target):
    return torch.sqrt(torch.mean((output-target)**2))


def remove_and_replace_outlier(signal, m_val = 4, ci_cut=False, verbose=False):

    def outliers_index(data, m=m_val):
        if ci_cut:
            return [abs(data - np.mean(data)) > m * np.std(data)][0]
        else:
            return np.array((data > 250) | (data < 20))

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if type(signal) is torch.Tensor:
        try:
            signal = signal.numpy()
        except TypeError:
            signal = signal.cpu().numpy()

    if signal.ndim > 1:
        signal = np.squeeze(signal)

    if verbose:
        print('Signal')
        print(type(signal))
        print(signal.shape)
        print(signal)

    outlier_index = outliers_index(signal)
    if verbose:
        print('outlier_index')
        print(outlier_index)

    if (~outlier_index).all():
        print("No outlier detected with m = {}".format(m_val))
        if verbose:
            print('Signal min: {}'.format(np.min(signal)))
            print('Signal max: {}'.format(np.max(signal)))
        return np.expand_dims(signal, axis=1)
    else:
        print("{} of outliers detected from {} points.".format(outlier_index.sum(), len(outlier_index)))
        outlier_points = list(np.where(outlier_index==True))[0]
        if verbose:
            print('outlier_points')
            print(outlier_points)

        for pnt in outlier_points:

            i = copy.deepcopy(pnt)
            if verbose:
                print('Current pnt value: {}'.format(pnt))
                print('Starting i: {}'.format(i))
                print('while loop test: {}'.format(outlier_index[i]))

            while outlier_index[i] == True:
                i -= 1
                if verbose:
                    print('Updated i: {}'.format(i))
            pnt_left_idx = i
            if verbose:
                print('left_idx_found : {}'.format(pnt_left_idx))

            j = copy.deepcopy(pnt)
            if verbose:
                print('Current pnt value: {}'.format(pnt))
                print('Starting j: {}'.format(j))
                print('while loop test: {}'.format(outlier_index[j]))

            while outlier_index[j] == True:
                j += 1
                if verbose:
                    print('Updated j: {}'.format(j))
            pnt_right_idx = j
            if verbose:
                print('right_idx_found : {}'.format(pnt_right_idx))

            interpolated = signal[pnt_left_idx] + signal[pnt_right_idx] / 2
            if verbose:
                print('interpolated to: {} --> {}'.format(signal[pnt], interpolated))
            signal[pnt] = interpolated

        return np.expand_dims(signal, axis=1)


def find_inception_layers_pt(_inception):
    inds = []
    for i in range(len(_inception.Inception)):
        if isinstance(_inception.Inception[i], InceptionModule_dilated):
            inds.append(i)
    return inds


def find_spatialnl_layers_pt(_inception):
    inds = []
    for i in range(len(_inception.Inception)):
        if isinstance(_inception.Inception[i], SpatialNL):
            inds.append(i)
    return inds

###############################################################################
# Inception Net Modules
###############################################################################


class InceptionModule(nn.Module):

    def __init__(self, in_channel, nfilter=32):
        super(InceptionModule, self).__init__()
        # implement same padding with (kernel_size//2) for pytorch
        # Ref: https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606

        # 1x1 conv path
        self.path0 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1X1 conv -> 3x3 conv path
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv path
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=5, padding=(5 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv path
        self.path3 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )


    def forward(self, x):
        print('x shape: {}'.format(x.shape))
        y0 = self.path0(x)
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)
        print('y0 shape: {}'.format(y0.shape))
        print('y1 shape: {}'.format(y1.shape))
        print('y2 shape: {}'.format(y2.shape))
        print('y3 shape: {}'.format(y3.shape))
        print('cat shape: {}'.format(torch.cat([y0, y1, y2, y3], 1).shape))

        return torch.cat([y0, y1, y2, y3], 1)


class InceptionModule_dilated(nn.Module):

    def __init__(self, in_channel, nfilter=32):
        super(InceptionModule_dilated, self).__init__()
        # implement same padding with (kernel_size//2) for pytorch
        # Ref: https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606

        # 1x1 conv path
        self.path0 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1X1 conv -> 3x3 conv path
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=3, padding=int(((3-1)*5)/2),
                      dilation=5),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv path
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=5, padding=int(((5-1)*7)/2),
                      dilation=7),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv path
        self.path3 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # Dilation output size calculation
        # o = output
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation
        # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1

        # padding = ((s-1)*i + (k-1)*d)/2

    def forward(self, x):
        y0 = self.path0(x)
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)

        # print(y0.shape)
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        return torch.cat([y0, y1, y2, y3], 1)


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
       revised to 1d
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale

        super(SpatialNL, self).__init__()
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, d = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, d)

        x = self.z(x)
        x = self.bn(x) + residual

        return x


###################################
# Inception Net Architecture
###################################


class Inception1DNet(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(True)
        )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        while (i < nlayer):
            dynamicInception[str(j)] = InceptionModule(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_aswh = self.fnn(x[0])

        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)
        concat = torch.cat([out_aswh, out_cnn], 1)

        out = self.regressor(concat)

        return out


class Inception1DNet_NL_compact_dilated(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet_NL_compact_dilated, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(True)
        )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter * 4),
            nn.Dropout(0.5)
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        k = 0
        while (i < nlayer):
            if i > (nlayer - 3):
                dynamicInception[str(j)] = SpatialNL(nfilter * 4, nfilter * 4)
                j += 1
            dynamicInception[str(j)] = InceptionModule_dilated(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear((nfilter * 4) + 4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_aswh = self.fnn(x[0])

        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)
        concat = torch.cat([out_aswh, out_cnn], 1)

        out = self.regressor(concat)

        return out


class Inception1DNet_NL_compact_dilated_no_ashw(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet_NL_compact_dilated_no_ashw, self).__init__()

        # remove ashw
        # self.fnn = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.ReLU(True)
        # )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter * 4),
            nn.Dropout(0.5)
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        k = 0
        while (i < nlayer):
            if i > (nlayer - 3):
                dynamicInception[str(j)] = SpatialNL(nfilter * 4, nfilter * 4)
                j += 1
            dynamicInception[str(j)] = InceptionModule_dilated(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            # for noashw
            nn.Linear(nfilter*4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)

        out = self.regressor(out_cnn)

        return out

