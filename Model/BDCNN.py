import math
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F

from Model.ClassBD_Module import CLASSBD
from Model.ConvQuadraticOperation import ConvQuadraticOperation




class BDWDCNN(nn.Module):
    """
    Quadraic in Attention CNN
    """

    def __init__(self, n_classes) -> object:
        super(BDWDCNN, self).__init__()

        self.classbd = CLASSBD()

        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1D_1', nn.Conv1d(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))




        self.__make_layer(16, 32, 1, 2)
        self.__make_layer(32, 64, 1, 3)   #  改64
        self.__make_layer(64, 64, 1, 4)   #  改64
        self.__make_layer(64, 64, 1, 5)   #  改64
        self.__make_layer(64, 64, 0, 6)
        self.fc1 = nn.Linear(192, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, n_classes)

    def __make_layer(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def funcD_norm(self, y, halfFilterlength=32):
        y_1 = torch.squeeze(y)
        y_1 = y_1[halfFilterlength:-halfFilterlength]
        y_2 = y_1 - torch.mean(y_1)
        y_abs = torch.abs(y_2)
        y_max = torch.max(y_abs)
        y_D_norm = torch.norm(y_abs, 2)
        D_norm = y_max / y_D_norm

        return D_norm



    def forward(self, x):

        a2, k, g = self.classbd(x)
        # backbone
        out = self.cnn(a2)
        out = self.fc1(out.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1), k, g

        # return k, g


if __name__ == '__main__':
    X = torch.rand(1, 1, 2048)
    m = BDWDCNN()
    s = time.time()
    m(X)
    e = time.time()
    print(e-s)

    print(flop_count_str(FlopCountAnalysis(m, X)))