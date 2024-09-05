import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


class ConvQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight_r = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias_g, 1)
            nn.init.constant_(self.bias_b, 0)

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias == True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:

        nn.init.normal_(self.weight_r, mean=0,
                        std=np.sqrt(0.25 / (self.weight_r.shape[1] * np.prod(self.weight_r.shape[2:]))) * 8)
        self.__reset_bias()

    def forward(self, x):
        if self.bias == False:
            out = F.conv1d(x, self.weight_r, None, self.stride, self.padding, 1, 1) \
                  * F.conv1d(x, self.weight_g, None, self.stride, self.padding, 1, 1) \
                  + F.conv1d(torch.pow(x, 2), self.weight_b, None, self.stride, self.padding, 1, 1)
        else:
            out = F.conv1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1)\
            * F.conv1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1) \
            + F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)

        return out


class ConvQuadraticOperation2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvQuadraticOperation2D, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight_r = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias_g, 1)
            nn.init.constant_(self.bias_b, 0)

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias == True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:

        nn.init.normal_(self.weight_r, mean=0,
                        std=np.sqrt(0.25 / (self.weight_r.shape[1] * np.prod(self.weight_r.shape[2:]))) * 8)
        self.__reset_bias()

    def forward(self, x):
        if self.bias == False:
            out = F.conv2d(x, self.weight_r, None, self.stride, self.padding, 1, 1) \
                  * F.conv2d(x, self.weight_g, None, self.stride, self.padding, 1, 1) \
                  + F.conv2d(torch.pow(x, 2), self.weight_b, None, self.stride, self.padding, 1, 1)
        else:
            out = F.conv2d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1)\
            * F.conv2d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1) \
            + F.conv2d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)

        return out



if __name__ == '__main__':
    a = torch.randn(20, 1, 2048)
    b = ConvQuadraticOperation(1, 16, 64, 8, 28)
    c = b(a)
    print(c.shape)
    e = d(c)
    print(e.shape)
