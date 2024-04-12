import math

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F
from Model.ConvQuadraticOperation import ConvQuadraticOperation, InverseConvQuadraticOperation
from utils.train_function import group_parameters
from Model.weight_init import Laplace_fast
def hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.

        Parameters:
        -------------
        x (Tensor) : The signal data.
                     The Hilbert transform is performed along last dimension of `x`.

        Returns:
        -------------
        analytic (Tensor): A complex tensor with the same shape of `x`,
                           representing its analytic signal.

    '''

    N = x.shape[-1]
    # Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        x[..., 1: N // 2] *= 2
        x[..., N // 2 + 1:] = 0
    else:
        x[..., 1: (N + 1) // 2] *= 2
        x[..., (N + 1) // 2:] = 0
    return torch.fft.ifft(x)


def hilbert_transform(signal):
    N = signal.shape[-1]
    spectrum = torch.fft.fft(signal)
    h_spectrum = torch.zeros_like(spectrum)
    h_spectrum[..., :N // 2] = 2 * spectrum[..., :N // 2]  # double the positive frequency part
    h_spectrum[..., N // 2 + 1:] = -2 * spectrum[..., N // 2 + 1:]  # double the negative frequency part
    if N % 2 == 0:
        h_spectrum[..., N // 2] = spectrum[..., N // 2]  # keep the Nyquist frequency component as is
    analytic_signal = torch.fft.ifft(h_spectrum)

    # 计算包络谱
    amplitude_envelope = torch.abs(analytic_signal)

    # 计算相位
    instantaneous_phase = torch.angle(analytic_signal)

    # 计算频率
    instantaneous_frequency = torch.diff(instantaneous_phase, dim=-1)

    return amplitude_envelope, instantaneous_frequency

def get_envelope_frequency(x, fs, ret_analytic=False, **kwargs):
    '''
        Compute the envelope and instantaneous freqency function of the given signal, using Hilbert transform.
        The transformation is done along the last axis.

        Parameters:
        -------------
        x (Tensor) :
            Signal data. The last dimension of `x` is considered as the temporal dimension.
        fs (real) :
            Sampling frequencies in Hz.
        ret_analytic (bool, optional) :
            Whether to return the analytic signal.
            ( Default: False )

        Returns:
        -------------
        (envelope, freq)             when `ret_analytic` is False
        (envelope, freq, analytic)   when `ret_analytic` is True

            envelope (Tensor) :
                       The envelope function, with its shape same as `x`.
            freq     (Tensor) :
                       The instantaneous freqency function measured in Hz, with its shape
                       same as `x`.
            analytic (Tensor) :
                       The analytic (complex) signal, with its shape same as `x`.
    '''


    analytic = hilbert(x)
    envelope = analytic.abs()
    en_raw_abs = envelope - torch.mean(envelope)
    es_raw = torch.fft.fft(en_raw_abs)
    es_raw_abs = torch.abs(es_raw) * 2 / len(en_raw_abs)
    sub = torch.cat((analytic[..., 1:] - analytic[..., :-1],
                     (analytic[..., -1] - analytic[..., -2]).unsqueeze(-1)
                     ), axis=-1)
    add = torch.cat((analytic[..., 1:] + analytic[..., :-1],
                     2 * analytic[..., -1].unsqueeze(-1)
                     ), axis=-1)
    freq = 2 * fs * ((sub / add).imag)
    freq[freq.isinf()] = 0
    del sub, add
    freq /= (2 * math.pi)
    return (es_raw_abs, freq) if not ret_analytic else (envelope, freq, analytic)



class CLASSBD(nn.Module):
    def __init__(self, l_input=2048, fs=64000) -> object:
        super(CLASSBD, self).__init__()
        self.fs = fs
        self.qtfilter = nn.Sequential(
            nn.AvgPool1d(1, 1),
            ConvQuadraticOperation(1, 16, 63, 1, 'same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            ConvQuadraticOperation(16, 1, 63, 1, 'same'),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(1, 1),
            nn.Sigmoid()
        )
        self.filter1 = nn.Linear(l_input, l_input)

        # self.W = nn.Parameter(torch.randn(2, 1))

    def funcKurtosis(self, y, halfFilterlength=32):
        y_1 = torch.squeeze(y)
        y_1 = y_1[halfFilterlength:-halfFilterlength]
        y_2 = y_1 - torch.mean(y_1)
        num = len(y_2)
        y_num = torch.sum(torch.pow(y_2, 4), dim=-1) / num
        std = torch.sqrt(torch.sum(torch.pow(y_2, 2), dim=-1) / num)
        y_dem = torch.pow(std, 4)
        loss = y_num / y_dem
        return nn.Parameter(loss.mean())


    def g_lplq(self, y, p=2, q=4):
        es_raw_abs, _ = get_envelope_frequency(y, fs=self.fs)
        # max_product_values = self.process_es(es_raw_abs[:,:,1: es_raw_abs.shape[-1] // 2])
        p = torch.tensor(p)
        q = torch.tensor(q)
        obj = torch.sign(torch.log(q / p)) * (torch.norm(es_raw_abs, p, dim=-1) / torch.norm(es_raw_abs, q, dim=-1)) ** p
        return nn.Parameter(obj.mean())

    def forward(self, x):
        # time filter
        a1 = self.qtfilter(x)
        k = self.funcKurtosis(a1)

        # frequency filter
        f1 = torch.fft.fft2(a1 - torch.mean(a1, dim=-1).unsqueeze(1))
        f1 = abs(f1)
        es = self.filter1(f1)
        es_raw_abs, _ = get_envelope_frequency(es, self.fs)
        a2 = abs(torch.fft.ifft2(es))
        g = self.g_lplq(es_raw_abs)



        # l = - self.W[0] * k + self.W[1] * g


        return a1, a2, -k, g