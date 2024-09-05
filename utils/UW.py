import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class UW():

    def __init__(self, task_num=3):
        super(UW, self).__init__()
        self.task_num = task_num
        self.init_param()
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * self.task_num)).cuda()

    def run(self, losses):
        loss = (losses / (2 * self.loss_scale.exp()) + self.loss_scale / 2).sum()
        loss.backward()
        return