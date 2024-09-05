import torch
import torch.nn as nn



class UW():
    def __init__(self, task_num=3):
        super(UW, self).__init__()
        self.task_num = task_num
        self.init_param()

    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5])).cuda()

    def forward(self, losses):
        loss = (losses / (self.task_num * self.loss_scale.exp()) + self.loss_scale / self.task_num).sum()
        loss.backward()
        return loss