import os
import random

import numpy as np
import torch
from sklearn.metrics import  confusion_matrix, recall_score, \
    precision_score, f1_score
from utils.JNUProcessing import JNU_Processing
from utils.PUProcessing import Paderborn_Processing
import wandb
from torch import nn
from torch.utils.data import DataLoader
from Model.BDCNN import BDWDCNN
from Model.BDMobileNet import MobileNetV3_Small
from Model.BDResNet import resnet18
from Model.BDTransformer import DSCTransformer
from utils.DatasetLoader import CustomTensorDataset





use_gpu = torch.cuda.is_available()

def UW(losses):
    loss_scale = nn.Parameter(torch.tensor([-0.5] * 3)).cuda()
    loss = (losses / (3 * loss_scale.exp()) + loss_scale / 3).sum()
    return loss


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def select_model(config):
    if config["chosen_model"] == 'bdresnet':
        model = resnet18(config['class_num'])
    if config["chosen_model"] == 'bdmobile':
        model = MobileNetV3_Small(config['class_num'])
    if config["chosen_model"] == 'bdtransformer':
        model = DSCTransformer(num_classes=config['class_num'])
    if config["chosen_model"] == 'bdcnn':
        model = BDWDCNN(config['class_num'])
    return model



def train(config, dataloader):
    net = select_model(config)
    if use_gpu:
        net.cuda()
    train_loss = []
    train_acc = []
    valid_acc = []
    max_acc = 0

    for e in range(config['epochs']):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0
            if phase == 'train':
                net.train()
            if phase == 'validation':
                net.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()
                optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],eta_min=1e-8)

                loss_func = nn.CrossEntropyLoss()
                y_hat, k, g = net(x)
                classifyloss = loss_func(y_hat, y)
                losses = torch.zeros(3).cuda()
                losses[0], losses[1], losses[2] = classifyloss, k, g
                loss = UW(losses)
                if phase == 'train':
                    optimizer.zero_grad()
                    # loss = uw.forward(losses)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                loss_total += loss.item()

                y_predict = y_hat.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    correct += (y_predict == y).cpu().squeeze().sum().numpy()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()

                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                          e + 1, step + 1, len(dataloader[phase].dataset), loss_total))

            acc = correct / total
            if phase == 'train':
                train_loss.append(loss_total)
                train_acc.append(acc)

            if phase == 'validation':
                valid_acc.append(acc)
                if acc > max_acc:
                    max_acc = acc
                    if not os.path.exists("Models"):
                        os.mkdir('Models')
                    # 存储模型参数
                    torch.save(net.state_dict(), f'Models/{config["dataset"]}_best_checkpoint_{config["chosen_model"]}.pth')
                    print("save model")
            print('%s ACC:%.4f' % (phase, acc))
    return net


def inference(dataloader, config):
    net = select_model(config)
    state_dict = torch.load(f'Models/{config["dataset"]}_best_checkpoint_{config["chosen_model"]}.pth')
    net.load_state_dict(state_dict)
    y_list, y_predict_list = [], []
    if use_gpu:
        net.cuda()
    net.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat, _, _ = net(x)

            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())

        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")

        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        print({
            "F1 Score": F1,
            "FPR": FPR,
            "Recall": recall,
            'PRE': precision})
        return F1


def main(config):
    random_seed(config['seed'])

    if config['dataset'] == "Paderborn":
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = Paderborn_Processing(file_path=os.path.join('../ClassBD/data', config['dataset']), load=config['chosen_dataset'], noise=config['add_noise'], snr=config['snr'])
        config['class_num'] = 14

    elif config['dataset'] == 'JNU':
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = JNU_Processing(file_path=os.path.join('../ClassBD/data', config['dataset']),noise=config['add_noise'], snr=config['snr'])
        config['class_num'] = 10


    train_dataset = CustomTensorDataset(Train_X, Train_Y)
    valid_dataset = CustomTensorDataset(Val_X, Val_Y)
    test_dataset = CustomTensorDataset(Test_X, Test_Y)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
    train(config, data_loaders)
    inference(test_loader, config)



if __name__ == '__main__':
    config = {'seed': 42,
              'batch_size': 128,
              'epochs': 200,
              'lr': 0.5,
              'add_noise': 'Gaussian', # Gaussian, pink, Laplace, airplane, truck
              'snr': -4, #dB
              'dataset': 'Paderborn', # Paderborn, JNU
              'chosen_dataset': 'N15_M07_F04', # N09_M07_F10; N15_M01_F10; N15_M07_F04; N15_M07_F10;
              'chosen_model': 'bdcnn',   # bdcnn, bdresnet, bdtransformer, bdmobile
              'class_num': 14  # default
              }
    main(config)
