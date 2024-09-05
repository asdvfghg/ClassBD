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
import torch.nn.functional as F





use_gpu = torch.cuda.is_available()
print('GPU: %s' %(use_gpu))

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
    if config.chosen_model == 'bdresnet':
        model = resnet18(config.class_num)
    if config.chosen_model == 'bdmobile':
        model = MobileNetV3_Small(config.class_num)
    if config.chosen_model == 'bdtransformer':
        model = DSCTransformer(num_classes=config.class_num)
    if config.chosen_model == 'bdcnn':
        model = BDWDCNN(config.class_num)
    return model

def funcKurtosis(y, halfFilterlength=32):
    y_1 = torch.squeeze(y)
    y_1 = y_1[halfFilterlength:-halfFilterlength]
    y_2 = y_1 - torch.mean(y_1)
    num = len(y_2)
    y_num = torch.sum(torch.pow(y_2, 4)) / num
    std = torch.sqrt(torch.sum(torch.pow(y_2, 2)) / num)
    y_dem = torch.pow(std, 4)
    loss = y_num / y_dem
    return loss

def loss_fn(x, y, target_y):
    loss_x = -funcKurtosis(x)
    loss_y = F.cross_entropy(y, target_y)
    return loss_x, loss_y


def train(config, dataloader):
    net = select_model(config)
    if use_gpu:
        net.cuda()
    wandb.watch(net, log="all")

    train_loss = []
    train_acc = []
    valid_acc = []
    max_acc = 0

    for e in range(config.epochs):
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
                optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,eta_min=1e-8)

                loss_func = nn.CrossEntropyLoss()
                y_hat, k, g = net(x)
                classifyloss = loss_func(y_hat, y)
                losses = torch.zeros(3).cuda()
                losses[0], losses[1], losses[2]= classifyloss, k, g
                loss = UW(losses)

                if phase == 'train':
                    optimizer.zero_grad()
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
                wandb.log({
                    "Train Accuracy": 100. * acc,
                    "Train Loss": loss_total})
            if phase == 'validation':
                valid_acc.append(acc)
                wandb.log({
                    "Validation Accuracy": 100. * acc})
                if acc > max_acc:
                    max_acc = acc
                    if not os.path.exists("Models"):
                        os.mkdir('Models')
                    # 存储模型参数
                    torch.save(net.state_dict(), f'Models/{config.path}_best_checkpoint_{config.chosen_model}.pth')
                    print("save model")
            print('%s ACC:%.4f' % (phase, acc))
    return net


def inference(dataloader, chosen_model):
    net = select_model(chosen_model)
    state_dict = torch.load(f'Models/{config.path}_best_checkpoint_{chosen_model}.pth')
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
        wandb.log({
            "F1 Score": F1,
            "FPR": FPR,
            "Recall": recall,
            'PRE': precision})
        return F1


def main(config):
    random_seed(config.seed)

    if config.path == "Paderborn":
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = Paderborn_Processing(file_path=os.path.join('../ClassBD/data', config.path), load=config.chosen_dataset, noise=config.add_noise, snr=config.snr)
        config.class_num = 14

    elif config.path == 'JNU':
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = JNU_Processing(file_path=os.path.join('data', config.path),noise=config.add_noise, snr=config.snr)
        config.class_num = 10


    train_dataset = CustomTensorDataset(Train_X, Train_Y)
    valid_dataset = CustomTensorDataset(Val_X, Val_Y)
    test_dataset = CustomTensorDataset(Test_X, Test_Y)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
    train(config, data_loaders)
    inference(test_loader, config)



if __name__ == '__main__':
    # wandb initialization, you need to create a wandb account and enter the username in 'entity'
    wandb.init(project="ClassBD", entity="jing-xiaoliao")

    # WandB – Config is a variable that holds and saves hypermarkets and inputs
    config = wandb.config  # Initialize config
    config.log_interval = 200  # how many batches to wait before logging training status
    config.seed = 42  # random seed (default: 42)

    # Hyperparameters, lr and alpha need to fine-tune
    config.batch_size = 128  # input batch size for training (default: 64)
    config.epochs = 200  # number of epochs to train (default: 10)
    config.lr = 0.5  # learning rate (default: 0.5)


    # noisy condition6
    config.add_noise = 'Gaussian' # Gaussian, pink, Laplace, airplane, truck
    config.snr = -4 # dB

    # dataset and model
    config.path = 'Paderborn' # Paderborn JNU
    config.chosen_dataset = 'N09_M07_F10' # N09_M07_F10; N15_M01_F10; N15_M07_F04; N15_M07_F10;
    config.chosen_model = 'bdcnn'   # bdcnn, bdresnet, bdtransformer, bdmobile

    main(config)
