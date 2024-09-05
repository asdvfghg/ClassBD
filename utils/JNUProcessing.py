import sklearn.model_selection
from scipy.io import loadmat, savemat
import os
from sklearn import preprocessing
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from scipy.fft import fft
from tqdm import tqdm

from noise_observe import add_audio_noise
from utils.aug_function import *




def JNU_Processing(file_path,  length=2048, use_sliding_window=True, step_size=100,
                        sample_number=1000, normal=True, noise='Gaussian', snr = -6):


    # 获得训练集和测试集文件夹下所有.mat文件名
    train_filenames = os.listdir(file_path)
    # test_filenames = os.listdir(test_path)
    # 将文件名列表中结尾不是.mat的文件名去掉，防止下面loadmat报错


    def wgn(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(np.absolute(x) ** 2, axis=1) / x.shape[0]
        npower = xpower / snr
        npower = np.repeat(npower.reshape(-1, 1), x.shape[1], axis=1)
        return np.random.standard_normal(x.shape) * np.sqrt(npower)

    def add_noise(data, snr_num):
        rand_data = wgn(data, snr_num)
        n_data = data + rand_data
        return n_data, rand_data
    def capture(path, filenames):
        """
        读取paderborn数据集中的振动数据(vibration_1)
        :param path: 文件目录路径
        :param filenames: 文件名
        :return: 每个文件的振动数据(dict)
        """
        data = {}
        for i in tqdm(filenames):
            # 文件路径
            file_path = os.path.join(path, i)
            file = np.loadtxt(open(file_path, "rb"))
            data[i] = file
        return data

    def slice(data, samp_num):
        """
        切分文件的数据
        :param data: 数据
        :param data_number:  健康类的样本个数
        :return:
        """

        data_keys = data.keys()  # 获取各个文件名(用于统计每个类别的文件数目)

        Data_Samples = []  # 存储采样后的数据
        Test_DataSamples = []
        Labels = []  # 存储标签
        Test_Labels = []
        for key in tqdm(data_keys):
            slice_data = data[key]
            samp_num = len(slice_data) // step_size
            start = 0  # 采样信号的起点索引
            end_index = len(slice_data)
            if 'n' in key:
                class_num = 0
            elif 'ob600' in key:
                class_num = 1
            elif 'ob800' in key:
                class_num = 2
            elif 'ob1000' in key:
                class_num = 3
            elif 'ib600' in key:
                class_num = 4
            elif 'ib800' in key:
                class_num = 5
            elif 'ib1000' in key:
                class_num = 6
            elif 'tb600' in key:
                class_num = 7
            elif 'tb800' in key:
                class_num = 8
            elif 'tb1000' in key:
                class_num = 9

            for i in range(samp_num):
                if use_sliding_window:
                    if i > round(0.75 * samp_num):
                        test_sample = slice_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append(test_sample)
                            Test_Labels.append(class_num)

                    else:
                        sample = slice_data[start:start + length]
                        if len(sample) == length:
                            Data_Samples.append(sample)
                            Labels.append(class_num)

                else:
                    if i > round(0.75 * samp_num):
                        test_sample = slice_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append(test_sample)
                            Test_Labels.append(class_num)
                    else:
                        random_start = np.random.randint(low=0, high=(end_index - length))
                        sample = slice_data[random_start:random_start + length]
                        if len(sample) == length:
                            Data_Samples.append(sample)
                            Labels.append(class_num)
                start = start + step_size

        Data_Samples = np.array(Data_Samples)
        Labels = np.array(Labels)
        Test_Labels = np.array(Test_Labels)
        Test_DataSamples = np.array(Test_DataSamples)
        return Data_Samples, Labels, Test_DataSamples, Test_Labels

    def scalar_stand(Train_X, Val_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Val_X = scalar.transform(Val_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Val_X, Test_X

    def valid_test_slice(Train_X, Train_Y, valid_size=0.2):
        ss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size)
        for train_index, test_index in ss.split(Train_X, Train_Y):
            X_train, X_test = Train_X[train_index], Train_X[test_index]
            Y_train, Y_test = Train_Y[train_index], Train_Y[test_index]
            return X_train, Y_train, X_test, Y_test

    # 获取训练文件和测试文件的振动数据
    train_data = capture(file_path, train_filenames)
    # 获取采样后的训练集和测试集数据以及标签
    Train_X, Train_Y, Test_X, Test_Y = slice(train_data, sample_number)

    Train_X, Train_Y, Val_X, Val_Y = valid_test_slice(Train_X, Train_Y, 0.2)


    if noise == 'Gaussian':
        Train_X, _ = add_noise(Train_X, snr)
        Val_X, _ = add_noise(Val_X, snr)
        Test_X, _ = add_noise(Test_X, snr)

    elif noise == 'airplane':
        noise = np.load('data/audio/airplanenoise1e-10.npy')
        Train_X = add_audio_noise(noise, Train_X)
        Val_X = add_audio_noise(noise, Val_X)
        Test_X = add_audio_noise(noise, Test_X)

    elif noise == 'truck':
        noise = np.load('data/audio/trucknoise1e-10.npy')
        Train_X = add_audio_noise(noise, Train_X)
        Val_X = add_audio_noise(noise, Val_X)
        Test_X = add_audio_noise(noise, Test_X)
        # 将训练数据标准化
    if normal:
        # Train_X1, Val_X1, Test_X1 = scalar_stand(Train_X1, Val_X1, Test_X1)
        Train_X, Val_X, Test_X = scalar_stand(Train_X, Val_X, Test_X)

    # savemat('Test_JNU.mat', {'Test_X': Test_X,
    #                          'Test_Y': Test_Y,
    #                          'N_Test_X': Test_X1})
    Train_X, Val_X, Test_X = Train_X[:, np.newaxis, :], Val_X[:, np.newaxis, :], Test_X[:, np.newaxis, :]

    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Val_X = torch.tensor(Val_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Val_Y = torch.tensor(Val_Y, dtype=torch.float)
    Test_Y = torch.tensor(Test_Y, dtype=torch.long)


    # torch.save(data, train_path + 'testdataset.pth')
    return Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y
    # 生成Dataset
    # train_dataset = TensorDataset(Train_X, Train_Y)
    # test_dataset = TensorDataset(Test_X, Test_Y)
    # torch.save(train_dataset, train_path + 'traindataset.pth')
    #



if __name__ == '__main__':
    file_path = '../data/JNU'
    JNU_Processing(file_path=file_path)
    # for batch, (i, j) in enumerate(train_loader):
    #     print(batch, i, j)
    # train_file_path = r'C:\Users\0\Desktop\Paderborn_other\train_a.pt'
    # val_file_path = r'C:\Users\0\Desktop\Paderborn_other\val_a.pt'
    # #
    # train_loader, test_loader = Paderborn_DataLoader_Processed_Data(train_file_path, val_file_path, 60)
    # for batch, (i, j) in enumerate(train_loader):
    #     print(i)


