import sklearn.model_selection
from scipy.io import loadmat, savemat
import os
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from scipy.fft import fft
from tqdm import tqdm
import acoustics.generator as ag

from utils.noise_observe import add_audio_noise

HBdata = ['K001',"K002",'K003','K004','K005','K006']
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']



def Paderborn_Processing(file_path, load, length=2048, use_sliding_window=True, step_size=2048,
                        sample_number=20, normal=True, noise = 'Gaussian', snr = -6):

    train_path = os.path.join(file_path, load)   # 训练集样本路径

    # 获得训练集和测试集文件夹下所有.mat文件名
    train_filenames = os.listdir(train_path)

    for i in train_filenames:
        if not i.endswith('.mat'):
            train_filenames.remove(i)

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

    def laplace_nosie(x, snr):
        P_signal = np.sum(abs(x) ** 2, axis=1) / len(x)
        P_noise = P_signal / (10 ** (snr / 10))
        P_noise = np.repeat(P_noise.reshape(-1, 1), x.shape[1], axis=1)
        white_noise = np.random.laplace(size=x.shape) * np.sqrt(P_noise)
        signal_add_noise = x + white_noise
        return signal_add_noise, x

    def pink_noise(x, snr):
        P_signal = np.sum(abs(x) ** 2, axis=1) / len(x)
        P_noise = P_signal / (10 ** (snr / 10))
        P_noise = np.repeat(P_noise.reshape(-1, 1), x.shape[1], axis=1)
        white_noise = ag.noise(x.shape[0] * x.shape[1], color='pink').reshape(x.shape[0], x.shape[1]) * np.sqrt(P_noise)
        signal_add_noise = x + white_noise
        return signal_add_noise, x

    def capture(path, filenames):
        """
        读取paderborn数据集中的振动数据(vibration_1)
        :param path: 文件目录路径
        :param filenames: 文件名
        :return: 每个文件的振动数据(dict)
        """
        data = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(path, i)
            file = loadmat(file_path)
            file_keys = i.strip('.mat')
            for j in file[file_keys][0][0]:
                if 'Name' in str(j.dtype):
                    if 'vibration_1' in j[0]['Name']:
                        index = np.argwhere(j[0]['Name'] == 'vibration_1')
                        data[file_keys] = j[0]['Data'][index][0][0][0]
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
        Labels = []  # 存储标签
        Test_Samples = []
        Test_Labels = []
        for key in tqdm(data_keys):
            slice_data = data[key]
            start = 0  # 采样信号的起点索引
            end_index = len(slice_data)
            if 'K0' in key:
                class_num = 0
            elif 'KA04' in key:
                class_num = 1
            elif 'KA15' in key:
                class_num = 2
            elif 'KA16' in key:
                class_num = 3
            elif 'KA22' in key:
                class_num = 4
            elif 'KA30' in key:
                class_num = 5
            elif 'KB23' in key:
                class_num = 6
            elif 'KB24' in key:
                class_num = 7
            elif 'KB27' in key:
                class_num = 8
            elif 'KI04' in key:
                class_num = 9
            elif 'KI14' in key:
                class_num = 9
            elif 'KI16' in key:
                class_num = 10
            elif 'KI17' in key:
                class_num = 11
            elif 'KI18' in key:
                class_num = 12
            elif 'KI21' in key:
                class_num = 13
            if key[-2:] == "20":
                samp_num1 = len(slice_data) // step_size
            else:
                samp_num1 = samp_num
            for i in range(samp_num1):
                if use_sliding_window:
                    sample = slice_data[start:start + length]
                    start = start + step_size
                else:
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                if key[-2:] == "20":
                    Test_Samples.append(sample)
                    Test_Labels.append(class_num)
                else:
                    Data_Samples.append(sample)
                    # Data_Samples.append(np.abs(fft(sample)))
                    Labels.append(class_num)
        Data_Samples = np.array(Data_Samples)
        Labels = np.array(Labels)
        Test_Samples = np.array(Test_Samples)
        Test_Labels = np.array(Test_Labels)
        return Data_Samples, Labels, Test_Samples, Test_Labels

    def scalar_stand(Train_X, Val_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Val_X = scalar.transform(Val_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Val_X, Test_X

    # 获取训练文件和测试文件的振动数据
    train_data = capture(train_path, train_filenames)

    # test_data = capture(test_path, test_filenames)

    # 获取采样后的训练集和测试集数据以及标签
    Train_X, Train_Y, Test_X, Test_Y = slice(train_data, sample_number)

    Train_X, Val_X, Train_Y, Val_Y = sklearn.model_selection.train_test_split(Train_X, Train_Y, train_size=0.8, test_size=0.2)

    if noise == 'Gaussian':
        Train_X, _ = add_noise(Train_X, snr)
        Val_X, _ = add_noise(Val_X, snr)
        Test_X, _ = add_noise(Test_X, snr)

    elif noise == 'pink':
        Train_X, _ = pink_noise(Train_X, snr)
        Val_X, _ = pink_noise(Val_X, snr)
        Test_X, _ = pink_noise(Test_X, snr)

    elif noise == 'Laplace':
        Train_X, _ = laplace_nosie(Train_X, snr)
        Val_X, _ = laplace_nosie(Val_X, snr)
        Test_X, _ = laplace_nosie(Test_X, snr)

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
        Train_X, Val_X, Test_X = scalar_stand(Train_X, Val_X, Test_X)


    Train_X, Val_X, Test_X = Train_X[:, np.newaxis, :], Val_X[:, np.newaxis, :], Test_X[:, np.newaxis, :]



    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Val_X = torch.tensor(Val_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Val_Y = torch.tensor(Val_Y, dtype=torch.float)
    Test_Y = torch.tensor(Test_Y, dtype=torch.long)


    return Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y




if __name__ == '__main__':
    file_path = '../data/Paderborn'
    load = 'N15_M07_F10'
    Paderborn_Processing(file_path=file_path, load=load, noise='airplane')



