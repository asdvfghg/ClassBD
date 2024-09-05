import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, X, y, transform=None):
        # assert all(X.size(0) == tensor.size(0) for tensor in X)
        # assert all(y.size(0) == tensor.size(0) for tensor in y)

        self.X = X
        self.y = y

        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]

        if self.transform:
            x = self.transform(x)

        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.X)




class NoisyDataset(Dataset):
    """
    一个带有噪声的PyTorch数据集。

    参数:
    dataset: PyTorch数据集，原始数据集
    noise_factor: 噪声因子，用于控制噪声的强度
    """
    def __init__(self, dataset, snr=-6):
        self.dataset = dataset
        self.noise_factor = snr



    def wgn(self, x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(torch.abs(x) ** 2, dim=1) / x.shape[0]
        npower = xpower / snr
        return torch.randn_like(x) * torch.sqrt(npower)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        # 添加高斯噪声
        # noise = self.noise_factor * torch.randn(*data.size())
        noise = self.wgn(data, self.noise_factor)
        noisy_data = data + noise

        # 确保数据仍在有效范围内
        noisy_data = torch.clamp(noisy_data, 0., 1.)

        return noisy_data, target

    def __len__(self):
        return len(self.dataset)