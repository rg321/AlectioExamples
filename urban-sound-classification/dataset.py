from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class NumpySBDDataset(Dataset):
    def __init__(self, input_x, input_y):
        self.data = input_x
        self.labels = input_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_got = torch.from_numpy(self.data[idx]).float()

        data_got = torch.unsqueeze(data_got, 0)

        labels_got = torch.from_numpy(np.array(self.labels[idx]))
        return data_got, labels_got
