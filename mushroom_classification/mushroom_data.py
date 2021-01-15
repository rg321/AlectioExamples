import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MushroomDataset(Dataset):
    """Mushroom Dataset from Kaggle"""

    def __init__(self, csv_file, transform=None):

        df = pd.read_csv(csv_file)
        df = pd.get_dummies(df)

        labels = df.iloc[:, :1]  # 1 is edible and 0 is poisonous
        data = df.iloc[:, 2:]
        print("Data shape:", data.shape)
        print("Labels shape:", labels.shape)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data[:])

    def __getitem__(self, idx):

        data_got = self.data.iloc[idx, :]
        labels_got = self.labels.iloc[idx, :]

        #         print("size of the data is:", data_got.shape)
        #         print("size of the labels is:", labels_got.shape)

        data_got = torch.from_numpy(data_got.to_numpy()).float()
        labels_got = torch.from_numpy(labels_got.to_numpy()).float()

        return data_got, labels_got
