import torch.utils.data as data
from torchvision import datasets, transforms
import torch

import os
import numpy as np
from numpy.random import randint

class VideoMNISTDataset(data.Dataset):
    def __init__(self, transform=None, train=True, num_segments=8, dense_segments=False, random_shift=True):
        self.transform = transform
        self.train = train
        self.num_segments = num_segments
        self.dense_segments = dense_segments
        self.random_shift = random_shift


        dataset = datasets.MNIST(root='./data', train=self.train, download=True, transform=self.transform)
        idx_1 = dataset.targets == 1
        dataset.targets = dataset.targets[idx_1]
        dataset.data = dataset.data[idx_1]
        dataloader_1 = torch.utils.data.DataLoader(dataset, batch_size=6742, shuffle=False)
        data_1 = list(dataloader_1)[0][0]
        label_1 = list(dataloader_1)[0][1]

        dataset = datasets.MNIST(root='./data', train=self.train, download=False, transform=self.transform)
        idx_2 = dataset.targets == 2
        dataset.targets = dataset.targets[idx_2]
        dataset.data = dataset.data[idx_2]
        dataloader_2 = torch.utils.data.DataLoader(dataset, batch_size=5958, shuffle=False)
        data_2 = list(dataloader_2)[0][0]
        label_2 = list(dataloader_2)[0][1]

        segments = 60

        new_length_1 = label_1.size(0) // segments
        new_length_2 = label_2.size(0) // segments
        label_1 = label_1[0:new_length_1]
        data_1 = data_1[0:new_length_1 * segments]
        label_2 = label_2[0:new_length_2]
        data_2 = data_2[0:new_length_2 * segments]

        c, w, h = data_1.size(1), data_1.size(2), data_1.size(3)

        data_1 = data_1.view(new_length_1, segments*c, w, h)
        data_2 = data_2.view(new_length_2, segments*c, w, h)

        self.data = torch.cat([data_1, data_2], dim=0)
        self.targets = torch.cat([label_1, label_2], dim=0)

        print(self.data.size())
        print(self.targets.size())

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, index):

        if not self.dense_segments:
            if self.random_shift:
                average_duration = 60 // self.num_segments
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                             size=self.num_segments)
                offsets = [(x in offsets) for x in range(60)]
            else:
                tick = 60 / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                offsets += 1
                offsets = [(x in offsets) for x in range(60)]
            video_data = self.data[index][offsets, :, :]
            print('video_data:', video_data.size())
        else:
            video_data = self.data[index]
        video_target = self.targets[index]

        return video_data, video_target

