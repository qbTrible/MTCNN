from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
import random

class FaceDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(random.sample(open(os.path.join(path, "positive.txt")).readlines(), 350000))
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(random.sample(open(os.path.join(path, "part.txt")).readlines(), 350000))

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)

        # print(img_data.shape)
        img_data = img_data.permute(2, 0, 1)
        # print(img_data.shape)

        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset(r"F:\celeba1\12")
    print(dataset[0][0].shape)

