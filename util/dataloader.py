import os
import random

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim

from sklearn.metrics import log_loss


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            for file in sorted(os.listdir(self.root_dir)):
                if file.lower().endswith('jpg'):
                    img_path = os.path.join(self.root_dir, file)
                    self.samples.append((img_path,))
        else:
            all_class = sorted(os.listdir(self.root_dir))
            self.classes = [
                class_ for class_ in all_class
                if os.path.isdir(os.path.join(self.root_dir, class_))
            ]
            self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

            for class_name in self.classes:
                class_folder = os.path.join(root_dir, class_name)
                for file in os.listdir(class_folder):
                    if file.lower().endswith('.jpg'):
                        img_path = os.path.join(class_folder, file)
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.is_test:
            img_path = self.samples[index][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label


