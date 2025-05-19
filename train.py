import os
import random

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim

from sklearn.metrics import log_loss


import config
from util.dataloader import ImageDataset
from util.model import ResNet152


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
])

dataset = ImageDataset(config.train_root, transform=None)
print(f"총이미지수 {len(dataset.samples) = }")

target = [label for _, label in dataset.samples]
class_name = dataset.classes

train_index, val_index = train_test_split(range(len(target)), test_size=0.2, stratify=target, random_state=config.seed)

train_dataset = Subset(ImageDataset(config.train_root, transform=train_transform), indices=train_index)
val_dataset = Subset(ImageDataset(config.train_root, transform=val_transform), indices=val_index)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)






model = ResNet152(num_classes=len(class_name)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=config.lr)

best_logloss = float('inf')



for epoch in range(config.epochs):
    #train

    model.train()
    train_loss = 0.
    for image, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Training"):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    #val
    model.eval()
    val_loss = 0.
    correct = 0
    total= 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Validation"):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            val_loss += loss.item()

            #acc
            _, preds = torch.max(outputs, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            #log loss
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100*correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_name))))


    print(f"Train loss:{avg_train_loss:4f} \n Valid Loss : {avg_val_loss:4f} \n Valid Acc : {val_acc:4f}%")

    if val_logloss < best_logloss:
        best_logloss = val_logloss
        torch.save(model.state_dict(), f"resnet152_basemodel.pth")
        print(f"Best model saved at epoch{epoch+1} LogLoss : {val_logloss:4f}")