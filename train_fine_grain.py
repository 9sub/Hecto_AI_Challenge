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
import timm
from util.dataloader import ImageDataset, FineGrainImageDataset

from util.model import BilinearResDense, FineGrainResNet50

import config
import wandb




#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(config.img_size, config.batch_size)

train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

dataset = FineGrainImageDataset(config.train_root, transform=None)
print(f"총이미지수 {len(dataset.samples) = }")

target = [label for _, label in dataset.samples]


train_index, val_index = train_test_split(range(len(target)), test_size=0.2, stratify=target, random_state=config.seed)

train_dataset = Subset(FineGrainImageDataset(config.train_root, transform=train_transform), indices=train_index)
val_dataset = Subset(FineGrainImageDataset(config.train_root, transform=val_transform), indices=val_index)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

class_name = dataset.classes


#model = VisionTransformer(num_classes=len(class_name)).to(device)
#model = timm.create_model('cait_xxs24_384', pretrained=True, num_classes=396)

model = FineGrainResNet50(num_classes=len(class_name), pretrained=True)

model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

# 얼리스토핑용 변수
patience = 5

no_improve_epochs = 0
best_logloss = float('inf')
best_acc = float('-inf')

wandb.init(project="car-classification", name=f"{config.model_name}")
wandb.config.update({
    "epochs": config.epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.lr,
    "img_size": config.img_size,
    "model": config.model_name,
    "num_classes": len(class_name),
})


for epoch in range(config.epochs):
    #train

    model.train()
    train_loss = 0.
    #for image, label in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Training"):
    for image, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Training"):
        image, labels = image.to(device), labels.to(device)
        #image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        #outputs = model(image)
        #loss = criterion(outputs, label)
        logits = model(image)
        #loss = criterion(logits, label)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    #val
    model.eval()
    val_loss    = 0.0
    correct     = 0
    total       = 0
    all_probs   = []
    all_labels  = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Validation"):
            images, labels = images.to(device), labels.to(device)

            # 1) forward + loss
            logits = model(images)
            loss   = criterion(logits, labels)
            val_loss += loss.item()

            # 2) accuracy
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # 3) logloss data
            probs      = F.softmax(logits, dim=1)
            all_probs .extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4) metrics
    avg_val_loss = val_loss / len(val_loader)
    val_acc      = 100.0 * correct / total
    val_logloss  = log_loss(all_labels, all_probs, 
                            labels=list(range(len(class_name))))



    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # ——— Print Summary ———
    print(
        f"Epoch {epoch+1}/{config.epochs} | LR: {current_lr:.2e}\n"
        f"  Train Loss: {avg_train_loss:.4f}\n"
        f"  Val Loss  : {avg_val_loss:.4f}\n"
        f"  Val Acc   : {val_acc:.2f}%\n"
        f"  Val LogLoss: {val_logloss:.4f}"
    )

    wandb.log({
        "epoch": epoch+1,
        "learning_rate": current_lr,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc,
        "val_logloss": val_logloss,
    })


    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"./models/{config.model_name}_basemodel.pth")
        print(f"[Checkpoint] Saved best combined acc {best_acc:.2f}% at epoch {epoch+1}")
    else:
        no_improve_epochs += 1
        print(f"No improvement in accuracy for {no_improve_epochs} epochs.")
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
        break

wandb.finish()