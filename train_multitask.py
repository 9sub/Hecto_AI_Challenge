import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import wandb
import torch.nn.functional as F

import config
from util.dataloader import MultiTaskImageDataset
from util.model import MultiTaskModel
from util.set_seed import seed_all

seed_all(config.seed, device='mps')

# 데이터 변환 예시
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

root_dir = config.train_root
crop_dir = config.crop_root

dataset = MultiTaskImageDataset(root_dir, crop_dir, transform)

device = torch.device('mps')
model = MultiTaskModel(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer =  optim.AdamW(model.parameters(), lr=config.lr)


targets = [label for _, _, label in dataset.samples]
train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets, random_state=config.seed)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)


train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

best_logloss = float('inf')

wandb.init(project="car-classification", name="multitask_efficientnet_v2_s")
wandb.config.update({
    "epochs": config.epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.lr,
    "img_size": config.img_size,
    "model": config.model_name,
    "num_classes": len(dataset.classes),
})


for epoch in range(config.epochs):
    #train

    model.train()
    train_loss = 0.
    for orig_img, crop_img, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Training"):
        orig_img, crop_img, labels = orig_img.to(device), crop_img.to(device), labels.to(device)

        optimizer.zero_grad()
        out_full, out_crop = model(orig_img, crop_img)

        loss_full = criterion(out_full, labels)
        loss_crop = criterion(out_crop, labels)
        loss = loss_full + loss_crop

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * orig_img.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    avg_train_loss = train_loss / len(train_loader)

    #val
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for orig_img, crop_img, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{config.epochs}] Validation"):
            orig_img, crop_img, labels = orig_img.to(device), crop_img.to(device), labels.to(device)
            out_full, out_crop = model(orig_img, crop_img)

            loss = criterion(out_full, labels)  # validation은 전체 이미지 출력만 사용 예시
            val_loss += loss.item() * orig_img.size(0)

            preds = out_full.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(out_full, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / total
    val_acc = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(dataset.classes))))

    print(f"Train loss: {avg_train_loss:.4f} | Valid loss: {avg_val_loss:.4f} | Valid Acc: {val_acc:.2f}% | LogLoss: {val_logloss:.4f}")

    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc,
        "val_logloss": val_logloss,
    })


    if val_logloss < best_logloss:
        best_logloss = val_logloss
        torch.save(model.state_dict(), f"./models/{config.model_name}_basemodel.pth")
        print(f"Best model saved at epoch{epoch+1} LogLoss : {val_logloss:4f}")

wandb.finish()