import os
import random

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold  # KFold 대신 StratifiedKFold 사용

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim

from sklearn.metrics import log_loss
import timm
from util.dataloader import ImageDataset, FineGrainImageDataset  # util 폴더의 dataloader 가정

from util.model import BilinearResDense, FineGrainResNet50, FineGrainConvNext, \
                       FineGrainResNext50, FineGrainResNet50LoRA
from util.loss import FocalLoss, ContrastiveLoss  # util 폴더의 loss 가정

import config  # config.py 파일이 있다고 가정
import wandb

# CUDA 사용 가능 여부에 따라 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"이미지 크기: {config.img_size}, 배치 크기: {config.batch_size}")

# =============================================
# 1) Train용 데이터 증강 (Augmentation) 정의
# =============================================
# train_transform = transforms.Compose([
#     # 1) RandomResizedCrop: 차량이 화면 중앙이 아닐 때도 잘 학습하도록 크롭 및 리사이즈
#     transforms.RandomResizedCrop(size=config.img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),

#     # 2) RandomHorizontalFlip: 좌우 반전
#     transforms.RandomHorizontalFlip(p=0.5),

#     # 3) ColorJitter: 밝기, 대비, 채도, 색상 랜덤 조정
#     transforms.ColorJitter(
#         brightness=0.3,   # 밝기 ±30%
#         contrast=0.3,     # 대비 ±30%
#         saturation=0.3,   # 채도 ±30%
#         hue=0.1           # 색상 ±10%
#     ),

#     # 4) RandomGrayscale: 5% 확률로 흑백으로 변환
#     transforms.RandomGrayscale(p=0.05),

#     # 5) RandomAffine: 회전 ±10도, 이동 ±5%, 확대·축소 ±5%, 기울임(shear)
#     transforms.RandomAffine(
#         degrees=10,
#         translate=(0.05, 0.05),
#         scale=(0.95, 1.05),
#         shear=5  # shear 각도 ±5도
#     ),

#     # 6) 마지막으로 Tensor 변환 및 정규화
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),

#     # 7) RandomErasing: 일정 확률로 이미지 일부분 지우기 (스티커나 번호판 가림에 대비)
#     transforms.RandomErasing(p=0.2,
#                              scale=(0.02, 0.2),
#                              ratio=(0.3, 3.3),
#                              value='random')
# ])


train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# =============================================
# 2) Validation용 데이터 변환 정의 (증강 없이 검증만)
# =============================================
val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =============================================
# 이후 코드는 기존과 동일 (K-Fold, LoRA, AMP 등)
# =============================================

# 전체 데이터셋 로드 (transform 없음)
full_dataset = FineGrainImageDataset(config.train_root, transform=None)
print(f"총 이미지 수: {len(full_dataset.samples)}")

# 계층적 K-겹 분할을 위해 타겟(레이블) 추출
targets = [label for _, label in full_dataset.samples]

class_names = full_dataset.classes
num_classes = len(class_names)

# K-겹 교차 검증 설정 (5-겹)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

# 결과 저장용 리스트
all_fold_val_accuracies = []
all_fold_val_loglosses = []

# LoRA 하이퍼파라미터
lora_rank = 8
lora_alpha = 16.0

# AMP(Automatic Mixed Precision)용 스케일러
scaler = torch.cuda.amp.GradScaler()

for fold, (train_index, val_index) in enumerate(skf.split(full_dataset, targets)):
    print(f"\n--- 폴드 {fold+1}/{n_splits} 시작 ---")

    # ────────── 3) Train/Validation Dataset & DataLoader ──────────
    train_dataset = Subset(
        FineGrainImageDataset(config.train_root, transform=train_transform),
        indices=train_index
    )
    val_dataset   = Subset(
        FineGrainImageDataset(config.train_root, transform=val_transform),
        indices=val_index
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count() // 2,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=os.cpu_count() // 2,
                            pin_memory=True)

    # ────────── 4) 모델, 손실 함수, 옵티마이저, 스케줄러 초기화 ──────────
    model = FineGrainResNet50LoRA(num_classes=num_classes,
                                  pretrained=True,
                                  lora_r=lora_rank,
                                  lora_alpha=lora_alpha)
    model.to(device)

    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=config.epochs)

    patience = 5
    no_improve_epochs = 0
    best_acc = float('-inf')
    best_val_logloss_for_best_acc = float('inf')

    wandb.init(project="car-classification",
               name=f"{config.model_name}_LoRA_fold_{fold+1}")
    wandb.config.update({
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "img_size": config.img_size,
        "model": config.model_name + "_LoRA_AMP_Augmented",
        "num_classes": num_classes,
        "fold": fold + 1,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "use_amp": True,
    })

    # ────────── 5) 에포크 루프 ──────────
    for epoch in range(config.epochs):
        # ======== Train 단계 ========
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader,
                                   desc=f"[폴드 {fold+1} 에포크 {epoch+1}/{config.epochs}] 훈련"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # (A) Mixed Precision 자동 스케일링
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            # (B) 스케일된 손실로 역전파
            scaler.scale(loss).backward()

            # (C) 옵티마이저 스텝 & 스케일러 업데이트
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ======== Validation 단계 ========
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc=f"[폴드 {fold+1} 에포크 {epoch+1}/{config.epochs}] 검증"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    probs = F.softmax(logits, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        val_logloss = log_loss(all_labels, all_probs,
                               labels=list(range(num_classes)))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"fold {fold+1} epoch {epoch+1}/{config.epochs} | lr : {current_lr:.2e}\n"
            f"  train loss: {avg_train_loss:.4f}\n"
            f"  val loss: {avg_val_loss:.4f}\n"
            f"  val acc: {val_acc:.2f}%\n"
            f"  val LogLoss: {val_logloss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_logloss": val_logloss,
            "lora_A_norm": model.classifier.A.norm().item(),
            "lora_B_norm": model.classifier.B.norm().item(),
        })

        # Early Stopping
        if val_acc > best_acc:
            no_improve_epochs = 0
            best_acc = val_acc
            best_val_logloss_for_best_acc = val_logloss
            torch.save(model.state_dict(),
                       f"./models/{config.model_name}_LoRA_fold_{fold+1}_best_acc.pth")
            print(f"[체크포인트] fold {fold+1}에서 최고 acc {best_acc:.2f}% 달성, 모델 저장.")
        else:
            no_improve_epochs += 1
            print(f"fold {fold+1}에서 acc 개선 없음: {no_improve_epochs} epoch 연속.")

        if no_improve_epochs >= patience:
            print(f"fold {fold+1}에서 {no_improve_epochs} epoch 동안 개선이 없어 조기 종료합니다.")
            break

    all_fold_val_accuracies.append(best_acc)
    all_fold_val_loglosses.append(best_val_logloss_for_best_acc)
    wandb.finish()

# ────────── 전체 Fold 요약 ──────────
print("\n--- K-겹 교차 검증 요약 ---")
print(f"{n_splits}개 폴드의 평균 검증 정확도: {np.mean(all_fold_val_accuracies):.2f}%")
print(f"검증 정확도의 표준 편차: {np.std(all_fold_val_accuracies):.2f}%")
print(f"최고 정확도일 때의 평균 검증 LogLoss: {np.mean(all_fold_val_loglosses):.4f}")
print(f"최고 정확도일 때의 검증 LogLoss 표준 편차: {np.std(all_fold_val_loglosses):.4f}")
