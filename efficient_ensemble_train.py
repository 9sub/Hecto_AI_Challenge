import os
import random
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim

import timm
import config # Assuming you have a config.py with train_root, img_size, batch_size, lr, epochs, seed, model_save_dir
import wandb

class FineGrainChangeLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if is_test:
            # 테스트 모드: 이미지 경로만 저장
            self.samples = sorted([
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith('.jpg')
            ])
            self.classes = [] # For test, no classes are needed
            self.class_to_idx = {} # For test, no class_to_idx is needed
            return

        # 1) base 클래스 디렉토리(예: '소나타_2013_2014') 목록 추출
        base_classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # 2) pose 디렉토리(예: side, front, back)는 각 base 클래스 내부에 있음
        #    combined_classes = ['소나타_2013_2014_side', '소나타_2013_2014_front', ...]
        combined_classes = []
        for base in base_classes:
            base_path = os.path.join(root_dir, base)
            for pose in sorted(os.listdir(base_path)):
                pose_path = os.path.join(base_path, pose)
                if os.path.isdir(pose_path):
                    combined_classes.append(f"{base}_{pose}")

        # 3) 클래스 → 인덱스 매핑
        self.classes = combined_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 4) (이미지 경로, label_idx) 쌍 생성
        self.samples = []
        for base in base_classes:
            base_path = os.path.join(root_dir, base)
            for pose in sorted(os.listdir(base_path)):
                pose_path = os.path.join(base_path, pose)
                if not os.path.isdir(pose_path):
                    continue
                combined = f"{base}_{pose}"
                idx = self.class_to_idx[combined]
                for fname in os.listdir(pose_path):
                    if not fname.lower().endswith('.jpg'):
                        continue
                    path = os.path.join(pose_path, fname)
                    self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            path = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img
        else:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            # base_class 라벨만 필요할 때:
            # combined_label = self.classes[label]
            # base_label = combined_label.rsplit('_', 1)[0]
            # base_label 은 예: '소나타_2013_2014'

            # For training, we'll use the 'label' (combined class index)
            return img, label

# 논문에 따른 단순한 EfficientNet-b0 모델 정의
class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        # 기존 classifier 제거하고 새로운 classifier 추가
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# 논문에 따른 단순한 Combination Layer
class SimpleCombinationLayer(nn.Module):
    def __init__(self, num_models, num_classes):
        super(SimpleCombinationLayer, self).__init__()
        # 각 모델의 출력을 결합하는 단순한 선형 레이어
        self.combination_layer = nn.Linear(num_models * num_classes, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, model_outputs):
        # model_outputs: list of [batch_size, num_classes] tensors
        concatenated = torch.cat(model_outputs, dim=1)  # [batch_size, num_models * num_classes]
        output = self.dropout(concatenated)
        output = self.combination_layer(output)
        return output

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"이미지 크기: {config.img_size}, 배치 크기: {config.batch_size}, lr: {config.lr}")

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 전체 데이터셋 로드
    full_dataset = FineGrainChangeLabelDataset(config.train_root, transform=None, is_test=False)
    print(f"총 이미지 수: {len(full_dataset.samples)}")

    # `FineGrainChangeLabelDataset`의 `self.classes`를 사용하여 전체 클래스 수를 가져옵니다.
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"총 클래스 수 (Combined Labels): {num_classes}")

    # 논문에 따른 Bagging: 데이터를 두 개의 disjoint subset으로 분할
    all_indices = np.arange(len(full_dataset))
    random.shuffle(all_indices)

    # 데이터를 정확히 두 개로 분할 (논문의 방법론)
    split_point = len(all_indices) // 2
    subset1_indices = all_indices[:split_point]
    subset2_indices = all_indices[split_point:]

    print(f"Subset 1 크기: {len(subset1_indices)}")
    print(f"Subset 2 크기: {len(subset2_indices)}")

    scaler = torch.cuda.amp.GradScaler()
    model_paths = []

    # =============================================
    # A. 두 개의 EfficientNet-b0 모델 개별 학습
    # =============================================
    print("\n--- A. 두 개의 EfficientNet-b0 모델 개별 학습 시작 ---")

    for model_idx in range(2):  # 논문에 따라 정확히 2개 모델
        print(f"\n--- EfficientNet-b0 모델 {model_idx+1}/2 학습 시작 ---")

        # 현재 모델이 학습할 데이터셋 선택
        if model_idx == 0:
            train_indices = subset1_indices
            val_indices = subset2_indices  # 다른 subset을 validation으로 사용
        else:
            train_indices = subset2_indices
            val_indices = subset1_indices

        # FineGrainChangeLabelDataset 인스턴스에 transform 적용
        model_train_dataset = Subset(
            FineGrainChangeLabelDataset(config.train_root, transform=train_transform, is_test=False),
            indices=train_indices
        )
        model_val_dataset = Subset(
            FineGrainChangeLabelDataset(config.train_root, transform=val_transform, is_test=False),
            indices=val_indices
        )

        model_train_loader = DataLoader(model_train_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=os.cpu_count() // 2,
                                         pin_memory=True)
        model_val_loader = DataLoader(model_val_dataset,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=os.cpu_count() // 2,
                                        pin_memory=True)

        # EfficientNet-b0 모델 초기화
        model = EfficientNetB0Model(num_classes=num_classes, pretrained=True)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        patience = 5
        no_improve_epochs = 0
        best_acc = float('-inf')

        # Wandb 초기화
        wandb.init(project="efficient-adaptive-ensemble",
                   name=f"EfficientNet-b0_Model_{model_idx+1}",
                   reinit=True)
        wandb.config.update({
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "img_size": config.img_size,
            "model_idx": model_idx + 1,
            "num_classes": num_classes,
            "use_amp": True,
        })

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0

            for images, labels in tqdm(model_train_loader,
                                      desc=f"[Model {model_idx+1} Epoch {epoch+1}] Train"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(model_train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for images, labels in tqdm(model_val_loader,
                                          desc=f"[Model {model_idx+1} Epoch {epoch+1}] Val"):
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

            avg_val_loss = val_loss / len(model_val_loader)
            val_acc = 100.0 * correct / total
            val_logloss = log_loss(all_labels, all_probs, labels=list(range(num_classes)))

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Model {model_idx+1} Epoch {epoch+1}/{config.epochs} | lr : {current_lr:.2e}\n"
                f"   train loss: {avg_train_loss:.4f}\n"
                f"   val loss: {avg_val_loss:.4f}\n"
                f"   val acc: {val_acc:.2f}%\n"
                f"   val LogLoss: {val_logloss:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_logloss": val_logloss,
            })

            model_save_name = f"EfficientNet-b0_Model_{model_idx+1}_best_acc.pth"
            model_save_path = os.path.join(config.model_save_dir, model_save_name)

            if val_acc > best_acc:
                no_improve_epochs = 0
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                print(f"[체크포인트] 모델 {model_idx+1} 최고 acc {best_acc:.2f}% 달성, 모델 저장.")
            else:
                no_improve_epochs += 1
                print(f"모델 {model_idx+1} acc 개선 없음: {no_improve_epochs} epoch 연속.")

            if no_improve_epochs >= patience:
                print(f"모델 {model_idx+1} 조기 종료합니다.")
                break

        model_paths.append(model_save_path)
        wandb.finish()

        # GPU 메모리 해제
        model.cpu()
        del model
        torch.cuda.empty_cache()
        print(f"모델 {model_idx+1} GPU 메모리에서 해제 및 캐시 비움.")

    # =============================================
    # B. Adaptive Combination Layer 학습
    # =============================================
    print("\n--- B. Adaptive Combination Layer 학습 시작 ---")

    # 학습된 모델들 다시 로드
    trained_models = []
    for i, model_path in enumerate(model_paths):
        model = EfficientNetB0Model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 기본 모델은 평가 모드로 고정
        trained_models.append(model)

    # Combination Layer 초기화
    combination_model = SimpleCombinationLayer(num_models=2, num_classes=num_classes).to(device)

    # 전체 데이터셋으로 Combination Layer 학습
    full_train_dataset = FineGrainChangeLabelDataset(config.train_root, transform=train_transform, is_test=False)
    full_train_loader = DataLoader(full_train_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=os.cpu_count() // 2,
                                     pin_memory=True)

    full_val_dataset = FineGrainChangeLabelDataset(config.train_root, transform=val_transform, is_test=False)
    full_val_loader = DataLoader(full_val_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=os.cpu_count() // 2,
                                   pin_memory=True)

    combination_criterion = nn.CrossEntropyLoss()
    combination_optimizer = optim.AdamW(combination_model.parameters(), lr=config.lr * 0.1)  # 더 낮은 학습률
    combination_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(combination_optimizer, T_max=20)  # 더 적은 epoch

    patience = 5
    no_improve_epochs = 0
    best_val_logloss = float('inf')

    wandb.init(project="efficient-adaptive-ensemble",
               name="Adaptive_Combination_Layer",
               reinit=True)
    wandb.config.update({
        "combination_epochs": 20,
        "combination_lr": config.lr * 0.1,
        "num_models": 2,
        "num_classes": num_classes,
    })

    for epoch in range(20):  # 논문에서는 fine-tuning이므로 적은 epoch 사용
        combination_model.train()
        train_loss = 0.0

        for images, labels in tqdm(full_train_loader,
                                  desc=f"[Combination Epoch {epoch+1}] Train"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            combination_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # 기본 모델들의 출력 얻기
                model_outputs = []
                for model in trained_models:
                    with torch.no_grad(): # Ensure base models are not trained here
                        output = model(images)
                    model_outputs.append(output)

                # Combination layer를 통한 최종 예측
                final_output = combination_model(model_outputs)
                loss = combination_criterion(final_output, labels)

            scaler.scale(loss).backward()
            scaler.step(combination_optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(full_train_loader)

        combination_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(full_val_loader,
                                      desc=f"[Combination Epoch {epoch+1}] Val"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    model_outputs = []
                    for model in trained_models:
                        output = model(images)
                        model_outputs.append(output)

                    final_output = combination_model(model_outputs)
                    loss = combination_criterion(final_output, labels)
                    val_loss += loss.item()

                    preds = final_output.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    probs = F.softmax(final_output, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(full_val_loader)
        val_acc = 100.0 * correct / total
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(num_classes)))

        combination_scheduler.step()
        current_lr = combination_scheduler.get_last_lr()[0]

        print(
            f"Combination Epoch {epoch+1}/20 | lr : {current_lr:.2e}\n"
            f"   train loss: {avg_train_loss:.4f}\n"
            f"   val loss: {avg_val_loss:.4f}\n"
            f"   val acc: {val_acc:.2f}%\n"
            f"   val LogLoss: {val_logloss:.4f}"
        )

        wandb.log({
            "combination_epoch": epoch + 1,
            "combination_lr": current_lr,
            "combination_train_loss": avg_train_loss,
            "combination_val_loss": avg_val_loss,
            "combination_val_accuracy": val_acc,
            "combination_val_logloss": val_logloss,
        })

        combination_save_path = os.path.join(config.model_save_dir, "Adaptive_Combination_Layer_best.pth")
        if val_logloss < best_val_logloss:
            no_improve_epochs = 0
            best_val_logloss = val_logloss
            torch.save(combination_model.state_dict(), combination_save_path)
            print(f"[체크포인트] Combination Layer 최고 LogLoss {best_val_logloss:.4f} 달성, 모델 저장.")
        else:
            no_improve_epochs += 1
            print(f"Combination Layer LogLoss 개선 없음: {no_improve_epochs} epoch 연속.")

        if no_improve_epochs >= patience:
            print(f"Combination Layer 조기 종료합니다.")
            break

    wandb.finish()

    # 메모리 해제
    for model in trained_models:
        model.cpu()
        del model
    combination_model.cpu()
    del combination_model
    torch.cuda.empty_cache()
    print("모든 모델 GPU 메모리에서 해제 및 캐시 비움.")

if __name__ == "__main__":
    train()