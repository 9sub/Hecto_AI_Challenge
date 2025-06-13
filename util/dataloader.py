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



class MultiTaskImageDataset(Dataset):
    def __init__(self, root_dir, crop_dir, transform=None):
        
        self.root_dir = root_dir
        self.crop_dir = crop_dir
        self.transform = transform
        self.samples = []

        self.classes = [
            d for d in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.class_to_index = {class_ : i for i, class_ in enumerate(self.classes)}

        for class_name in self.classes:
            origin_class_dir = os.path.join(root_dir, class_name)
            crop_class_dir = os.path.join(crop_dir, class_name)

            if not os.path.exists(crop_class_dir):
                raise ValueError(f"크롭 폴더가 없습니다: {crop_class_dir}")
            
            for file in os.listdir(origin_class_dir):
                if file.lower().endswith('.jpg'):
                    origin_path = os.path.join(origin_class_dir, file)
                    crop_path = os.path.join(crop_class_dir, file)
                    if os.path.exists(crop_path):
                        label = self.class_to_index[class_name]
                        self.samples.append((origin_path, crop_path, label))
                    else:
                        print(f"크롭 이미지 없음 {crop_path}")


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        origin_path, crop_path, label = self.samples[index]
        origin_img = Image.open(origin_path).convert('RGB')
        crop_img = Image.open(crop_path).convert('RGB')

        if self.transform:
            origin_img = self.transform(origin_img)
            crop_img = self.transform(crop_img)

        return origin_img, crop_img, label



class FineGrainImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test

        # 테스트면 파일 경로 리스트만 저장
        if is_test:
            self.samples = sorted([
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith('.jpg')
            ])
            return

        # 학습/검증용
        # 클래스 이름은 폴더명(예: "소나타_2018_2019")
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            folder = os.path.join(root_dir, c)
            for f in os.listdir(folder):
                if f.lower().endswith('.jpg'):
                    self.samples.append(
                        (os.path.join(folder,f), self.class_to_idx[c])
                    )

    def __len__(self):
        return len(self.samples) if not self.is_test else len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            path = self.samples[idx]
            img  = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img
        else:
            path, lbl = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, lbl



class TTADataset(Dataset):
    def __init__(self, root_dir, is_test=True): # transform 제거
        self.root_dir = root_dir
        self.is_test = is_test
        self.image_files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) # 다양한 확장자 지원
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        return image, image_path # 원본 PIL 이미지와 경로 반환


class FineGrainImageDatasetContrastiveLoss(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test

        if is_test:
            self.samples = sorted([
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith('.jpg')
            ])
            return

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            folder = os.path.join(root_dir, c)
            for f in os.listdir(folder):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(folder, f), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 첫 번째 이미지
        path1, lbl1 = self.samples[idx]
        img1 = Image.open(path1).convert('RGB')
        if self.transform: img1 = self.transform(img1)

        # 두 번째 이미지 랜덤 선택
        idx2 = random.randint(0, len(self.samples)-1)
        path2, lbl2 = self.samples[idx2]
        img2 = Image.open(path2).convert('RGB')
        if self.transform: img2 = self.transform(img2)

        # pair label: 같은 클래스면 1, 아니면 0
        pair_lbl = 1 if lbl1 == lbl2 else 0

        return img1, img2, lbl1, lbl2, pair_lbl



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
            # 테스트 모드: 경로만 있음
            path = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        else:
            # 학습 모드: 경로와 라벨이 있음
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            # base_class 라벨만 필요할 때:
            combined_label = self.classes[label]
            base_label = combined_label.rsplit('_', 1)[0]
            # base_label 은 예: '소나타_2013_2014'

            return img, label, base_label