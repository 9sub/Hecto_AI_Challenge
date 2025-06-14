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

# util.dataloader에서 FineGrainImageDataset을 직접 가져오는 대신, 여기에 수정된 버전을 정의합니다.

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

# --- 수정된 FineGrainImageDataset 클래스 ---
class FineGrainImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False, exclude_files=None):
        self.transform = transform
        self.is_test = is_test
        self.exclude_files = set(exclude_files) if exclude_files else set()

        # 테스트면 파일 경로 리스트만 저장
        if is_test:
            all_files = [
                f for f in os.listdir(root_dir)
                if f.lower().endswith('.jpg')
            ]
            self.samples = sorted([
                os.path.join(root_dir, f) for f in all_files
                if f not in self.exclude_files # 제외할 파일 필터링
            ])
            return

        # 학습/검증용
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            folder = os.path.join(root_dir, c)
            for f_name in os.listdir(folder):
                if f_name.lower().endswith('.jpg'):
                    # 파일명 전체 (예: "뉴_A6_2012_2014_0046.jpg")를 exclude_files에서 확인
                    # 또는, 만약 exclude_files에 확장자가 없는 이름만 있다면,
                    # base_name = os.path.splitext(f_name)[0]
                    # if base_name not in self.exclude_files:
                    
                    if f_name not in self.exclude_files: # 제외할 파일 필터링
                        self.samples.append(
                            (os.path.join(folder, f_name), self.class_to_idx[c])
                        )
        print(f"Dataset initialized. Excluded {len(exclude_files) if exclude_files else 0} files. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            path = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img
        else:
            path, lbl = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, lbl

# --- train 함수 (이전 답변과 동일하게 logloss 기준으로 early stopping 적용) ---
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

    # --- 제외할 파일 목록 정의 ---
    # 여기에 제공해주신 파일명들을 .jpg 확장자와 함께 추가합니다.
    # 파일명이 `차종_세대_연도_연도_일련번호` 형식으로 되어있으므로,
    # 실제 파일 시스템에 저장된 이름과 정확히 일치해야 합니다.
    exclude_file_names = [
        "뉴_A6_2012_2014_0046.jpg", "4시리즈_G22_2024_2025_0031.jpg", "Q7_4M_2020_2023_0011.jpg",
        "A_클래스_W177_2020_2025_0034.jpg", "레인지로버_스포츠_2세대_2018_2022_0014.jpg", "레인지로버_스포츠_2세대_2018_2022_0017.jpg",
        "EQA_H243_2021_2024_0063.jpg", "GLS_클래스_X167_2020_2024_0013.jpg", "레인지로버_5세대_2023_2024_0030.jpg",
        "911_992_2020_2024_0006.jpg", "G_클래스_W463b_2019_2025_0030.jpg", "G_클래스_W463b_2019_2025_0049.jpg",
        "더_뉴_그랜드_스타렉스_2018_2021_0078.jpg", "더_뉴_그랜드_스타렉스_2018_2021_0079.jpg", "더_뉴_그랜드_스타렉스_2018_2021_0080.jpg",
        "아반떼_N_2022_2023_0035.jpg", "아반떼_N_2022_2023_0064.jpg", "더_뉴_아반떼_2014_2016_0031.jpg",
        "5시리즈_G60_2024_2025_0056.jpg", "5시리즈_G60_2024_2025_0010.jpg", "K5_2세대_2016_2018_0007.jpg",
        "뉴_체어맨_W_2012_2016_0009.jpg", "뉴_체어맨_W_2012_2016_0022.jpg", "뉴_체어맨_W_2012_2016_0043.jpg",
        "뉴_체어맨_W_2012_2016_0005.jpg", "싼타페_TM_2019_2020_0009.jpg", "머스탱_2015_2023_0086.jpg",
        "디_올뉴니로_2022_2025_0061.jpg", "디_올뉴니로_2022_2025_0085.jpg", "디_올뉴니로_2022_2025_0066.jpg",
        "더_올뉴G80_2021_2024_0070.jpg", "더_올뉴G80_2021_2024_0023.jpg", "더_올뉴G80_2021_2024_0054.jpg",
        "더_올뉴G80_2021_2024_0076.jpg", "X4_F26_2015_2018_0068.jpg", "더_뉴_QM6_2024_2025_0040.jpg",
        "GLB_클래스_X247_2020_2023_0008.jpg", "A8_D5_2018_2023_0084.jpg", "K3_2013_2015_0045.jpg",
        "Q5_FY_2021_2024_0032.jpg", "디_올뉴니로EV_2023_2024_0062.jpg", "디_올뉴니로EV_2023_2024_0012.jpg",
        "디_올뉴니로EV_2023_2024_0070.jpg", "디_올뉴니로EV_2023_2024_0048.jpg", "디_올뉴니로EV_2023_2024_0011.jpg",
        "뉴_QM5_2012_2014_0019.jpg", "뉴_QM5_2012_2014_0002.jpg", "뉴_QM5_2012_2014_0071.jpg",
        "뉴_QM5_2012_2014_0042.jpg", "뉴_QM5_2012_2014_0017.jpg", "뉴_QM5_2012_2014_0012.jpg",
        "그랜저TG_2007_2008_0015.jpg", "그랜저TG_2007_2008_0023.jpg", "그랜저TG_2007_2008_0004.jpg",
        "그랜저TG_2007_2008_0075.jpg", "그랜저TG_2007_2008_0024.jpg", "그랜저TG_2007_2008_0069.jpg",
        "그랜저TG_2007_2008_0008.jpg", "그랜저TG_2007_2008_0022.jpg", "S_클래스_W223_2021_2025_0071.jpg",
        "S_클래스_W223_2021_2025_0008.jpg", "뉴_G80_2025_2026_0059.jpg", "뉴_G80_2025_2026_0043.jpg",
        "뉴_G80_2025_2026_0054.jpg", "뉴_G80_2025_2026_0023.jpg", "뉴_G80_2025_2026_0066.jpg",
        "뉴_G80_2025_2026_0011.jpg", "뉴_G80_2025_2026_0080.jpg", "뉴_G80_2025_2026_0042.jpg",
        "뉴_G80_2025_2026_0029.jpg", "뉴_G80_2025_2026_0070.jpg", "뉴_G80_2025_2026_0034.jpg",
        "파나메라_2010_2016_0000.jpg", "파나메라_2010_2016_0036.jpg", "카이엔_PO536_2019_2023_0035.jpg",
        "카이엔_PO536_2019_2023_0054.jpg", "더_기아_레이_EV_2024_2025_0035.jpg", "더_기아_레이_EV_2024_2025_0064.jpg",
        "더_기아_레이_EV_2024_2025_0047.jpg", "더_기아_레이_EV_2024_2025_0069.jpg", "더_기아_레이_EV_2024_2025_0078.jpg",
        "더_기아_레이_EV_2024_2025_0040.jpg", "글래디에이터_JT_2020_2023_0075.jpg", "뉴_SM5_임프레션_2008_2010_0070.jpg",
        "뉴_SM5_임프레션_2008_2010_0033.jpg", "뉴_SM5_임프레션_2008_2010_0077.jpg", "뉴_SM5_임프레션_2008_2010_0019.jpg",
        "뉴_SM5_임프레션_2008_2010_0041.jpg", "YF쏘나타_2009_2012_0042.jpg", "YF쏘나타_2009_2012_0064.jpg",
        "YF쏘나타_2009_2012_0043.jpg", "YF쏘나타_2009_2012_0003.jpg", "YF쏘나타_2009_2012_0048.jpg",
        "YF쏘나타_2009_2012_0032.jpg", "YF쏘나타_2009_2012_0068.jpg", "YF쏘나타_2009_2012_0011.jpg",
        "YF쏘나타_2009_2012_0053.jpg", "더_뉴_스파크_2019_2022_0040.jpg", "더_올뉴투싼_하이브리드_2021_2023_0022.jpg",
        "더_올뉴투싼_하이브리드_2021_2023_0024.jpg", "더_올뉴투싼_하이브리드_2021_2023_0069.jpg", "더_올뉴투싼_하이브리드_2021_2023_0037.jpg",
        "더_올뉴투싼_하이브리드_2021_2023_0042.jpg", "더_올뉴투싼_하이브리드_2021_2023_0068.jpg", "더_올뉴투싼_하이브리드_2021_2023_0038.jpg",
        "아베오_2012_2016_0014.jpg", "아베오_2012_2016_0065.jpg", "아베오_2012_2016_0061.jpg",
        "아베오_2012_2016_0023.jpg", "아베오_2012_2016_0049.jpg", "아베오_2012_2016_0085.jpg",
        "아베오_2012_2016_0052.jpg", "아베오_2012_2016_0018.jpg", "레인지로버_4세대_2018_2022_0048.jpg",
        "C_클래스_W204_2008_2015_0068.jpg", "더_뉴_코나_2021_2023_0081.jpg", "컨티넨탈_GT_3세대_2018_2023_0007.jpg",
        "7시리즈_G11_2016_2018_0040.jpg", "2시리즈_액티브_투어러_U06_2022_2024_0004.jpg", "GLE_클래스_W167_2019_2024_0068.jpg",
        "RAV4_5세대_2019_2024_0020.jpg", "더_뉴_파사트_2012_2019_0067.jpg", "X7_G07_2019_2022_0052.jpg",
        "G_클래스_W463_2009_2017_0011.jpg", "2시리즈_그란쿠페_F44_2020_2024_0042.jpg", "6시리즈_GT_G32_2018_2020_0018.jpg",
        "Q7_4M_2016_2019_0045.jpg", "XJ_8세대_2010_2019_0033.jpg", "XJ_8세대_2010_2019_0085.jpg",
        "XJ_8세대_2010_2019_0026.jpg", "XJ_8세대_2010_2019_0043.jpg", "XJ_8세대_2010_2019_0083.jpg",
        "XJ_8세대_2010_2019_0062.jpg", "XJ_8세대_2010_2019_0064.jpg", "XJ_8세대_2010_2019_0063.jpg",
        "XJ_8세대_2010_2019_0050.jpg", "XJ_8세대_2010_2019_0029.jpg", "XJ_8세대_2010_2019_0041.jpg",
        "XJ_8세대_2010_2019_0005.jpg", "XJ_8세대_2010_2019_0084.jpg", "XJ_8세대_2010_2019_0052.jpg",
        "뉴_ES300h_2013_2015_0000.jpg", "뉴_CC_2012_2016_0001.jpg", "뉴_CC_2012_2016_0002.jpg",
        "디_올뉴싼타페_2024_2025_0052.jpg", "디_올뉴싼타페_2024_2025_0039.jpg", "디_올뉴싼타페_2024_2025_0020.jpg",
        "디_올뉴싼타페_2024_2025_0024.jpg", "디_올뉴싼타페_2024_2025_0080.jpg", "디_올뉴싼타페_2024_2025_0047.jpg",
        "디_올뉴싼타페_2024_2025_0009.jpg", "디_올뉴싼타페_2024_2025_0015.jpg", "디_올뉴싼타페_2024_2025_0007.jpg",
        "디_올뉴싼타페_2024_2025_0050.jpg", "마칸_2022_2024_0011.jpg", "마칸_2022_2024_0033.jpg",
        "마칸_2022_2024_0049.jpg", "마칸_2022_2024_0053.jpg", "마칸_2022_2024_0052.jpg",
        "마칸_2022_2024_0016.jpg", "마칸_2022_2024_0025.jpg", "3시리즈_F30_2013_2018_0069.jpg",
        "3시리즈_F30_2013_2018_0036.jpg", "뉴_SM5_플래티넘_2013_2014_0084.jpg", "뉴_SM5_플래티넘_2013_2014_0004.jpg",
        "뉴_SM5_플래티넘_2013_2014_0074.jpg", "뉴_SM5_플래티넘_2013_2014_0007.jpg", "뉴_SM5_플래티넘_2013_2014_0047.jpg",
        "뉴_SM5_플래티넘_2013_2014_0023.jpg", "뉴_SM5_플래티넘_2013_2014_0086.jpg", "뉴_SM5_플래티넘_2013_2014_0061.jpg",
        "아반떼_MD_2011_2014_0082.jpg", "아반떼_MD_2011_2014_0009.jpg", "아반떼_MD_2011_2014_0081.jpg",
        "콰트로포르테_2017_2022_0074.jpg", "레이_2012_2017_0063.jpg", "YF쏘나타_하이브리드_2011_2015_0033.jpg",
        "YF쏘나타_하이브리드_2011_2015_0028.jpg", "YF쏘나타_하이브리드_2011_2015_0072.jpg", "YF쏘나타_하이브리드_2011_2015_0043.jpg",
        "YF쏘나타_하이브리드_2011_2015_0003.jpg", "YF쏘나타_하이브리드_2011_2015_0013.jpg", "YF쏘나타_하이브리드_2011_2015_0014.jpg",
        "YF쏘나타_하이브리드_2011_2015_0060.jpg", "YF쏘나타_하이브리드_2011_2015_0020.jpg", "YF쏘나타_하이브리드_2011_2015_0035.jpg",
        "YF쏘나타_하이브리드_2011_2015_0016.jpg", "디_올뉴그랜저_2023_2025_0039.jpg", "디_올뉴그랜저_2023_2025_0082.jpg",
        "디_올뉴그랜저_2023_2025_0002.jpg", "디_올뉴그랜저_2023_2025_0057.jpg", "디_올뉴그랜저_2023_2025_0049.jpg",
        "디_올뉴그랜저_2023_2025_0072.jpg", "디_올뉴그랜저_2023_2025_0046.jpg", "디_올뉴그랜저_2023_2025_0020.jpg",
        "디_올뉴그랜저_2023_2025_0062.jpg", "디_올뉴그랜저_2023_2025_0034.jpg", "디_올뉴그랜저_2023_2025_0086.jpg",
        "디_올뉴그랜저_2023_2025_0018.jpg", "디_올뉴그랜저_2023_2025_0019.jpg", "디_올뉴그랜저_2023_2025_0038.jpg",
        "CLS_클래스_C257_2019_2023_0021.jpg", "XF_X260_2016_2020_0023.jpg", "뉴_GV80_2024_2025_0033.jpg",
        "뉴_GV80_2024_2025_0012.jpg", "뉴_GV80_2024_2025_0021.jpg", "뉴_GV80_2024_2025_0037.jpg",
        "뉴_GV80_2024_2025_0061.jpg", "뉴_GV80_2024_2025_0010.jpg", "뉴_GV80_2024_2025_0069.jpg",
        "뉴_GV80_2024_2025_0080.jpg", "라브4_4세대_2013_2018_0065.jpg", "라브4_4세대_2013_2018_0000.jpg",
        "라브4_4세대_2013_2018_0085.jpg", "라브4_4세대_2013_2018_0018.jpg", "라브4_4세대_2013_2018_0016.jpg",
        "라브4_4세대_2013_2018_0064.jpg", "라브4_4세대_2013_2018_0009.jpg", "라브4_4세대_2013_2018_0008.jpg",
        "라브4_4세대_2013_2018_0083.jpg", "라브4_4세대_2013_2018_0014.jpg", "라브4_4세대_2013_2018_0010.jpg",
        "라브4_4세대_2013_2018_0070.jpg", "라브4_4세대_2013_2018_0022.jpg", "라브4_4세대_2013_2018_0043.jpg",
        "그랜드_체로키_WL_2021_2023_0018.jpg", "5008_2세대_2021_2024_0055.jpg", "5008_2세대_2021_2024_0051.jpg",
        "더_뉴_K3_2세대_2022_2024_0001.jpg", "ES300h_7세대_2019_2026_0028.jpg", "프리우스_4세대_2019_2022_0050.jpg",
        "프리우스_4세대_2019_2022_0000.jpg", "프리우스_4세대_2019_2022_0052.jpg", "Q50_2014_2017_0031.jpg",
        "7시리즈_F01_2009_2015_0029.jpg", "7시리즈_F01_2009_2015_0044.jpg", "뉴쏘렌토_R_2013_2014_0077.jpg",
        "뉴쏘렌토_R_2013_2014_0058.jpg", "뉴쏘렌토_R_2013_2014_0018.jpg", "뉴쏘렌토_R_2013_2014_0004.jpg",
        "뉴쏘렌토_R_2013_2014_0016.jpg", "뉴쏘렌토_R_2013_2014_0009.jpg", "뉴쏘렌토_R_2013_2014_0083.jpg",
        "뉴쏘렌토_R_2013_2014_0024.jpg", "뉴쏘렌토_R_2013_2014_0042.jpg", "뉴쏘렌토_R_2013_2014_0079.jpg",
        "뉴쏘렌토_R_2013_2014_0002.jpg", "박스터_718_2017_2024_0034.jpg", "박스터_718_2017_2024_0011.jpg",
        "박스터_718_2017_2024_0041.jpg", "박스터_718_2017_2024_0002.jpg", "박스터_718_2017_2024_0044.jpg",
        "박스터_718_2017_2024_0001.jpg", "박스터_718_2017_2024_0030.jpg", "박스터_718_2017_2024_0077.jpg",
        "박스터_718_2017_2024_0007.jpg", "박스터_718_2017_2024_0062.jpg", "박스터_718_2017_2024_0078.jpg",
        "박스터_718_2017_2024_0065.jpg", "박스터_718_2017_2024_0069.jpg", "박스터_718_2017_2024_0051.jpg",
        "박스터_718_2017_2024_0068.jpg", "박스터_718_2017_2024_0036.jpg", "박스터_718_2017_2024_0061.jpg",
        "박스터_718_2017_2024_0014.jpg", "X3_G01_2022_2024_0029.jpg", "타이칸_2021_2025_0065.jpg",
        "타이칸_2021_2025_0003.jpg", "E_클래스_W212_2010_2016_0069.jpg", "SM7_뉴아트_2008_2011_0050.jpg",
        "SM7_뉴아트_2008_2011_0030.jpg", "SM7_뉴아트_2008_2011_0017.jpg", "SM7_뉴아트_2008_2011_0049.jpg",
        "SM7_뉴아트_2008_2011_0069.jpg", "SM7_뉴아트_2008_2011_0040.jpg", "SM7_뉴아트_2008_2011_0020.jpg",
        "SM7_뉴아트_2008_2011_0067.jpg", "SM7_뉴아트_2008_2011_0083.jpg", "SM7_뉴아트_2008_2011_0045.jpg",
        "SM7_뉴아트_2008_2011_0053.jpg", "SM7_뉴아트_2008_2011_0077.jpg", "라브4_5세대_2019_2024_0082.jpg",
        "라브4_5세대_2019_2024_0030.jpg", "라브4_5세대_2019_2024_0031.jpg", "라브4_5세대_2019_2024_0055.jpg",
        "라브4_5세대_2019_2024_0047.jpg", "라브4_5세대_2019_2024_0070.jpg", "라브4_5세대_2019_2024_0075.jpg",
        "라브4_5세대_2019_2024_0011.jpg", "라브4_5세대_2019_2024_0058.jpg", "라브4_5세대_2019_2024_0029.jpg",
        "라브4_5세대_2019_2024_0013.jpg", "Q30_2017_2019_0075.jpg", "Q30_2017_2019_0074.jpg",
        "베뉴_2020_2024_0005.jpg", "4시리즈_F32_2014_2020_0027.jpg", "익스플로러_2016_2017_0072.jpg",
        "마칸_2019_2021_0035.jpg", "마칸_2019_2021_0028.jpg", "마칸_2019_2021_0000.jpg",
        "마칸_2019_2021_0042.jpg", "마칸_2019_2021_0073.jpg", "마칸_2019_2021_0048.jpg",
        "마칸_2019_2021_0032.jpg", "더_뉴스포티지R_2014_2016_0083.jpg", "더_뉴스포티지R_2014_2016_0013.jpg",
        "더_뉴스포티지R_2014_2016_0000.jpg", "더_뉴스포티지R_2014_2016_0051.jpg", "더_뉴스포티지R_2014_2016_0076.jpg",
        "티볼리_에어_2016_2019_0047.jpg", "레니게이드_2019_2023_0041.jpg", "뉴_카이엔_2011_2018_0047.jpg",
        "뉴_카이엔_2011_2018_0048.jpg", "뉴_카이엔_2011_2018_0065.jpg", "뉴_카이엔_2011_2018_0049.jpg"
    ]

    # 전체 데이터셋 로드 (제외할 파일 목록 전달)
    full_dataset = FineGrainImageDataset(config.train_root, transform=None, is_test=False, exclude_files=exclude_file_names)
    print(f"총 이미지 수 (제외 파일 반영): {len(full_dataset.samples)}")

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

        # FineGrainImageDataset 인스턴스에 transform 적용 및 제외 파일 목록 전달
        # 주의: Subset을 만들 때, exclude_files는 FineGrainImageDataset 생성 시점에만 적용됩니다.
        # Subset은 이미 필터링된 전체 데이터셋에서 인덱스를 선택하므로, 여기에 exclude_files를 또 전달할 필요는 없습니다.
        model_train_dataset = Subset(
            FineGrainImageDataset(config.train_root, transform=train_transform, is_test=False, exclude_files=exclude_file_names),
            indices=train_indices
        )
        model_val_dataset = Subset(
            FineGrainImageDataset(config.train_root, transform=val_transform, is_test=False, exclude_files=exclude_file_names),
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
        best_val_logloss = float('inf') # logloss는 낮을수록 좋으므로 무한대로 초기화

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

            model_save_name = f"EfficientNet-b0_Model_{model_idx+1}_best_logloss.pth" # logloss 기준으로 저장
            model_save_path = os.path.join(config.model_save_dir, model_save_name)

            if val_logloss < best_val_logloss: # logloss가 낮아질수록 좋음
                no_improve_epochs = 0
                best_val_logloss = val_logloss
                torch.save(model.state_dict(), model_save_path)
                print(f"[체크포인트] 모델 {model_idx+1} 최고 LogLoss {best_val_logloss:.4f} 달성, 모델 저장.")
            else:
                no_improve_epochs += 1
                print(f"모델 {model_idx+1} LogLoss 개선 없음: {no_improve_epochs} epoch 연속.")

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

    # 전체 데이터셋으로 Combination Layer 학습 (제외 파일 목록 전달)
    full_train_dataset = FineGrainImageDataset(config.train_root, transform=train_transform, is_test=False, exclude_files=exclude_file_names)
    full_train_loader = DataLoader(full_train_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=os.cpu_count() // 2,
                                         pin_memory=True)

    full_val_dataset = FineGrainImageDataset(config.train_root, transform=val_transform, is_test=False, exclude_files=exclude_file_names)
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

        combination_save_path = os.path.join(config.model_save_dir, "Adaptive_Combination_Layer_best_logloss.pth") # logloss 기준으로 저장
        if val_logloss < best_val_logloss: # logloss가 낮아질수록 좋음
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