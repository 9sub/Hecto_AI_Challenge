import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm

from util.model import FineGrainResNet50LoRA, MultiTaskResNet50LoRAContrastiveLoss
from util.dataloader import ImageDataset, FineGrainImageDataset
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 1) 데이터셋 및 DataLoader 준비
# ------------------------------
# 원본 전처리: Resize → ToTensor → Normalize
val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 이미지 데이터셋 생성 (is_test=True → 레이블 없이 이미지만 반환)
orig_test_dataset = ImageDataset(config.test_root,
                                 transform=val_transform,
                                 is_test=True)

orig_test_loader = DataLoader(orig_test_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=os.cpu_count() // 2,
                              pin_memory=True)

# 클래스 이름 리스트는 FineGrainImageDataset을 통해 얻음
train_dataset_full = FineGrainImageDataset(config.train_root, transform=None)
class_names = train_dataset_full.classes
num_classes = len(class_names)

# --------------------------------------------------
# 2) Fold별 체크포인트 파일 경로 리스트 정의
# --------------------------------------------------
checkpoint_dir  = "./models"
n_splits = 5
fold_ckpt_paths = [
    os.path.join(checkpoint_dir,
                 f"FineGrainResNet50_512_5fold_Contrastive Loss_LoRA_fold_{i+1}_best_acc.pth")
    for i in range(n_splits)  # 예: config.n_splits == 5
]

# 파일 유무 확인
for path in fold_ckpt_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint 파일이 없습니다: {path}")

# ------------------------------
# 3) Softmax 확률 누적용 배열 초기화
# ------------------------------
N = len(orig_test_dataset)  # 테스트 이미지 개수
C = num_classes  # 클래스 개수
sum_probs = np.zeros((N, C), dtype=np.float32)

# ------------------------------
# 4) Fold별 모델 로드 및 Inference
# ------------------------------
for fold_idx, ckpt_path in enumerate(fold_ckpt_paths):
    print(f"[Fold {fold_idx+1}] 체크포인트 로딩: {ckpt_path}")

    # (1) 모델 초기화 및 weight 로드
    model = MultiTaskResNet50LoRAContrastiveLoss(num_classes=C, pretrained=False).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # (2) 각 배치마다 확률을 평균 낸 뒤 누적
    start_idx = 0
    with torch.no_grad():
        for orig_batch in tqdm(orig_test_loader,
                               total=len(orig_test_loader),
                               desc=f"Fold {fold_idx+1} Inference"):

            orig_images = orig_batch.to(device, non_blocking=True)  # [B, C, H, W]
            B = orig_images.size(0)  # 배치 크기

            # (a) 원본 이미지 예측
            #logits_orig = model(orig_images)  # [B, C]
            _, logits_orig = model(orig_images)  # [B, C]
            probs_orig = F.softmax(logits_orig, dim=1)  # [B, C]

            # (b) 확률을 numpy로 변환
            probs_np = probs_orig.cpu().numpy()  # numpy [B, C]

            # (c) sum_probs에 누적
            sum_probs[start_idx:start_idx + B] += probs_np
            start_idx += B

    # 모델 삭제 및 GPU 메모리 비우기
    del model
    torch.cuda.empty_cache()

print("▶ 모든 Fold 확률 누적 완료.")

# ------------------------------
# 5) 평균 확률 계산 및 최종 예측
# ------------------------------
num_folds = len(fold_ckpt_paths)
avg_probs = sum_probs / num_folds  # (N, C)
pred_indices = np.argmax(avg_probs, axis=1)
pred_labels = [class_names[idx] for idx in pred_indices]

# ------------------------------
# 6) Submission 파일 생성
# ------------------------------
submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
class_columns = submission.columns[1:]  # 'ID' 다음부터 클래스명

# avg_probs를 DataFrame으로 만들어 submission 형식에 맞추기
df_probs = pd.DataFrame(avg_probs, columns=class_columns)
submission[class_columns] = df_probs.values

output_path = './output/FineGrainResNet50_512_5fold_contrastive_loss.csv'
submission.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"▶ 최종 결과 저장 완료: {output_path}")