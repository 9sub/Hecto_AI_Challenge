import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import timm
from torch import nn
import config
from PIL import Image

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")
print(f"이미지 크기: {config.img_size}, 배치 크기: {config.batch_size}")

# ------------------------------
# 1) 모델 정의
# ------------------------------
class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class SimpleCombinationLayer(nn.Module):
    def __init__(self, num_models, num_classes):
        super().__init__()
        self.combination_layer = nn.Linear(num_models * num_classes, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, model_outputs):
        concatenated = torch.cat(model_outputs, dim=1)
        output = self.dropout(concatenated)
        return self.combination_layer(output)

# ------------------------------
# 2) Dataset 정의
# ------------------------------
class FineGrainChangeLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if is_test:
            self.samples = sorted([
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith('.jpg')
            ])
            return
        
        base_classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        combined = []
        for base in base_classes:
            for pose in sorted(os.listdir(os.path.join(root_dir, base))):
                path = os.path.join(root_dir, base, pose)
                if os.path.isdir(path):
                    combined.append(f"{base}_{pose}")

        self.classes = combined
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for base in base_classes:
            for pose in sorted(os.listdir(os.path.join(root_dir, base))):
                pose_path = os.path.join(root_dir, base, pose)
                if not os.path.isdir(pose_path):
                    continue
                idx = self.class_to_idx[f"{base}_{pose}"]
                for fname in os.listdir(pose_path):
                    if not fname.lower().endswith('.jpg'):
                        continue
                    self.samples.append((os.path.join(pose_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            path = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, path
        
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ------------------------------
# 3) DataLoader 준비
# ------------------------------
val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# fine-grained 클래스 정보 로드
train_dataset_for_classes = FineGrainChangeLabelDataset(config.train_root, transform=None, is_test=False)
fine_grained_classes = train_dataset_for_classes.classes
num_fine_classes = len(fine_grained_classes)
print(f"Fine-grained 클래스 수: {num_fine_classes}")
print(f"Fine-grained 클래스 예시: {fine_grained_classes[:5]}")

# 테스트 데이터셋 및 로더
test_dataset = FineGrainChangeLabelDataset(config.test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True
)
print(f"테스트 이미지 수: {len(test_dataset)}")

# ------------------------------
# 4) 체크포인트 경로
# ------------------------------
checkpoint_dir = getattr(config, 'model_save_dir', './models')
model_paths = [
    os.path.join(checkpoint_dir, "EfficientNet-b0_Model_1_best_acc.pth"),
    os.path.join(checkpoint_dir, "EfficientNet-b0_Model_2_best_acc.pth")
]
combination_path = os.path.join(checkpoint_dir, "Adaptive_Combination_Layer_best.pth")

for path in model_paths + [combination_path]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"체크포인트 파일이 존재하지 않습니다: {path}")

print("로드할 모델 체크포인트:")
for i, path in enumerate(model_paths, 1):
    print(f"  EfficientNet-b0 Model {i}: {path}")
print(f"  Combination Layer: {combination_path}")

# ------------------------------
# 5) 모델 로드
# ------------------------------
print("\n--- 모델 로드 시작 ---")
base_models = []
for i, mp in enumerate(model_paths, 1):
    print(f"EfficientNet-b0 Model {i} 로딩...")
    m = EfficientNetB0Model(num_classes=num_fine_classes, pretrained=False)
    m.load_state_dict(torch.load(mp, map_location=device))
    m.to(device).eval()
    base_models.append(m)

print("Combination Layer 로딩...")
combination_model = SimpleCombinationLayer(num_models=len(base_models), num_classes=num_fine_classes)
combination_model.load_state_dict(torch.load(combination_path, map_location=device))
combination_model.to(device).eval()
print("모델 로드 완료!")

# ------------------------------
# 6) 추론 수행
# ------------------------------
print("\n--- 추론 시작 ---")
all_image_paths = []
all_fine_probabilities = []

with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="추론 중"):
        images = images.to(device, non_blocking=True)
        outputs = []
        for m in base_models:
            with torch.cuda.amp.autocast():
                outputs.append(m(images))
        with torch.cuda.amp.autocast():
            final_logits = combination_model(outputs)
        probs = F.softmax(final_logits, dim=1)
        for i in range(probs.size(0)):
            all_image_paths.append(paths[i])
            all_fine_probabilities.append(probs[i].cpu().numpy())
print("추론 완료!")

# ------------------------------
# 7) 결과 저장 (수정된 부분)
# ------------------------------
print("\n--- 결과 저장 시작 ---")
# 이미지 ID 추출
image_ids = [os.path.basename(p).split('.')[0] for p in all_image_paths]

# 샘플 제출 파일에서 헤더 순서 불러오기
sample_df = pd.read_csv('./data/sample_submission.csv')
submission_columns = sample_df.columns.tolist()
print("샘플 제출 헤더:", submission_columns)

# 제출용 데이터 구성 (숫자형 인덱스 매핑)
submission_data = []
for img_id, probs in zip(image_ids, all_fine_probabilities):
    row = {'ID': img_id}
    for idx, prob in enumerate(probs):
        col = str(idx)
        row[col] = prob
    submission_data.append(row)

submission_df = pd.DataFrame(submission_data)
# 부족한 컬럼(클래스)을 0.0으로 채우기
for c in submission_columns:
    if c not in submission_df.columns:
        submission_df[c] = 0.0

# 컬럼 순서 맞추기
submission_df = submission_df[submission_columns]

# 파일 저장
output_path = './output/final_submission.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"▶ 제출 파일 저장 완료: {output_path}")

# ------------------------------
# 8) 예측 요약 (선택)
# ------------------------------
print("\n--- 예측 결과 요약 ---")
preds = [fine_grained_classes[np.argmax(p)] for p in all_fine_probabilities]
unique, counts = np.unique(preds, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"{cls}: {cnt}개 ({cnt/len(preds)*100:.1f}%)")

# ------------------------------
# 9) 메모리 정리
# ------------------------------
print("\n--- 메모리 정리 ---")
for i, m in enumerate(base_models, 1):
    m.cpu()
    del m
    print(f"  ✓ Base Model {i} 메모리 해제")
combination_model.cpu(); del combination_model
torch.cuda.empty_cache()
print("  ✓ GPU 캐시 정리 완료")
print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")