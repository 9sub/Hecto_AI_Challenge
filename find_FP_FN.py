import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from util.dataloader import ImageDataset
from util.model import EfficientNetB0
import config
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Validation 데이터셋 및 로더 준비
val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
])

dataset = ImageDataset(config.train_root, transform=None)
target = [label for _, label in dataset.samples]
train_index, val_index = train_test_split(range(len(target)), test_size=0.2, stratify=target, random_state=config.seed)
val_dataset = Subset(ImageDataset(config.train_root, transform=val_transform), indices=val_index)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# 2) 모델 로드
model = EfficientNetB0(num_classes=len(dataset.classes))
model.load_state_dict(torch.load('./models/efficientnet_base_512_basemodel_9epoch.pth', map_location=device))  # 저장된 best model 경로로 변경
model.to(device)
model.eval()

# 3) 클래스명 리스트
class_names = dataset.classes

# 4) 오분류 리스트 초기화
misclassified = []

# 5) 오분류 수집 및 출력
with torch.no_grad():
    for idx, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        incorrect_mask = (preds != labels).cpu().numpy()
        for i, incorrect in enumerate(incorrect_mask):
            if incorrect:
                # val_loader 내 인덱스 -> dataset 내 인덱스 계산
                dataset_idx = val_index[idx * config.batch_size + i]
                img_path, true_label = dataset.samples[dataset_idx]

                misclassified.append({
                    'image_path': img_path,
                    'true_label': class_names[true_label],
                    'pred_label': class_names[preds[i].item()]
                })

print(f"총 오분류 샘플 수: {len(misclassified)}")

# 6) CSV로 저장
df_mis = pd.DataFrame(misclassified)
df_mis.to_csv('misclassified_samples.csv', index=False, encoding='utf-8-sig')

# # 7) 상위 5개 오분류 샘플 시각화
# for item in misclassified[:5]:
#     img = Image.open(item['image_path']).convert('RGB')
#     plt.imshow(img)
#     plt.title(f"True: {item['true_label']} | Pred: {item['pred_label']}")
#     plt.axis('off')
#     plt.show()