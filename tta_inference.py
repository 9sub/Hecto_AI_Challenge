import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset # Dataset 추가
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from PIL import Image # PIL 추가
import os # os 추가

# util.model, util.dataloader, config 등은 기존과 동일하게 사용한다고 가정
from util.model import FineGrainConvNext # 예시 모델
from util.dataloader import FineGrainImageDataset, TTADataset
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. TTA용 변환(Transforms) 정의 ---
# 기본 val_transform (원본 이미지용 또는 중앙 크롭 등)
base_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 추가 TTA 변환 예시 (수평 뒤집기)
tta_hflip_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(p=1.0), # 항상 뒤집기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 추가 TTA 변환 예시 (ColorJitter)
tta_color_jitter_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 약간의 색상 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 사용할 TTA 변환 리스트
tta_transforms_list = [
    base_transform,          # 원본 (또는 기본 변환)
    tta_hflip_transform,     # 수평 뒤집기
    # tta_color_jitter_transform, # 추가 변환 (필요에 따라)
]

# --- Custom Collate Function ---
def custom_tta_collate_fn(batch):
    """
    Custom collate function for TTA.
    It takes a batch of (PIL_Image, image_path) tuples and returns a list of PIL_Images and a list of image_paths.
    Transforms are applied inside the inference loop.
    """
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

# 테스트 데이터셋 (TTA용, 변환은 루프 내에서 적용)
test_dataset_tta = TTADataset(config.test_root, is_test=True)
# DataLoader에 custom_tta_collate_fn 지정
test_loader_tta = DataLoader(test_dataset_tta, batch_size=1, shuffle=False, collate_fn=custom_tta_collate_fn)


# 클래스 이름 로드 (기존 방식 유지)
train_dataset_full = FineGrainImageDataset(config.train_root, transform=None)
class_name = train_dataset_full.classes

# 저장된 모델 로드 (기존 방식 유지)
model = FineGrainConvNext(num_classes=len(class_name), pretrained=False)
model_path = './models/FineGrainConvNext_224_FocalLoss_basemodel.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 추론
results = []
image_ids_ordered = [] # 파일 순서 유지를 위해 ID 저장

with torch.no_grad():
    for pil_images, image_paths in tqdm(test_loader_tta):
        # pil_images는 PIL Image 객체의 리스트 (길이는 배치 크기)
        for i in range(len(pil_images)):
            current_pil_image = pil_images[i]
            image_ids_ordered.append(os.path.basename(image_paths[i])) # 파일명 (ID) 저장

            batch_tta_probs = [] # 현재 이미지의 TTA 버전들로부터 얻은 확률 저장
            for tta_transform_item in tta_transforms_list:
                # Apply the current TTA transform to the PIL image
                # This handles Resize, ToTensor, and Normalize for each TTA variant
                augmented_tensor = tta_transform_item(current_pil_image)
                input_tensor = augmented_tensor.unsqueeze(0).to(device) # Add batch dimension
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                batch_tta_probs.append(probs.cpu()) # (1, num_classes) shaped tensor

            # Average all TTA prediction probabilities for the current image
            if batch_tta_probs:
                aggregated_probs_tensor = torch.stack(batch_tta_probs).mean(dim=0) # (1, num_classes)

                # Convert to dictionary for results
                prob_dict = {
                    class_name[j]: aggregated_probs_tensor[0, j].item()
                    for j in range(len(class_name))
                }
                results.append(prob_dict)

# Create DataFrame (order is preserved)
pred = pd.DataFrame(results)

# Load submission template
submission_template = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')

# Order class columns as in submission template
class_columns_ordered = submission_template.columns[1:]
pred_ordered = pred.reindex(columns=class_columns_ordered, fill_value=0.0)

# Add ID column to pred_ordered and reindex to match submission template
pred_ordered['ID'] = [os.path.splitext(name)[0] for name in image_ids_ordered]
pred_ordered = pred_ordered.set_index('ID')
pred_ordered = pred_ordered.reindex(index=submission_template['ID'].astype(str)).reset_index()

submission_final = submission_template.copy()
submission_final[class_columns_ordered] = pred_ordered[class_columns_ordered].values
submission_final.to_csv('./output/FineGrainConvNext_224_FocalLoss_TTA.csv', index=False, encoding='utf-8-sig')

print("TTA를 적용한 추론 완료 및 CSV 저장.")