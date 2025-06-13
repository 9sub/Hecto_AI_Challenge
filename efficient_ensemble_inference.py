import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import config
from util.dataloader import FineGrainImageDataset, FineGrainChangeLabelDataset  # 수정된 데이터셋 사용
from efficient_ensemble_train import EfficientNetB0Model, SimpleCombinationLayer
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform 정의
val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 테스트 데이터셋 생성 (is_test=True로 설정)
test_dataset = FineGrainChangeLabelDataset(config.test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 클래스 정보 로드 (학습 데이터셋에서)
train_dataset = FineGrainChangeLabelDataset(config.train_root, transform=None, is_test=False)
class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Test samples: {len(test_dataset)}")

# 모델 로드
model1 = EfficientNetB0Model(num_classes=num_classes, pretrained=False).to(device)
model2 = EfficientNetB0Model(num_classes=num_classes, pretrained=False).to(device)
combination_layer = SimpleCombinationLayer(num_models=2, num_classes=num_classes).to(device)

model1.load_state_dict(torch.load('./models/EfficientNet-b0_Model_1_best_acc.pth', map_location=device))
model2.load_state_dict(torch.load('./models/EfficientNet-b0_Model_2_best_acc.pth', map_location=device))
combination_layer.load_state_dict(torch.load('./models/Adaptive_Combination_Layer_best.pth', map_location=device))

model1.eval()
model2.eval()
combination_layer.eval()

print("Models loaded successfully!")

# 추론
results = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        
        # 두 base 모델의 출력
        output1 = model1(images)
        output2 = model2(images)
        
        # Adaptive Combination Layer로 최종 출력
        final_output = combination_layer([output1, output2])
        probs = F.softmax(final_output, dim=1)

        for prob in probs.cpu():
            result = {class_names[i]: prob[i].item() for i in range(num_classes)}
            results.append(result)

print(f"Inference completed! Generated {len(results)} predictions.")

# Submission 파일 생성
pred = pd.DataFrame(results)
submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
class_columns = submission.columns[1:]

# 클래스 순서 맞추기
pred = pred.reindex(columns=class_columns, fill_value=0)

submission[class_columns] = pred.values
submission.to_csv('./output/AdaptiveEnsemble_EfficientNetB0.csv', index=False, encoding='utf-8-sig')

print("Submission file saved: ./output/AdaptiveEnsemble_EfficientNetB0.csv")
