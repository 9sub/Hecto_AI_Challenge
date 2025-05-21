import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd

from util.model import ResNet152, MultiTaskModel
from util.dataloader import ImageDataset
import config

device = "mps"

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

test_dataset = ImageDataset(config.test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

dataset = ImageDataset(config.train_root, transform=None)
class_name = dataset.classes
# 저장된 모델 로드
model = MultiTaskModel(num_classes=len(class_name))
model.load_state_dict(torch.load('./models/Multitask efficientNet_basemodel.pth', map_location=device))
model.to(device)

# 추론
model.eval()
results = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        #outputs = model(images)
        outputs, _ = model(images, images)
        probs = F.softmax(outputs, dim=1)

        # 각 배치의 확률을 리스트로 변환
        for prob in probs.cpu():  # prob: (num_classes,)
            result = {
                class_name[i]: prob[i].item()
                for i in range(len(class_name))
            }
            results.append(result)
            
pred = pd.DataFrame(results)


submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')

# 'ID' 컬럼을 제외한 클래스 컬럼 정렬
class_columns = submission.columns[1:]

import unicodedata

def normalize_list(str_list):
    return [unicodedata.normalize('NFC', s) for s in str_list]

class_columns_norm = normalize_list(class_columns)
pred.columns = normalize_list(pred.columns)

pred = pred[class_columns_norm]


submission[class_columns_norm] = pred.values
submission.to_csv('./output/Multitask_efficientNet_10epoch_argmax.csv', index=False, encoding='utf-8-sig')