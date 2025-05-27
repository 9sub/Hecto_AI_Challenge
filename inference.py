import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from util.model import FineGrainResNet50

from util.model import ResNet152, MultiTaskModel, SwinArcClassifier
from util.dataloader import ImageDataset, FineGrainImageDataset
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

test_dataset = ImageDataset(config.test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

#dataset = ImageDataset(config.train_root, transform=None)

train_dataset_full = FineGrainImageDataset(config.train_root, transform=None)


class_name     = train_dataset_full.classes


#class_name = dataset.classes
# 저장된 모델 로드
#model = SwinArcClassifier(num_classes=len(class_name))

model= FineGrainResNet50(num_classes=len(class_name), pretrained=False).to(device)


model.load_state_dict(torch.load('./models/FineGrainResNet50_image512_basemodel.pth', map_location=device))
model.to(device)

# 추론
model.eval()
results = []

with torch.no_grad():
    for images in tqdm(test_loader):
        #images = images.to(device)
        images = images.to(device)
        #outputs = model(images)
        #out_model, out_year = model(images)
        outputs = model(images)
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
#class_columns = submission.columns[1:]

pred = pred.reindex(columns=class_columns, fill_value=0)


submission[class_columns] = pred.values
submission.to_csv('./output/FineGrainResNet50_image512.csv', index=False, encoding='utf-8-sig')