from torch import nn
import torchvision.models as models
import torch

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_v2_s(pretrained=True)
        self.shared_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.head_full = nn.Linear(backbone.classifier[1].in_features, num_classes)
        self.head_crop = nn.Linear(backbone.classifier[1].in_features, num_classes)

    def forward(self, x_full, x_crop):
        f_full = self.shared_extractor(x_full)
        f_crop = self.shared_extractor(x_crop)

        f_full = torch.flatten(f_full, 1)
        f_crop = torch.flatten(f_crop, 1)

        out_full = self.head_full(f_full)
        out_crop = self.head_crop(f_crop)

        return out_full, out_crop


