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