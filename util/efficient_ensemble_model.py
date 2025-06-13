# util/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm # EfficientNet, ViT를 위해 timm 라이브러리 사용


# ===============================================================
# LoRA Linear Layer
# ===============================================================
class LoRALinear(nn.Module):
    """
    nn.Linear 레이어에 Low-Rank 어댑터(A, B)를 붙인 형태.
    output = W x + (alpha/r) * B @ (A x)
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r            = r
        self.alpha        = alpha
        self.scaling      = alpha / r

        # (1) 원본 weight, bias: 학습하지 않도록 고정
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # (2) LoRA 어댑터용 A, B: 학습 가능
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01) # A는 작은 값으로 초기화 (0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))       # B는 0으로 초기화

        # (3) 원본 weight/bias 랜덤 초기화 (Xavier)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        out_orig = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(x, self.B @ self.A, bias=None) * self.scaling
        return out_orig + lora_out


# ===============================================================
# 기본 모델들 (Fine-Grained Classification용)
# ===============================================================

class FineGrainResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.classifier(f)


class FineGrainResNext50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone = models.resnext50_32x4d(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.classifier(f)


class FineGrainConvNext(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone = models.convnext_large(pretrained=pretrained)
        
        feat_dim = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ===============================================================
# LoRA 적용 기본 모델들 (추가 및 수정)
# ===============================================================

class FineGrainResNet50LoRA(nn.Module):
    def __init__(self, num_classes, pretrained=True, lora_r=8, lora_alpha=16.0):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained) 

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features 

        self.classifier = LoRALinear(
            in_features=feat_dim,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        # 백본의 모든 파라미터 고정
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.classifier(f)


class FineGrainResNext50LoRA(nn.Module):
    def __init__(self, num_classes, pretrained=True, lora_r=8, lora_alpha=16.0):
        super().__init__()
        backbone = models.resnext50_32x4d(pretrained=pretrained) 

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features 

        self.classifier = LoRALinear(
            in_features=feat_dim,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        # 백본의 모든 파라미터 고정
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.classifier(f)


class FineGrainConvNextLoRA(nn.Module): # 수정된 부분
    def __init__(self, num_classes, pretrained=True, lora_r=8, lora_alpha=16.0, model_name='convnext_large'):
        super().__init__()
        backbone = models.get_model(model_name, pretrained=pretrained)
        
        # ConvNeXt의 classifier는 (LayerNorm, Flatten, Linear)로 구성됨
        # feat_dim은 원래 Linear 층의 in_features
        feat_dim = backbone.classifier[2].in_features 
        
        # 특징 추출기 부분 (avgpool까지)
        self.features_extractor = nn.Sequential(
            backbone.features, # 주요 컨볼루션 블록들
            backbone.avgpool   # Adaptive average pooling
        )
        
        # 사전 분류기 레이어 (LayerNorm과 Flatten)
        self.pre_classifier = nn.Sequential(
            backbone.classifier[0], # LayerNorm
            backbone.classifier[1]  # Flatten
        )

        # LoRA가 적용된 최종 분류기
        self.classifier_lora = LoRALinear(
            in_features=feat_dim,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        # self.classifier_lora를 제외한 모든 파라미터 고정
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        for param in self.pre_classifier.parameters():
            param.requires_grad = False
        # LoRALinear의 파라미터는 기본적으로 requires_grad=True입니다.

    def forward(self, x):
        features = self.features_extractor(x)  # features와 avgpool 통과
        features = self.pre_classifier(features) # LayerNorm과 Flatten 통과
        logits = self.classifier_lora(features) # LoRA 분류기 통과
        return logits


class FineGrainEfficientNetLoRA(nn.Module):
    def __init__(self, num_classes, pretrained=True, lora_r=8, lora_alpha=16.0, model_name='efficientnet_b0'):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)

        feat_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.classifier = LoRALinear(
            in_features=feat_dim,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f)


class FineGrainViTLoRA(nn.Module):
    def __init__(self, num_classes, pretrained=True, lora_r=8, lora_alpha=16.0, model_name='vit_base_patch16_224'):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, dynamic_img_size=True)

        feat_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone

        self.classifier = LoRALinear(
            in_features=feat_dim,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        
        if len(features.shape) == 3:
            features = features[:, 0]
        
        return self.classifier(features)


# ===============================================================
# 앙상블 모델 (조합 레이어)
# ===============================================================

class EnsembleModel(nn.Module):
    def __init__(self, num_ensemble_models, num_classes):
        super().__init__()
        self.num_ensemble_models = num_ensemble_models
        self.num_classes = num_classes

        self.combination_layer = nn.Sequential(
            nn.Linear(num_ensemble_models * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, ensemble_outputs):
        combined_features = torch.cat(ensemble_outputs, dim=1)
        final_logits = self.combination_layer(combined_features)
        return final_logits


# ===============================================================
# 헬퍼 함수
# ===============================================================

def count_trainable_parameters(model):
    """
    주어진 PyTorch 모델의 학습 가능한 파라미터 수를 계산합니다.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_total_parameters(model):
    """
    주어진 PyTorch 모델의 총 파라미터 수를 계산합니다 (학습 가능/불가능 모두 포함).
    """
    return sum(p.numel() for p in model.parameters())