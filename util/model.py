from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import timm
import math

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
    

class SwinArcClassifier(nn.Module):
    """
    Swin-Transformer 백본 + ArcFace 분류 헤드
    — 차종 분류에서 클래스 간 경계가 뚜렷해져, 일반 FC 헤드 대비 성능 향상 기대
    """
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        emb_dim:   int  = 512,
        margin:    float= 0.3,
        scale:     float= 30.0,
    ):
        super().__init__()
        # 1) Swin 백본 로드 (분류 헤드 없이 feature만)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,           # 헤드 제거
            global_pool='avg'        # [B, C] 로 pooling
        )
        feat_dim = self.backbone.num_features  # Swin의 출력 채널 수

        # 2) 임베딩 프로젝션
        self.embedding = nn.Linear(feat_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight)

        # 3) ArcFace 분류기
        self.class_weight = nn.Parameter(torch.Tensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.class_weight)

        self.margin = margin
        self.scale  = scale

    def forward(self, x, labels=None):
        # 1) backbone → 특징 벡터 [B, feat_dim]
        feat = self.backbone(x)

        # 2) embedding 레이어로 차원 축소 → [B, emb_dim]
        emb = F.normalize(self.embedding(feat), dim=1)

        # 3) 분류용 weight 정규화
        W = F.normalize(self.class_weight, dim=1)  # [num_classes, emb_dim]

        # 4) ArcFace logits 계산
        #    cosθ = emb · W^T
        cos_t = emb @ W.t()                          # [B, num_classes]
        if labels is None:
            # inference 시에는 scale된 cos만 리턴
            return cos_t * self.scale

        #  add angular margin: cos(θ + m) = cosθ·cos m − sinθ·sin m
        theta     = torch.acos(cos_t.clamp(-1.0+1e-7, 1.0-1e-7))
        cos_t_m   = torch.cos(theta + self.margin)
        one_hot   = F.one_hot(labels, num_classes=W.size(0)).to(cos_t.dtype)
        logits    = torch.where(one_hot.bool(), cos_t_m, cos_t)

        # 5) 최종 scale
        return logits * self.scale, emb


    
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



class BilinearResDense(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # ResNet50 백본 (마지막 conv layer까지)
        resnet = models.resnet50(pretrained=pretrained)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, H, W]

        # DenseNet121 백본 (마지막 conv layer까지)
        densenet = models.densenet121(pretrained=pretrained)
        self.densenet_features = densenet.features  # [B, 1024, H, W]

        # ResNet과 DenseNet 출력 채널 수
        self.resnet_feat_dim = 2048
        self.densenet_feat_dim = 1024

        # 분류기: (ResNet 채널 수 * DenseNet 채널 수)
        self.classifier = nn.Linear(self.resnet_feat_dim * self.densenet_feat_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # ResNet 특징 맵
        res_feat = self.resnet_features(x)          # (B, 2048, H, W)
        # DenseNet 특징 맵
        dense_feat = self.densenet_features(x)      # (B, 1024, H, W)

        # 크기 맞추기: 두 특징 맵의 공간 크기(H, W)가 다를 수 있음
        # 가장 간단하게는 AdaptiveAvgPool2d 로 동일 크기로 맞춤
        H, W = 7, 7  # 일반적으로 ResNet50 마지막 conv 특징맵 크기
        res_feat = F.adaptive_avg_pool2d(res_feat, (H, W))
        dense_feat = F.adaptive_avg_pool2d(dense_feat, (H, W))

        # flatten 공간 차원
        res_feat = res_feat.view(B, self.resnet_feat_dim, H*W)      # (B, 2048, N)
        dense_feat = dense_feat.view(B, self.densenet_feat_dim, H*W)  # (B, 1024, N)

        # bilinear pooling (batch matrix multiply)
        bilinear = torch.bmm(res_feat, dense_feat.transpose(1, 2))  # (B, 2048, 1024)

        # flatten 채널 차원
        bilinear = bilinear.view(B, -1)  # (B, 2048*1024)

        # signed square root normalization
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-5)

        # L2 normalization
        bilinear = F.normalize(bilinear)

        # 분류기 통과
        out = self.classifier(bilinear)  # (B, num_classes)

        return out


# ─── ArcFace 헤드 ─────────────────────────────────────────
class ArcMarginProduct(nn.Module):
    """ArcFace head"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s, self.m = s, m
        # math 함수를 사용해 float에 대해 미리 계산
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, W)  # [B, C]
        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi  = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(cosine.dtype)
        logits  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s



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