import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        """
        alpha: 가중치 (스칼라 또는 클래스별 가중치 텐서)
        gamma: 감마 파라미터 (난이도 조절)
        logits: inputs가 raw logits인지 여부
        reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            # CrossEntropyLoss 내부에서 softmax 처리 전 logits 입력
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            # inputs가 확률 분포일 경우
            ce_loss = F.nll_loss(torch.log(inputs), targets, reduction='none')

        pt = torch.exp(-ce_loss)  # 정답 클래스 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Contrastive loss calculation
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) + 
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss