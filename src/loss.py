import torch
import torch.nn as nn
from collections import Counter
import numpy as np

# Class weights calculation
def calculate_class_weights(train_df, diagnosis_to_target):
    class_counter = Counter(train_df["target"])
    total_samples = sum(class_counter.values())
    inverse_weights = {cls: total_samples / (len(class_counter) * count) for cls, count in class_counter.items()}
    min_weight = min(inverse_weights.values())
    max_weight = max(inverse_weights.values())
    normalized_weights = {
        cls: 1.0 + (weight - min_weight) * (5.0 - 1.0) / (max_weight - min_weight)
        for cls, weight in inverse_weights.items()
    }
    adjusted_weights = normalized_weights.copy()
    adjusted_weights[diagnosis_to_target['MEL']] = 5.0  # MEL: High weight
    adjusted_weights[diagnosis_to_target['SCC']] = 4.5  # SCC: Prioritized
    adjusted_weights[diagnosis_to_target['DF']] = min(adjusted_weights[diagnosis_to_target['DF']], 4.0)
    adjusted_weights[diagnosis_to_target['VASC']] = min(adjusted_weights[diagnosis_to_target['VASC']], 4.0)
    adjusted_weights[diagnosis_to_target['NV']] = max(adjusted_weights[diagnosis_to_target['NV']], 1.2)
    adjusted_weights[diagnosis_to_target['UNK']] = 3.0
    class_weights_tensor = torch.tensor([adjusted_weights[cls] for cls in sorted(class_counter.keys())], dtype=torch.float32)
    return class_weights_tensor

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Move alpha to device if provided
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, min=1e-7, max=1-1e-7)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # per-sample loss if 'none'
