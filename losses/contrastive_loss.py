import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ContrastiveLoss(object):
    def __init__(self, device, temperature):

        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.to(device=self.device, non_blocking=True).float()

    def __call__(self, view1_features, view2_features):

        view1_features = nn.functional.normalize(view1_features, dim=1)
        view2_features = nn.functional.normalize(view2_features, dim=1)

        # Inter-modality alignment
        logits_per_view1 = view1_features @ view2_features.t()    # M*K^T = I
        logits_per_view2 = view2_features @ view1_features.t()   # K*M^T = I

        # Intra-modality alignment
        logits_clstr_view1 = view1_features @ view1_features.t()  # M*M^T = 0
        logits_clstr_view2 = view2_features @ view2_features.t()    # K*K^T = 0

        logits_per_view1 /= self.temperature
        logits_per_view2 /= self.temperature
        logits_clstr_view1 /= self.temperature
        logits_clstr_view2 /= self.temperature

        positive_mask = self._get_positive_mask(view1_features.shape[0])
        negatives_view1 = logits_clstr_view1 * positive_mask
        negatives_view2 = logits_clstr_view2 * positive_mask

        view1_logits = torch.cat([logits_per_view1, negatives_view1], dim=1)
        view2_logits = torch.cat([logits_per_view2, negatives_view2], dim=1)

        diag = np.eye(view1_features.shape[0])
        mask_v11 = torch.from_numpy((diag)).to(device=self.device).float()
        mask_v22 = torch.from_numpy((diag)).to(device=self.device).float()

        mask_neg_v1 = torch.zeros_like(negatives_view1)
        mask_neg_v2 = torch.zeros_like(negatives_view2)
        mask_v1 = torch.cat([mask_v11, mask_neg_v1], dim=1)
        mask_v2 = torch.cat([mask_v22, mask_neg_v2], dim=1)

        loss_1 = self.compute_loss(view1_logits, mask_v1)
        loss_2 = self.compute_loss(view2_logits, mask_v2)

        return (((loss_1.mean() + loss_2.mean())) / 2)

