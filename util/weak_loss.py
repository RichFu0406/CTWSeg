"""Losses for sparse-label point cloud semantic segmentation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakSupervisionLoss(nn.Module):
    """Combine sparse supervision with confidence-filtered pseudo labels."""

    def __init__(
        self,
        classes,
        ignore_label=255,
        dice_weight=0.5,
        pseudo_weight=1.0,
        entropy_weight=0.01,
        pseudo_threshold=0.9,
        pseudo_temperature=1.0,
        pseudo_rampup_epochs=20,
    ):
        super().__init__()
        self.classes = classes
        self.ignore_label = ignore_label
        self.dice_weight = dice_weight
        self.pseudo_weight = pseudo_weight
        self.entropy_weight = entropy_weight
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_temperature = pseudo_temperature
        self.pseudo_rampup_epochs = pseudo_rampup_epochs

    def _rampup(self, epoch):
        if self.pseudo_rampup_epochs <= 0:
            return 1.0
        progress = min(max(epoch / self.pseudo_rampup_epochs, 0.0), 1.0)
        return math.exp(-5.0 * (1.0 - progress) ** 2)

    def _supervised_dice(self, logits, target, labeled_mask):
        if not labeled_mask.any():
            return logits.sum() * 0.0

        probabilities = torch.softmax(logits[labeled_mask], dim=1)
        one_hot = F.one_hot(target[labeled_mask], self.classes).float()
        intersection = (probabilities * one_hot).sum(dim=0)
        denominator = probabilities.sum(dim=0) + one_hot.sum(dim=0)
        present_classes = one_hot.sum(dim=0) > 0
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice[present_classes].mean()

    def _pseudo_label_loss(self, logits, teacher_logits, unlabeled_mask):
        zero = logits.sum() * 0.0
        if not unlabeled_mask.any():
            return zero, zero, 0.0

        teacher_probabilities = torch.softmax(
            teacher_logits.detach() / self.pseudo_temperature,
            dim=1,
        )
        confidence, pseudo_target = teacher_probabilities.max(dim=1)
        selected_mask = unlabeled_mask & (confidence >= self.pseudo_threshold)

        student_probabilities = torch.softmax(logits[unlabeled_mask], dim=1)
        entropy = -(
            student_probabilities
            * torch.log(student_probabilities.clamp_min(1e-8))
        ).sum(dim=1).mean() / math.log(self.classes)

        if not selected_mask.any():
            return zero, entropy, 0.0

        selected_target = pseudo_target[selected_mask]
        point_loss = F.cross_entropy(
            logits[selected_mask],
            selected_target,
            reduction='none',
        )

        class_count = torch.bincount(
            selected_target,
            minlength=self.classes,
        ).float()
        present_classes = class_count > 0
        class_weight = torch.zeros_like(class_count)
        class_weight[present_classes] = class_count[present_classes].rsqrt()
        class_weight[present_classes] /= class_weight[present_classes].mean()

        point_weight = confidence[selected_mask] * class_weight[selected_target]
        pseudo_loss = (point_loss * point_weight).sum() / point_weight.sum().clamp_min(1e-8)
        coverage = selected_mask.sum().item() / unlabeled_mask.sum().item()
        return pseudo_loss, entropy, coverage

    def forward(self, logits, target, teacher_logits, epoch):
        labeled_mask = target != self.ignore_label
        unlabeled_mask = ~labeled_mask
        zero = logits.sum() * 0.0

        if labeled_mask.any():
            supervised_ce = F.cross_entropy(
                logits[labeled_mask],
                target[labeled_mask],
            )
        else:
            supervised_ce = zero

        supervised_dice = self._supervised_dice(logits, target, labeled_mask)
        pseudo_loss, entropy_loss, pseudo_coverage = self._pseudo_label_loss(
            logits,
            teacher_logits,
            unlabeled_mask,
        )
        rampup = self._rampup(epoch)

        loss = (
            supervised_ce
            + self.dice_weight * supervised_dice
            + rampup
            * (
                self.pseudo_weight * pseudo_loss
                + self.entropy_weight * entropy_loss
            )
        )
        details = {
            'supervised_ce': supervised_ce.detach().item(),
            'supervised_dice': supervised_dice.detach().item(),
            'pseudo_loss': pseudo_loss.detach().item(),
            'entropy_loss': entropy_loss.detach().item(),
            'pseudo_coverage': pseudo_coverage,
            'pseudo_rampup': rampup,
        }
        return loss, details
