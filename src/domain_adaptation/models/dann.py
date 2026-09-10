"""Assembles the shared building blocks (feature_extractor.py, heads.py,
grl.py) into the two models this experiment trains and compares:

`FeatureClassifier` -- Phase 2's supervised baseline. Feature extractor +
    fault classifier, trained on labeled source data only, NO domain
    discriminator. Run on the target test set with zero adaptation to
    measure the domain-shift performance drop the plan asks for.

`DANNModel` -- Phase 3. Adds a domain discriminator behind a
    GradientReversalLayer, trained jointly with the classifier so the
    feature extractor is pushed toward domain-invariant features.

`healthy_anchor_loss()` -- the simplified stand-in for the plan's
Conditional-MMD term (see DOMAIN_ADAPTATION_PLAN.md's "Revisions"): pulls
the mean feature vector of labeled-healthy source examples toward the mean
feature vector of the (all-healthy, by construction) target-adaptation
batch, each mini-batch.
"""
from __future__ import annotations

import torch
from torch import nn

from domain_adaptation.config import Config
from domain_adaptation.models.feature_extractor import CNNFeatureExtractor
from domain_adaptation.models.grl import GradientReversalLayer
from domain_adaptation.models.heads import MLPHead

N_CLASSES = 3  # healthy, inner_race, outer_race -- see data/cwru_loader.UNIVERSAL_LABELS
N_DOMAINS = 3  # cwru, pu_artificial, pu_target -- see data/dataset.DOMAIN_IDS


class FeatureClassifier(nn.Module):
    """Feature extractor + fault classifier, no domain adaptation."""

    def __init__(self, input_shape: tuple[int, int], config: Config):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_shape, config.model.base_channels, config.model.feature_dim)
        self.classifier = MLPHead(config.model.feature_dim, config.model.classifier_hidden, N_CLASSES, config.model.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        return features, self.classifier(features)


class DANNModel(nn.Module):
    """Feature extractor + fault classifier + domain discriminator (behind
    a GRL). `set_grl_lambda()` is called once per training step from
    train_dann.py, following the warm-up schedule in models/grl.py.
    """

    def __init__(self, input_shape: tuple[int, int], config: Config):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_shape, config.model.base_channels, config.model.feature_dim)
        self.classifier = MLPHead(config.model.feature_dim, config.model.classifier_hidden, N_CLASSES, config.model.dropout)
        self.domain_discriminator = MLPHead(config.model.feature_dim, config.model.domain_hidden, N_DOMAINS, config.model.dropout)
        self.grl = GradientReversalLayer(lambda_=0.0)

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        domain_logits = self.domain_discriminator(self.grl(features))
        return features, class_logits, domain_logits


def healthy_anchor_loss(
    source_features: torch.Tensor, source_labels: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    """Squared distance between the mean feature vector of labeled-healthy
    source examples in this batch and the mean feature vector of the
    target-adaptation batch (all healthy, by construction of TARGET_ADAPT
    -- see data/dataset.py). Returns 0 (not NaN) if the batch happens to
    contain no healthy source examples, so an unlucky batch composition
    doesn't crash training.
    """
    healthy_mask = source_labels == 0
    if healthy_mask.sum() == 0:
        return torch.zeros((), device=source_features.device, dtype=source_features.dtype)
    source_healthy_mean = source_features[healthy_mask].mean(dim=0)
    target_mean = target_features.mean(dim=0)
    return ((source_healthy_mean - target_mean) ** 2).sum()
