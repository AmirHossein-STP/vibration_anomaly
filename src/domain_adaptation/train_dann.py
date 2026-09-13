"""Phase 3: DANN adaptation -- adds a domain discriminator (behind a
Gradient Reversal Layer) and a healthy-anchor alignment loss on top of
Phase 2's classifier, so the feature extractor is pushed toward features
that (a) still support fault classification on labeled source data and
(b) look domain-invariant across CWRU / PU_Artificial / PU_Target.

Per training step: one batch of labeled SOURCE_TRAIN (CWRU + PU_Artificial
pooled) + one batch of unlabeled TARGET_ADAPT (the held-out condition's
healthy runs ONLY -- no fault example, labeled or not, from the target
ever appears here, or anywhere in this script). "One epoch" = one full
pass over SOURCE_TRAIN, cycling TARGET_ADAPT (usually the smaller of the
two) to fill in batches -- see `_cycle()`.

Loss = classification(source) + domain_loss_weight * domain(source+target,
GRL) + anchor_loss_weight * healthy_anchor(source_healthy, target).

Loads the normalizer train_baseline.py already fit and saved (NOT
re-fitting one here) so baseline and DANN see identically-normalized
inputs -- run train_baseline.py first.

Usage:
    python -m domain_adaptation.train_dann --config configs/domain_adaptation.yaml [--pu-root DIR]
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape

from domain_adaptation.config import Config, load_config
from domain_adaptation.data.dataset import build_domain_adaptation_datasets
from domain_adaptation.data.pu_loader import find_pu_root
from domain_adaptation.data.torch_dataset import LabeledDomainDataset, UnlabeledDomainDataset
from domain_adaptation.models.dann import N_DOMAINS, DANNModel, healthy_anchor_loss
from domain_adaptation.models.feature_extractor import select_device
from domain_adaptation.models.grl import grl_lambda_schedule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def run_train_epoch(
    model, source_loader, target_loader, optimizer, device,
    domain_weight, anchor_weight, epoch, total_epochs, grl_gamma, grad_clip_norm,
) -> dict:
    """One pass over SOURCE_TRAIN, cycling TARGET_ADAPT alongside it.

    The GRL lambda is updated PER ITERATION, not once per epoch (this is
    the original DANN paper's formulation, and v1 got it wrong -- see
    DOMAIN_ADAPTATION_PLAN.md's kernel v1/v2 status). It matters more than
    it looks: with a fast ramp (high grl_gamma), a per-epoch lambda would
    slam the discriminator with strong reversed gradients from the very
    first batch, before it has learned anything worth reversing --
    scrambling it into producing meaningless gradients. Per-iteration,
    lambda still starts at ~0 during the first batches of epoch 1, so the
    discriminator gets a clean head start, then the reversal ramps up
    smoothly within the epoch.
    """
    model.train(True)
    cls_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()
    target_iter = _cycle(target_loader)
    n_batches = len(source_loader)

    totals = {
        "cls_loss": 0.0, "domain_loss": 0.0, "anchor_loss": 0.0,
        "correct": 0, "n": 0, "lambda_sum": 0.0,
    }
    # Per-domain tallies, for a BALANCED domain accuracy. Raw accuracy is
    # misleading here: every batch is 64 source + 64 target, so pu_target is
    # 50% of it by construction and a discriminator that blindly answers
    # "target" scores 0.50 -- which is exactly what v2's 0.487-0.524 was.
    # Macro-averaging recall over the 3 domains makes chance a true 1/3.
    domain_correct = torch.zeros(N_DOMAINS)
    domain_total = torch.zeros(N_DOMAINS)
    for batch_idx, (X_s, domain_s, label_s) in enumerate(source_loader):
        # Global training progress in [0, 1], counted in ITERATIONS.
        progress = ((epoch - 1) * n_batches + batch_idx) / max(total_epochs * n_batches - 1, 1)
        lambda_ = grl_lambda_schedule(progress, grl_gamma)
        model.set_grl_lambda(lambda_)

        X_t, domain_t = next(target_iter)
        X_s, domain_s, label_s = X_s.to(device), domain_s.to(device), label_s.to(device)
        X_t, domain_t = X_t.to(device), domain_t.to(device)

        features_s, class_logits, domain_logits_s = model(X_s)
        features_t, _class_logits_t, domain_logits_t = model(X_t)

        cls_loss = cls_criterion(class_logits, label_s)
        domain_logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
        domain_targets = torch.cat([domain_s, domain_t], dim=0)
        domain_loss = domain_criterion(domain_logits, domain_targets)
        anchor_loss = healthy_anchor_loss(features_s, label_s, features_t)

        loss = cls_loss + domain_weight * domain_loss + anchor_weight * anchor_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        n = len(X_s)
        totals["cls_loss"] += cls_loss.item() * n
        totals["domain_loss"] += domain_loss.item() * n
        totals["anchor_loss"] += anchor_loss.item() * n
        totals["correct"] += (class_logits.argmax(dim=1) == label_s).sum().item()
        # Balanced domain accuracy (see domain_correct/domain_total above):
        # ~0.33 means the discriminator genuinely can't tell the domains
        # apart; higher means it still can. Much easier to read live than a
        # cross-entropy hovering near ln(3)=1.099 -- and, unlike raw
        # accuracy, not inflated by the batch's 50% target share.
        domain_preds = domain_logits.argmax(dim=1).cpu()
        domain_truth = domain_targets.cpu()
        for d in range(N_DOMAINS):
            mask = domain_truth == d
            domain_total[d] += mask.sum()
            domain_correct[d] += (domain_preds[mask] == d).sum()
        totals["lambda_sum"] += lambda_ * n
        totals["n"] += n

    n = totals.pop("n")
    seen = domain_total > 0
    balanced_domain_acc = float((domain_correct[seen] / domain_total[seen]).mean())
    return {
        "cls_loss": totals["cls_loss"] / n,
        "domain_loss": totals["domain_loss"] / n,
        "anchor_loss": totals["anchor_loss"] / n,
        "cls_acc": totals["correct"] / n,
        "domain_acc": balanced_domain_acc,
        "lambda": totals["lambda_sum"] / n,
    }


@torch.no_grad()
def run_val_epoch(model, val_loader, device) -> tuple[float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0, 0
    for X, _domain_id, label in val_loader:
        X, label = X.to(device), label.to(device)
        _features, class_logits, _domain_logits = model(X)
        loss = criterion(class_logits, label)
        total_loss += loss.item() * len(X)
        total_correct += (class_logits.argmax(dim=1) == label).sum().item()
        n += len(X)
    return total_loss / n, total_correct / n


def train(config: Config, pu_root, force_data: bool = False) -> None:
    set_seed(config.seed)

    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    normalizer_path = models_dir / "normalizer.npz"
    if not normalizer_path.exists():
        raise FileNotFoundError(
            f"{normalizer_path} not found -- run `python -m domain_adaptation.train_baseline` first "
            "(it fits and saves the normalizer both models share)."
        )
    normalizer = SpectrogramNormalizer.load(normalizer_path)

    print("Building datasets (cached after first run)...")
    datasets = build_domain_adaptation_datasets(config, pu_root, force=force_data)
    train_X, train_meta = datasets["source_train"]
    val_X, val_meta = datasets["source_val"]
    adapt_X, adapt_meta = datasets["target_adapt"]
    print(f"  source_train: {train_X.shape}  source_val: {val_X.shape}  target_adapt: {adapt_X.shape}")

    expected = expected_shape(config.common, config.stft)
    train_X_norm = normalizer.transform(train_X)
    val_X_norm = normalizer.transform(val_X)
    adapt_X_norm = normalizer.transform(adapt_X)

    device = select_device(config.train_dann.device)
    print(f"Using device: {device}")

    source_loader = DataLoader(
        LabeledDomainDataset(train_X_norm, train_meta.domain_id.to_numpy(), train_meta.universal_label.to_numpy()),
        batch_size=config.train_dann.batch_size,
        shuffle=True,
        drop_last=True,  # keeps every source batch full-size, so the paired target batch size is meaningful too
    )
    target_loader = DataLoader(
        UnlabeledDomainDataset(adapt_X_norm, adapt_meta.domain_id.to_numpy()),
        batch_size=config.train_dann.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        LabeledDomainDataset(val_X_norm, val_meta.domain_id.to_numpy(), val_meta.universal_label.to_numpy()),
        batch_size=config.train_dann.batch_size,
        shuffle=False,
    )

    model = DANNModel(expected, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # The domain discriminator gets its own parameter group so its learning
    # rate can be raised independently (config's discriminator_lr_multiplier,
    # default 1.0 = no change). Why the knob exists: in the minimax game the
    # discriminator has to stay one step AHEAD of the feature extractor to
    # produce meaningful reversed gradients. If domain_acc sits at chance
    # (~1/3) while the post-hoc probe in evaluate.py still separates the
    # domains easily, that means the in-training discriminator is simply
    # losing the race -- not that the features are aligned -- and this is the
    # dial to turn. See DOMAIN_ADAPTATION_PLAN.md's kernel v2 notes.
    base_lr = config.train_dann.learning_rate
    discriminator_lr = base_lr * config.train_dann.discriminator_lr_multiplier
    discriminator_params = list(model.domain_discriminator.parameters())
    discriminator_param_ids = {id(p) for p in discriminator_params}
    backbone_params = [p for p in model.parameters() if id(p) not in discriminator_param_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": discriminator_params, "lr": discriminator_lr},
        ],
        weight_decay=config.train_dann.weight_decay,
    )
    if config.train_dann.discriminator_lr_multiplier != 1.0:
        print(f"Discriminator LR: {discriminator_lr:.2e} ({config.train_dann.discriminator_lr_multiplier}x base {base_lr:.2e})")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_ckpt_path = models_dir / "dann_best.pt"
    total_epochs = config.train_dann.epochs

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        # NB: lambda is set per ITERATION inside run_train_epoch (v1 set it
        # once per epoch here -- see that function's docstring for why the
        # difference matters once grl_gamma is raised).
        train_metrics = run_train_epoch(
            model, source_loader, target_loader, optimizer, device,
            config.train_dann.domain_loss_weight, config.train_dann.anchor_loss_weight,
            epoch, total_epochs, config.train_dann.grl_gamma, config.train_dann.grad_clip_norm,
        )
        val_loss, val_acc = run_val_epoch(model, val_loader, device)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}/{total_epochs}  lambda~{train_metrics['lambda']:.3f}  "
            f"cls_loss={train_metrics['cls_loss']:.4f} cls_acc={train_metrics['cls_acc']:.3f}  "
            f"domain_loss={train_metrics['domain_loss']:.4f} domain_acc={train_metrics['domain_acc']:.3f} "
            f"anchor_loss={train_metrics['anchor_loss']:.5f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({dt:.1f}s){marker}"
        )
        history.append({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc, **train_metrics})

        if epochs_without_improvement >= config.train_dann.early_stopping_patience:
            print(f"Early stopping: no val improvement in {config.train_dann.early_stopping_patience} epochs.")
            break

    pd.DataFrame(history).to_csv(reports_dir / "dann_training_log.csv", index=False)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print("Next: python -m domain_adaptation.evaluate --config <same config>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/domain_adaptation.yaml")
    parser.add_argument("--pu-root", default=None, help="Root dir of the PU dataset (default: auto-detect /kaggle/input, else config's pu_root_local).")
    parser.add_argument("--force-data", action="store_true", help="Recompute cached spectrograms even if present.")
    args = parser.parse_args()

    config = load_config(args.config)
    pu_root = args.pu_root or find_pu_root(config.resolve(config.paths.pu_root_local))
    train(config, pu_root=pu_root, force_data=args.force_data)


if __name__ == "__main__":
    main()
