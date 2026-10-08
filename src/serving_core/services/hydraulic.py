"""Service `hydraulic` -- subproject 2: multi-rate sensor-fusion anomaly detection.

One cycle (60 s, 17 sensors at 3 native rates) = one `.npz` file whose arrays
are named after the sensors. One detector per monitored component, each
trained ONLY on cycles where that component is healthy (the study found that
"healthy on all four at once" leaves a handful of cycles).

What the study says about each component (also written into every result row
in `component_note`):
  cooler       ROC-AUC 1.00, but its score correlates with campaign time (r=0.88): treat as "promising, unproven"
  valve        ROC-AUC 0.89; only usable with a CALIBRATED threshold (the fixed 99th percentile gave recall 0)
  accumulator  ROC-AUC 0.87; same remark
  pump         ROC-AUC 0.66: EXPERIMENTAL, off by default (`--components pump` to opt in)

Therefore the threshold here is never a fixed percentile of training data: it
is set on a held-out healthy *calibration* block for an explicit target false
alarm rate (`calibration.target_false_alarm`), and when the training folder
also holds labelled degraded cycles the report measures ROC-AUC / recall on
them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hydraulic_fusion_ad.config import (
    Config, DataConfig, EvaluationConfig, ModelConfig, PathsConfig, SensorGroupConfig, SplitConfig, TrainConfig,
)
from hydraulic_fusion_ad.data.dataset import _block_shuffle_split
from hydraulic_fusion_ad.data.torch_dataset import MultiBranchDataset
from hydraulic_fusion_ad.features.normalize import MultiBranchNormalizer
from hydraulic_fusion_ad.models.fusion_autoencoder import (
    FusionAutoencoder, combined_mse, per_branch_mse, select_device,
)
from hydraulic_fusion_ad.train import run_epoch, set_seed
from serving_core.service import ConfigError, Context, Outcome, Service
from serving_core.signals import InputError
from serving_core.util import sha256_file

from serving_core.services.hydraulic_spec import (  # noqa: E402
    DEFAULT_COMPONENTS, GROUPS, HEALTHY, NOTES, SENSORS,
)


def load_cycle(path: Path) -> dict[str, np.ndarray]:
    """One cycle file -> {"slow": (8,60), "medium": (2,600), "fast": (7,6000)}, validated."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        raise InputError(f"cannot read file: {e}") from e
    if not path.suffix == ".npz":
        raise InputError("cycle files must be .npz with one array per sensor (PS1..PS6, EPS1, FS1, FS2, TS1..TS4, VS1, CE, CP, SE)")
    with z:
        missing = [s for s in SENSORS if s not in z.files]
        if missing:
            raise InputError(f"missing sensor array(s): {missing}")
        out = {}
        for g, (n, sensors) in GROUPS.items():
            rows = []
            for s in sensors:
                a = np.asarray(z[s]).squeeze().astype(np.float32)
                if a.shape != (n,):
                    raise InputError(f"sensor {s}: expected {n} samples (a 60 s cycle), got shape {a.shape}")
                if not np.isfinite(a).all():
                    raise InputError(f"sensor {s}: contains NaN/inf")
                rows.append(a)
            out[g] = np.stack(rows)
    return out


def make_config(cfg: dict, component: str) -> Config:
    m, t = cfg["model"], cfg["train"]
    return Config(
        seed=0,
        paths=PathsConfig("", "", "", "", ""),
        data=DataConfig(
            n_cycles_expected=0, profile_columns=list(HEALTHY), healthy_values=HEALTHY, valid_values={},
            sensor_groups={g: SensorGroupConfig(rate_hz={"slow": 1, "medium": 10, "fast": 100}[g], samples_per_cycle=n, sensors=ss)
                           for g, (n, ss) in GROUPS.items()},
            target_component={"cooler": "cooler", "valve": "valve", "pump": "pump", "accumulator": "accumulator"}[component],
        ),
        split=SplitConfig(train_fraction=0.7, val_fraction=0.15, block_size=cfg["split"]["block_size"]),
        model=ModelConfig(slow_hidden_dim=m["slow_hidden_dim"], slow_branch_latent=m["slow_branch_latent"],
                          medium_base_channels=m["medium_base_channels"], medium_branch_latent=m["medium_branch_latent"],
                          fast_base_channels=m["fast_base_channels"], fast_branch_latent=m["fast_branch_latent"],
                          fused_latent_dim=m["fused_latent_dim"], branch_weights={"slow": 1.0, "medium": 1.0, "fast": 1.0}),
        train=TrainConfig(batch_size=t["batch_size"], epochs=t["epochs"], learning_rate=t["learning_rate"],
                          weight_decay=t["weight_decay"], early_stopping_patience=t["early_stopping_patience"], device=t["device"]),
        evaluation=EvaluationConfig(threshold_percentile=99.0),
    )


@torch.no_grad()
def score_arrays(model, norm, arrays: dict[str, np.ndarray], device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    data = norm.transform(arrays)
    n = len(next(iter(data.values())))
    comb, branches = [], {g: [] for g in data}
    w = {"slow": 1.0, "medium": 1.0, "fast": 1.0}
    for i in range(0, n, 64):
        batch = {g: torch.from_numpy(a[i:i + 64]).float().to(device) for g, a in data.items()}
        be = per_branch_mse(batch, model(batch))
        comb.append(combined_mse(be, w).cpu().numpy())
        for g in data:
            branches[g].append(be[g].cpu().numpy())
    return np.concatenate(comb), {g: np.concatenate(v) for g, v in branches.items()}


class HydraulicService(Service):
    name = "hydraulic"
    description = "Per-component sensor-fusion anomaly detection for 60 s hydraulic cycles (17 sensors, 3 rates)."
    base_columns = ["file", "status", "verdict", "score", "message"]
    extra_columns = ["component", "threshold", "error_slow", "error_medium", "error_fast", "component_note"]

    def default_config(self) -> dict:
        return {
            "components": list(DEFAULT_COMPONENTS),
            "healthy_values": dict(HEALTHY),
            "model": {"slow_hidden_dim": 128, "slow_branch_latent": 16, "medium_base_channels": 16,
                      "medium_branch_latent": 32, "fast_base_channels": 16, "fast_branch_latent": 64, "fused_latent_dim": 32},
            "train": {"epochs": 60, "batch_size": 32, "learning_rate": 1.0e-3, "weight_decay": 1.0e-5,
                      "early_stopping_patience": 10, "device": "cpu"},
            "split": {"block_size": 20, "train_fraction": 0.70, "validation_fraction": 0.15},
            "calibration": {"target_false_alarm": 0.05, "min_healthy_cycles": 150},
        }

    def validate_config(self, cfg: dict) -> list[str]:
        p = []
        bad = [c for c in cfg["components"] if c not in HEALTHY]
        if bad or not cfg["components"]:
            p.append(f"components must be a non-empty list from {list(HEALTHY)} (got {cfg['components']})")
        if not 0 < cfg["calibration"]["target_false_alarm"] < 0.5:
            p.append("calibration.target_false_alarm must be between 0 and 0.5")
        s = cfg["split"]
        if not 0.3 <= s["train_fraction"] < 1 or not 0.05 <= s["validation_fraction"] < 0.5 or s["train_fraction"] + s["validation_fraction"] > 0.9:
            p.append("split: train_fraction in [0.3,1), validation_fraction in [0.05,0.5), together <= 0.9 (the rest calibrates the threshold)")
        return p

    def add_train_args(self, p) -> None:
        p.add_argument("--components", default=None, help="comma list, e.g. cooler,valve (default: config `components`)")
        p.add_argument("--seeds", type=int, default=1, help="seeds per component; best validation run is kept (default 1)")
        p.add_argument("--epochs", type=int, default=None, help="override train.epochs (testing)")

    # ---- training data -------------------------------------------------
    def _read_status(self, folder: Path) -> list[dict]:
        f = folder / "status.csv"
        if not f.is_file():
            raise InputError(f"{f} is missing. It lists one cycle per row, in TIME ORDER: columns file,cooler,valve,pump,accumulator "
                             "(the condition codes of the ZeMA profile.txt). Blank = unknown.")
        rows = list(csv.DictReader(open(f, encoding="utf8")))
        if not rows or "file" not in rows[0]:
            raise InputError("status.csv needs a header with a `file` column")
        return rows

    def train(self, ctx: Context) -> Outcome:
        cfg = ctx.config
        comps = ctx.args.components.split(",") if ctx.args.components else cfg["components"]
        bad = [c for c in comps if c not in HEALTHY]
        if bad:
            raise ConfigError(f"unknown component(s) {bad}; choose from {list(HEALTHY)}")
        hv = cfg["healthy_values"]
        rows = self._read_status(ctx.layout.train_dir)
        ctx.log.info("reading %d cycle files", len(rows))
        arrays: dict[str, list] = {g: [] for g in GROUPS}
        keep, skipped = [], []
        for i, r in enumerate(rows):
            try:
                cyc = load_cycle(ctx.layout.train_dir / r["file"])
            except (InputError, FileNotFoundError) as e:
                skipped.append(f"{r['file']}: {e}"); continue
            for g in GROUPS:
                arrays[g].append(cyc[g])
            keep.append(r)
        if not keep:
            raise InputError("no readable cycle files. " + "; ".join(skipped[:3]))
        A = {g: np.stack(v) for g, v in arrays.items()}
        files = [ctx.layout.train_dir / r["file"] for r in keep] + [ctx.layout.train_dir / "status.csv"]
        t_cfg, s_cfg, c_cfg = cfg["train"], cfg["split"], cfg["calibration"]
        epochs = ctx.args.epochs or t_cfg["epochs"]
        device = select_device(t_cfg["device"])
        metrics: dict = {"n_cycles": len(keep), "components": {}, "skipped": skipped}
        summary, warns = [], []
        if skipped:
            warns.append(f"{len(skipped)} cycle file(s) skipped: " + "; ".join(skipped[:3]))
        trained = {}
        for comp in comps:
            col = [r.get(comp, "") for r in keep]
            if all(v in ("", None) for v in col):
                raise InputError(f"status.csv has no `{comp}` values")
            val = np.array([int(float(v)) if v not in ("", None) else -10**9 for v in col])
            healthy_idx = np.where(val == hv[comp])[0]          # time order = row order
            degraded_idx = np.where((val != hv[comp]) & (val > -10**9))[0]
            if len(healthy_idx) < c_cfg["min_healthy_cycles"]:
                raise InputError(f"{comp}: only {len(healthy_idx)} healthy cycles (need >= {c_cfg['min_healthy_cycles']})")
            tr_i, va_i, ca_i = _block_shuffle_split(healthy_idx, s_cfg["train_fraction"], s_cfg["validation_fraction"], s_cfg["block_size"], seed=0)
            if min(len(tr_i), len(va_i), len(ca_i)) < 10:
                raise InputError(f"{comp}: split too small (train {len(tr_i)}, validation {len(va_i)}, calibration {len(ca_i)}); add cycles")
            sub = lambda idx: {g: A[g][idx] for g in A}
            norm = MultiBranchNormalizer().fit(sub(tr_i))
            hcfg = make_config(cfg, comp)
            per_seed = []
            for seed in range(ctx.args.seeds):
                set_seed(seed)
                model = FusionAutoencoder(hcfg).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"])
                tl = DataLoader(MultiBranchDataset(norm.transform(sub(tr_i))), batch_size=t_cfg["batch_size"], shuffle=True, drop_last=True)
                vl = DataLoader(MultiBranchDataset(norm.transform(sub(va_i))), batch_size=t_cfg["batch_size"], shuffle=False)
                best, state, bad_n = float("inf"), None, 0
                for ep in range(1, epochs + 1):
                    trs = run_epoch(model, tl, opt, device, hcfg.model.branch_weights, train=True)
                    vas = run_epoch(model, vl, opt, device, hcfg.model.branch_weights, train=False)
                    if vas["combined"] < best:
                        best, bad_n = vas["combined"], 0
                        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        bad_n += 1
                    ctx.log.info("%s seed %d epoch %3d train=%.4f val=%.4f%s", comp, seed, ep, trs["combined"], vas["combined"], " *" if bad_n == 0 else "")
                    if bad_n >= t_cfg["early_stopping_patience"]:
                        break
                model.load_state_dict(state)
                cal_err, _ = score_arrays(model, norm, sub(ca_i), device)
                per_seed.append({"seed": seed, "best_val_mse": best, "epochs_run": ep, "state": state, "cal_err": cal_err})
            win = min(per_seed, key=lambda r: r["best_val_mse"])
            thr = float(np.quantile(win["cal_err"], 1 - c_cfg["target_false_alarm"]))
            # fresh model object for diagnostics
            model = FusionAutoencoder(hcfg).to(device); model.load_state_dict(win["state"])
            m = {"n_healthy": int(len(healthy_idx)), "n_train": int(len(tr_i)), "n_validation": int(len(va_i)),
                 "n_calibration": int(len(ca_i)), "n_degraded_labelled": int(len(degraded_idx)), "threshold": thr,
                 "target_false_alarm": c_cfg["target_false_alarm"], "winner_seed": win["seed"],
                 "seeds": [{k: v for k, v in r.items() if k in ("seed", "best_val_mse", "epochs_run")} for r in per_seed],
                 "healthy_calibration_error_quantiles": {str(q): float(np.quantile(win["cal_err"], q)) for q in (0.5, 0.9, 0.95, 0.99)},
                 "note": NOTES[comp]}
            line = f"{comp}: {len(healthy_idx)} healthy cycles, threshold {thr:.4g} for {c_cfg['target_false_alarm']:.0%} false alarms"
            if len(degraded_idx) >= 20:
                from sklearn.metrics import roc_auc_score
                deg_err, _ = score_arrays(model, norm, sub(degraded_idx), device)
                y = np.r_[np.zeros(len(win["cal_err"])), np.ones(len(deg_err))]
                m["labelled_check"] = {
                    "roc_auc_vs_calibration_healthy": float(roc_auc_score(y, np.r_[win["cal_err"], deg_err])),
                    "recall_at_threshold": float((deg_err > thr).mean()), "n_degraded": int(len(deg_err)),
                    "caveat": "healthy calibration cycles set the threshold and are reused here; degraded cycles were never trained on"}
                line += f"; labelled check: ROC-AUC {m['labelled_check']['roc_auc_vs_calibration_healthy']:.3f}, recall {m['labelled_check']['recall_at_threshold']:.0%}"
            elif len(degraded_idx) == 0:
                warns.append(f"{comp}: no labelled degraded cycles in status.csv, so detection performance is UNMEASURED")
            if ctx.args.seeds == 1:
                m["single_seed"] = True
            if comp == "pump":
                warns.append("pump is EXPERIMENTAL (weak in the study)")
            if comp == "cooler":
                warns.append("cooler: in the study its score tracked campaign time; check that wear is not simply 'later in the log'")
            summary.append(line)
            trained[comp] = {"state": win["state"], "norm": norm, "thr": thr}
            metrics["components"][comp] = m
        if ctx.args.seeds == 1:
            warns.append("single seed per component: the study found seed spread can exceed effect sizes; use --seeds 3 if time allows")
        warns.append("cycle order in status.csv defines time; the split shuffles 20-cycle blocks across it (as in the study)")
        # write artifacts
        for comp, tr in trained.items():
            torch.save(tr["state"], ctx.version_dir / f"{comp}_model.pt")
            tr["norm"].save(ctx.version_dir / f"{comp}_normalizer.npz")
        (ctx.version_dir / "components.json").write_text(json.dumps({
            "components": {c: {"threshold": t["thr"], "note": NOTES[c]} for c, t in trained.items()},
            "model": cfg["model"], "healthy_values": hv}, indent=2), encoding="utf8")
        return Outcome("train", metrics, summary, files, warns)

    # ---- inference -----------------------------------------------------
    def load_model(self, version_dir: Path, meta: dict):
        comp = json.loads((version_dir / "components.json").read_text(encoding="utf8"))
        device = torch.device("cpu")
        out = {}
        for c, info in comp["components"].items():
            cfg = {"model": comp["model"], "train": {"batch_size": 32, "epochs": 1, "learning_rate": 1e-3, "weight_decay": 0,
                                                     "early_stopping_patience": 1, "device": "cpu"}, "split": {"block_size": 20}}
            m = FusionAutoencoder(make_config(cfg, c)).to(device)
            m.load_state_dict(torch.load(version_dir / f"{c}_model.pt", map_location=device, weights_only=True))
            m.eval()
            out[c] = {"model": m, "norm": MultiBranchNormalizer.load(version_dir / f"{c}_normalizer.npz", list(GROUPS)),
                      "threshold": info["threshold"], "note": info["note"]}
        return {"components": out, "device": device}

    def list_inputs(self, in_dir: Path) -> list[Path]:
        return [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")] if in_dir.is_dir() else []

    def infer_one(self, m, path: Path, ctx: Context) -> list[dict]:
        cyc = load_cycle(path)
        arrays = {g: v[None] for g, v in cyc.items()}
        rows = []
        for c, info in m["components"].items():
            comb, br = score_arrays(info["model"], info["norm"], arrays, m["device"])
            s = float(comb[0])
            rows.append({"component": c, "verdict": "anomaly" if s > info["threshold"] else "healthy", "score": s,
                         "threshold": info["threshold"], "error_slow": float(br["slow"][0]), "error_medium": float(br["medium"][0]),
                         "error_fast": float(br["fast"][0]), "component_note": info["note"], "message": ""})
        return rows

    def doctor(self, layout, cfg):
        has = (layout.train_dir / "status.csv").is_file()
        return [("ok" if has else "warn", "train/status.csv", "present" if has else "missing (needed for `train`)")]
