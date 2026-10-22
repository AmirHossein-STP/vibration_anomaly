"""XJTU-SY run-to-failure (MASTER_PLAN WP5/WP7; pre-registered in notes/17 §4).

Per bearing, its OWN early life is the healthy baseline: minutes 1-20, one
1 s window per minute (horizontal channel primary, vertical reported).

Detector: the frozen physics score (arm d, CPW+SES). The threshold is the
conformal quantile at alpha = 0.05 of the 20 calibration scores, i.e. their
maximum. An alarm is 3 consecutive minutes above it. Also reported:
- d_xf, the post-hoc cross-fitted calibration of notes/17 §5 (leave one minute out);
- the classical baseline: RMS above mean + 3 sd of the same 20 minutes,
  3 consecutive minutes.

Per bearing: first alarm, lead time to end of life, the alarm rate in
minutes 21-40 (ASSUMED healthy; an assumption, not a measurement), and
typing (majority race over the last 10% of life) against the documented
failed element.

Usage: python scripts/analysis/wp7_xjtu.py   (box: needs data/xjtu_sy for RMS)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data.public_datasets import xjtu_bearings
from wp1_rescore import conformal_threshold
from wp5_cache import load_wp5
from wp5_score import PEAKS, log_peaks

CAL_MIN, ALPHA, RUN = 20, 0.05, 3
ROOT = Path("data/xjtu_sy")
SINGLE_RACE = {"outer": 2, "inner": 1}


def first_sustained(over: np.ndarray, minutes: np.ndarray, start: int) -> int | None:
    """Minute at which `RUN` consecutive minutes > threshold are first complete (after `start`)."""
    k = 0
    for m, o in zip(minutes, over):
        if m <= start:
            continue
        k = k + 1 if o else 0
        if k >= RUN:
            return int(m)
    return None


def zscores(cal: pd.DataFrame, f: pd.DataFrame) -> np.ndarray:
    lp = log_peaks(cal)
    return (log_peaks(f) - lp.mean(0)) / (lp.std(0) + 1e-12)


def loo_cal_scores(cal: pd.DataFrame) -> np.ndarray:
    lp, out = log_peaks(cal), np.empty(len(cal))
    for i in range(len(cal)):
        m = np.arange(len(cal)) != i
        out[i] = ((lp[i] - lp[m].mean(0)) / (lp[m].std(0) + 1e-12)).max()
    return out


def rms_per_minute(path: str, channel: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return np.sqrt((np.asarray(arr[:, :, channel], dtype=np.float64) ** 2).mean(1))


def analyse(df: pd.DataFrame, bearings: pd.DataFrame, channel: str) -> pd.DataFrame:
    rows = []
    for b in bearings.itertuples():
        f = df[(df.unit == b.bearing) & (df.channel == channel)].sort_values("minute")
        minutes = f.minute.to_numpy()
        cal = f[f.minute <= CAL_MIN]
        z = zscores(cal, f)
        score = z.max(1)
        out = {"bearing": b.bearing, "condition": b.condition, "failed": b.failed, "life_min": int(b.n_minutes)}
        for arm, cal_scores in (("d", score[minutes <= CAL_MIN]), ("d_xf", loo_cal_scores(cal))):
            thr = conformal_threshold(cal_scores, ALPHA)
            over = score > thr
            alarm = first_sustained(over, minutes, CAL_MIN)
            early = (minutes > CAL_MIN) & (minutes <= 2 * CAL_MIN)
            out[f"{arm}_alarm"] = alarm
            out[f"{arm}_lead"] = (int(b.n_minutes) - alarm) if alarm else None
            out[f"{arm}_rate_21_40"] = float(over[early].mean()) if early.any() else np.nan
            out[f"{arm}_early_alarm"] = bool(alarm is not None and alarm <= 2 * CAL_MIN)
        # classical baseline
        rms = rms_per_minute(b.path, 0 if channel == "horizontal" else 1)
        r_cal = rms[:CAL_MIN]
        r_over = rms > r_cal.mean() + 3 * r_cal.std(ddof=1)
        r_alarm = first_sustained(r_over, np.arange(1, len(rms) + 1), CAL_MIN)
        out["rms_alarm"] = r_alarm
        out["rms_lead"] = (int(b.n_minutes) - r_alarm) if r_alarm else None
        out["rms_rate_21_40"] = float(r_over[CAL_MIN:2 * CAL_MIN].mean())
        # typing: majority over the last 10% of life (>= 3 minutes)
        tail = minutes > b.n_minutes - max(3, int(np.ceil(0.1 * b.n_minutes)))
        pred = np.where(z[tail, 3:].max(1) > z[tail, :3].max(1), 1, 2)
        out["typed"] = {1: "inner", 2: "outer"}[int(np.bincount(pred, minlength=3)[1:].argmax() + 1)]
        out["typed_inner_share"] = float((pred == 1).mean())
        out["typing_correct"] = (out["typed"] == b.failed) if b.failed in SINGLE_RACE else None
        rows.append(out)
        f = f.assign(score=score)
        traj[(b.bearing, channel)] = (minutes, score, conformal_threshold(score[minutes <= CAL_MIN], ALPHA),
                                      rms, r_cal.mean() + 3 * r_cal.std(ddof=1))
    return pd.DataFrame(rows)


traj: dict = {}


def readings(t: pd.DataFrame, arm: str) -> dict:
    race = t[~t.failed.isin(["cage"])]
    single = t[t.failed.isin(SINGLE_RACE)]
    alarmed = race[f"{arm}_alarm"].notna()
    no_later = sum(1 for r in race.to_dict("records")
                   if pd.notna(r[f"{arm}_alarm"]) and (pd.isna(r["rms_alarm"]) or r[f"{arm}_alarm"] <= r["rms_alarm"]))
    n_typed = int(single.typing_correct.astype(bool).sum())
    return {"alarm_before_end": int(alarmed.sum()), "of": len(race), "no_later_than_rms": no_later,
            "typing_correct": n_typed, "typing_of": len(single),
            "supports": bool(alarmed.sum() >= 10 and no_later > len(race) / 2 and n_typed >= 2 / 3 * len(single))}


def figure(path: Path, channel: str, bearings: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(3, 5, figsize=(16, 8.5), sharey=True)
    for ax, b in zip(axes.ravel(), bearings.itertuples()):
        minutes, score, thr, rms, rthr = traj[(b.bearing, channel)]
        # both indexed to their own alarm threshold: one axis, no dual scale
        ax.plot(minutes, np.maximum(score / thr, 1e-2), color=blue, lw=1.2, label="physics score / threshold")
        ax.plot(np.arange(1, len(rms) + 1), rms / rthr, color=orange, lw=1.2, label="RMS / (mean + 3 sd)")
        ax.axhline(1, color=muted, lw=0.8, ls="--")
        ax.axvspan(0, CAL_MIN, color="#f0efec", lw=0)
        ax.set_yscale("log")
        ax.set_title(f"{b.bearing} · {b.failed}", fontsize=9, color=ink, loc="left")
        ax.tick_params(labelsize=7, colors=muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.supxlabel("minute of life (shaded: calibration, minutes 1-20)", fontsize=9, color=muted)
    fig.supylabel("value / its own alarm threshold (log)", fontsize=9, color=muted)
    fig.suptitle(f"XJTU-SY run to failure, {channel} channel: healthy-referenced physics score vs RMS",
                 fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/wp7"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bearings = xjtu_bearings(ROOT)
    df = load_wp5("xjtu_sy", "hd_cpw_ses")
    res, L = {}, ["# WP7 / WP5: XJTU-SY run to failure", "",
                  "Generated by `scripts/analysis/wp7_xjtu.py`; pre-registered in notes/17 §4. Own baseline = minutes 1-20; "
                  "alarm = 3 consecutive minutes over the threshold (conformal alpha 0.05 = max of the 20 calibration scores). "
                  "`rate 21-40` assumes minutes 21-40 are healthy.", ""]
    for channel in ("horizontal", "vertical"):
        t = analyse(df, bearings, channel)
        res[channel] = {"bearings": t.to_dict(orient="records"),
                        "readings": {arm: readings(t, arm) for arm in ("d", "d_xf")}}
        L += [f"## {channel} channel", ""]
        for arm in ("d", "d_xf"):
            r = res[channel]["readings"][arm]
            L.append(f"- **{arm}{' (post-hoc)' if arm == 'd_xf' else ' (pre-registered)'}**: alarm before end of life on "
                     f"{r['alarm_before_end']}/{r['of']} race-involving bearings; no later than RMS on {r['no_later_than_rms']}/{r['of']}; "
                     f"typing correct {r['typing_correct']}/{r['typing_of']} single-race failures -> "
                     f"**{'supports' if r['supports'] else 'does not support'}** (criteria ≥10, majority, ≥2/3).")
        L += ["", "| bearing | failed | life (min) | d alarm (lead) | d rate 21-40 | d_xf alarm (lead) | d_xf rate 21-40 | "
              "RMS alarm (lead) | RMS rate 21-40 | typed (inner share) |", "|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda a, l: "—" if a is None or (isinstance(a, float) and np.isnan(a)) else f"{int(a)} ({int(l)})"
        for r in t.itertuples():
            L.append(f"| {r.bearing} | {r.failed} | {r.life_min} | {fmt(r.d_alarm, r.d_lead)} | {r.d_rate_21_40:.2f} | "
                     f"{fmt(r.d_xf_alarm, r.d_xf_lead)} | {r.d_xf_rate_21_40:.2f} | {fmt(r.rms_alarm, r.rms_lead)} | "
                     f"{r.rms_rate_21_40:.2f} | {r.typed}{'' if r.typing_correct is None else (' ✓' if r.typing_correct else ' ✗')} "
                     f"({r.typed_inner_share:.2f}) |")
        L.append("")
        figure(args.out / f"xjtu_{channel}.png", channel, bearings)
    md = "\n".join(L) + "\n"
    (args.out / "wp7_xjtu_report.md").write_text(md)
    (args.out / "wp7_xjtu_results.json").write_text(json.dumps(res, indent=1, default=lambda o: None if o is None else float(o)))
    print(md)


if __name__ == "__main__":
    main()
