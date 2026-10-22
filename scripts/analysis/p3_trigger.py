"""Phase 3 Track C: the trend trigger as a sequential test (notes/30, pre-registered).

The statistic is v1.1's level (log RMS per record vs the unit's own baseline; LDM: the
operating-point-conditioned residual). Only the trigger changes:
    T0  v1.1 CUSUM, h from FA budget 0 leave one unit out
    T1  conformal CUSUM martingale, fixed baseline reference, alarm at Ville's 100
    T2  the same, sliding reference lagged by B records
    T3  CUSUM on the EWMA forecast residual
    T4  fleet-relative CUSUM (IMS only, descriptive)

Usage (box; needs xjtu_sy staged and ~/work/scratch/ims_service):
    python3 scripts/analysis/p3_trigger.py
Output: outputs/p3/trigger/results.json, report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.method import CUSUM_K, cusum
from p3_ims_cache import DATA as IMS_DATA, stamp_time
from v2_cache import load_v2
from wp18_ims import FAILED as IMS_FAILED
from wp7_xjtu import rms_per_minute

OUT = Path("outputs/p3/trigger")
EPS = (0.05, 0.1, 0.2, 0.3, 0.5)
LOG_A = np.log(100.0)
GATE_FA_DAY = 0.2


# --- record series per unit -----------------------------------------------------
# unit dict: s (statistic per record), hours (time of each record), B, fa (bool mask), failed

def unit(s, hours, B, fa_lo, fa_hi, failed, test=None):
    s, hours = np.asarray(s, float), np.asarray(hours, float)
    fa = np.zeros(len(s), bool)
    if fa_hi > fa_lo:
        fa[fa_lo:fa_hi] = True
    return {"s": s, "hours": hours, "B": B, "fa": fa, "failed": failed, "test": test}


def ims_units() -> dict:
    out = {}
    for u in sorted(p.name for p in (IMS_DATA / "train").iterdir()):
        t, b = (int(x[1:]) for x in u.split("_"))
        files = sorted((IMS_DATA / "train" / u).glob("*.npy")) + sorted((IMS_DATA / "in" / u).glob("*.npy"))
        rms = np.array([np.sqrt(np.mean((x - x.mean()) ** 2)) for x in (np.load(f) for f in files)])
        t0 = stamp_time(files[0].name)
        hours = [(stamp_time(f.name) - t0).total_seconds() / 3600 for f in files]
        failed = (t, b) in IMS_FAILED
        n, B = len(files), 200
        out[u] = unit(np.log(rms), hours, B, B, int(n * (0.5 if failed else 0.9)), failed, test=t)
    return out


def xjtu_units() -> dict:
    out = {}
    for r in pdx.xjtu_bearings(Path("data/xjtu_sy")).itertuples():
        s = np.log(rms_per_minute(r.path, 0))
        n = len(s)
        out[r.bearing] = unit(s, np.arange(1, n + 1) / 60, 20, 20, 60 if n >= 120 else 0, True)
    return out


def ldm_units() -> dict:
    """notes/25 L_cond: log RMS minus a linear fit on [log speed, static, dynamic load] over the baseline."""
    cpw = load_v2("paderborn_ldm", fset="cpw")
    cpw = cpw[(cpw.channel == "accHorizRear_A") & (cpw.set_rpm >= 1000)]
    out = {}
    for b, d in cpw.groupby("unit"):
        rec = d.groupby("record").agg(rms=("td_rms", "mean"), speed=("speed_rpm", "first"), stat=("stat_N", "first"),
                                      dyn=("peak_dyn_N", "first"), time=("time", "first")).sort_index()
        n, B = len(rec), 150
        lr = np.log(rec.rms.to_numpy())
        X = np.c_[np.ones(n), np.log(rec.speed.to_numpy()), rec.stat.to_numpy(), rec.dyn.to_numpy()]
        coef, *_ = np.linalg.lstsq(X[:B], lr[:B], rcond=None)
        t = pd.to_datetime(rec.time)
        hours = ((t - t.iloc[0]).dt.total_seconds() / 3600).to_numpy()
        out[b] = unit(lr - X @ coef, hours, B, B, 450 if n >= 1000 else 0, True)
    return out


def femto_units() -> dict:
    df = pd.read_csv("outputs/wp16/femto_records.csv.gz")
    out = {}
    for b, d in df.groupby("bearing"):
        d = d.sort_values("record")
        n, B = len(d), 180
        out[b] = unit(np.log(d.rms_h.to_numpy()), d.record.to_numpy() * 10 / 3600, B, B,
                      B + 720 if n >= B + 820 else 0, True)
    return out


# --- triggers -------------------------------------------------------------------

def zstd(u) -> np.ndarray:
    base = u["s"][:u["B"]]
    return (u["s"] - base.mean()) / (base.std(ddof=1) + 1e-12)


def alarms_cusum(x: np.ndarray, start: int, h: float, k: float = CUSUM_K) -> list[int]:
    out, c = [], 0.0
    for t in range(start, len(x)):
        c = max(0.0, c + x[t] - k)
        if c > h:
            out.append(t)
            c = 0.0
    return out


def conformal_p(ref: np.ndarray, v: float, rng) -> float:
    return (np.sum(ref > v) + rng.uniform() * (np.sum(ref == v) + 1)) / (len(ref) + 1)


def log_bet(p: float) -> float:
    return float(np.log(np.mean([e * p ** (e - 1) for e in EPS])))


def alarms_martingale(u, sliding: bool, seed: int = 0) -> list[int]:
    s, B, rng = u["s"], u["B"], np.random.default_rng(seed)
    out, L = [], 0.0
    for t in range(B, len(s)):
        ref = s[t - 2 * B:t - B] if sliding and t > 2 * B else s[:B]
        L = max(0.0, L) + log_bet(conformal_p(ref, s[t], rng))
        if L >= LOG_A:
            out.append(t)
            L = 0.0
    return out


def ewma_residual(u) -> np.ndarray:
    z, B = zstd(u), u["B"]
    lam, f = 2 / (B + 1), 0.0
    r = np.empty(len(z))
    for t in range(len(z)):
        r[t] = z[t] - f
        f = lam * z[t] + (1 - lam) * f
    return r / (r[1:B].std(ddof=1) + 1e-12)


def fleet_z(units: dict) -> dict:
    """IMS: unit z minus the median z of the other bearings of the same test, record by record."""
    out = {}
    for name, u in units.items():
        sib = [zstd(v) for k, v in units.items() if k != name and v["test"] == u["test"]]
        n = min(len(u["s"]), *(len(x) for x in sib))
        out[name] = zstd(u)[:n] - np.median(np.vstack([x[:n] for x in sib]), 0)
    return out


def loo_h(series: dict, units: dict) -> dict:
    """FA budget 0 leaving one unit out: h = max CUSUM over the OTHER units' false-alarm regions."""
    peak = {k: (np.max(cusum(x, start=units[k]["B"])[units[k]["fa"][:len(x)]]) if units[k]["fa"][:len(x)].any() else -np.inf)
            for k, x in series.items()}
    return {k: max(v for o, v in peak.items() if o != k) + 1e-9 for k in series}


def score(units: dict, alarms: dict) -> dict:
    fa_events = fa_hours = 0.0
    det, leads, per = 0, [], {}
    for k, u in units.items():
        a = np.asarray(alarms[k], int)
        fa_idx = np.flatnonzero(u["fa"])
        if len(fa_idx):
            fa_events += np.isin(a, fa_idx).sum()
            fa_hours += u["hours"][fa_idx[-1]] - u["hours"][fa_idx[0] - 1]
        after = (fa_idx[-1] + 1) if len(fa_idx) else u["B"]
        hit = a[a >= after]
        lead = float(u["hours"][-1] - u["hours"][hit[0]]) if u["failed"] and len(hit) else None
        if lead is not None:
            det += 1
            leads.append(lead)
        per[k] = {"failed": u["failed"], "n_alarms": int(len(a)), "lead_h": lead,
                  "fa_events": int(np.isin(a, fa_idx).sum()) if len(fa_idx) else None}
    return {"fa_per_day": float(fa_events / (fa_hours / 24)) if fa_hours else float("nan"),
            "fa_days": float(fa_hours / 24), "detected": det, "failed": int(sum(u["failed"] for u in units.values())),
            "median_lead_h": float(np.median(leads)) if leads else 0.0, "units": per}


def run_dataset(units: dict, fleet: bool = False) -> dict:
    res = {}
    z = {k: zstd(u) for k, u in units.items()}
    h = loo_h(z, units)
    res["T0"] = score(units, {k: alarms_cusum(z[k], units[k]["B"], h[k]) for k in units})
    res["T1"] = score(units, {k: alarms_martingale(u, sliding=False) for k, u in units.items()})
    res["T2"] = score(units, {k: alarms_martingale(u, sliding=True) for k, u in units.items()})
    r = {k: ewma_residual(u) for k, u in units.items()}
    h3 = loo_h(r, units)
    res["T3"] = score(units, {k: alarms_cusum(r[k], units[k]["B"], h3[k]) for k in units})
    if fleet:
        f = fleet_z(units)
        trimmed = {k: {**u, "fa": u["fa"][:len(f[k])], "s": u["s"][:len(f[k])], "hours": u["hours"][:len(f[k])]}
                   for k, u in units.items()}
        h4 = loo_h(f, trimmed)
        res["T4"] = score(trimmed, {k: alarms_cusum(f[k], units[k]["B"], h4[k]) for k in units})
    return res


def ims_nonfailing_fa(units: dict, res: dict) -> dict:
    out = {}
    for t, r in res.items():
        ev = sum(r["units"][k]["fa_events"] for k, u in units.items() if not u["failed"])
        days = sum((u["hours"][np.flatnonzero(u["fa"])[-1]] - u["hours"][u["B"] - 1]) / 24
                   for u in units.values() if not u["failed"])
        out[t] = ev / days
    return out


def main():
    data = {"ims": ims_units(), "xjtu_sy": xjtu_units(), "paderborn_ldm": ldm_units(), "femto": femto_units()}
    res = {d: run_dataset(u, fleet=(d == "ims")) for d, u in data.items()}
    ims_nf = ims_nonfailing_fa(data["ims"], res["ims"])
    total = {t: sum(res[d][t]["detected"] for d in res if t in res[d]) for t in ("T0", "T1", "T2", "T3")}
    verdict = {t: bool(ims_nf[t] <= GATE_FA_DAY and total[t] >= total["T0"]) for t in ("T1", "T2", "T3")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"results": res, "ims_nonfailing_fa_per_day": ims_nf,
                                                  "detected_total": total, "pass": verdict}, indent=1, default=float))
    L = ["# Phase 3 Track C: the trigger as a sequential test (notes/30)", "",
         "Generated by `scripts/analysis/p3_trigger.py`. Statistic = v1.1 level (LDM: L_cond). "
         f"Gate: IMS non-failing <= {GATE_FA_DAY} false events/day and detected (of 53) >= T0.", "",
         "| trigger | IMS non-failing FA/day | detected of 53 | gate |", "|---|---|---|---|"]
    for t in ("T0", "T1", "T2", "T3"):
        L.append(f"| {t} | {ims_nf[t]:.2f} | {total[t]} | {'—' if t == 'T0' else ('PASS' if verdict[t] else 'fail')} |")
    L.append(f"| T4 (IMS only, descriptive) | {ims_nf['T4']:.2f} | {res['ims']['T4']['detected']}/4 on IMS | — |")
    L += ["", "| dataset | trigger | FA/day in FA regions (days observed) | detected | median lead h |", "|---|---|---|---|---|"]
    for d, r in res.items():
        for t, x in r.items():
            L.append(f"| {d} | {t} | {x['fa_per_day']:.2f} ({x['fa_days']:.1f}) | {x['detected']}/{x['failed']} | {x['median_lead_h']:.1f} |")
    md = "\n".join(L) + "\n"
    (OUT / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
