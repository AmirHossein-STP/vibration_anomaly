"""Service `bearing`: rolling-element bearing monitoring for a fleet of units,
running method v1.0 (src/domain_adaptation/method.py; specification notes/22).

What it does, per unit (one physical bearing position):
* `calibrate` reads that unit's HEALTHY recordings from `train/<unit>/` (or
  `train/` with a `unit` sidecar key, or `train/` for the default unit) and
  stores its own baseline. No fault data, no source dataset, no training.
* `infer` scores new recordings from `in/<unit>/` against that unit's
  baseline: share of 1 s windows over the cross-fitted conformal threshold ->
  verdict (healthy / watch / faulty), which race the rise points at
  (typing), and per-unit CUSUM trend statistics across recordings.

Bearing per unit: a catalog designation (src/domain_adaptation/bearings.py),
explicit geometry, or manufacturer fault factors. With none of these the unit
runs the geometry-free score (notes/23): detection without typing.

Shaft speed per recording is REQUIRED (sidecar `rpm`, or `rpm` in the
config / per unit); speed estimated from vibration is not adequate (notes/20).

Not covered (notes/22 §5): ball and cage faults, machines without a speed
signal, units without healthy history of their own.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from domain_adaptation.bearings import FaultOrders, from_dims, from_factors, lookup
from domain_adaptation.data.transforms import resample_to_rate
from domain_adaptation.features.envelope import BearingGeometry, envelope_order_features
from domain_adaptation.features.geofree import BIN_CENTRES, BIN_NAMES, geofree_features
from domain_adaptation.iso20816 import velocity_rms_mm_s, zone
from domain_adaptation.method import (
    CPW_SES, CUSUM_K, HOP, LINES, METHOD_VERSION, OUTER, WIN, Baseline, default_groups, fit_baseline,
    fit_level_model, level_z,
)
from serving_core.service import ConfigError, Context, NotSupported, Outcome, Service
from serving_core.signals import InputError, list_signal_files, load_signal, read_sidecar
from serving_core.util import iso

FS_MODEL = 12000
MIN_CUSUM_H = 5.0          # floor for a trend threshold set from a short baseline
TREND_FILE = "bearing_trend.csv"
TREND_COLUMNS = ["time", "unit", "file", "physics_z", "level_z", "cusum_physics", "cusum_level", "trend_alarm"]


# ───────────────────────── bearing and speed ─────────────────────────
def resolve_bearing(d: dict | None) -> BearingGeometry | FaultOrders | None:
    """Config `bearing:` block -> geometry, fault orders, or None (geometry-free)."""
    d = d or {}
    dims = [d.get(k) for k in ("n_balls", "ball_diameter", "pitch_diameter")]
    if d.get("designation"):
        try:
            return lookup(str(d["designation"]))
        except KeyError as e:
            raise ConfigError(str(e).strip('"'))
    if all(v is not None for v in dims):
        return from_dims(int(dims[0]), float(dims[1]), float(dims[2]), float(d.get("contact_angle_deg") or 0.0),
                         name=str(d.get("name") or "bearing"))
    if d.get("bpfo") is not None and d.get("bpfi") is not None:
        return from_factors(float(d["bpfo"]), float(d["bpfi"]), d.get("ftf"), d.get("bsf"), name=str(d.get("name") or "factors"))
    if any(v is not None for v in dims) or d.get("bpfo") is not None or d.get("bpfi") is not None:
        raise ConfigError("bearing: give a designation, OR n_balls + ball_diameter + pitch_diameter together, "
                          "OR bpfo + bpfi (fault factors in orders) -- or nothing for the geometry-free mode")
    return None


def geometry_to_dict(g) -> dict | None:
    if g is None:
        return None
    if isinstance(g, FaultOrders):
        return {"kind": "factors", "name": g.name, "bpfo": g.bpfo, "bpfi": g.bpfi, "ftf": g.ftf, "bsf": g.bsf}
    return {"kind": "dims", "name": g.name, "n_balls": g.n_balls, "ball_diameter": g.ball_diameter,
            "pitch_diameter": g.pitch_diameter, "contact_angle_deg": g.contact_angle_deg, "bpfo": g.bpfo, "bpfi": g.bpfi}


def geometry_from_dict(d: dict | None):
    if d is None:
        return None
    if d["kind"] == "factors":
        return FaultOrders(d["name"], d["bpfo"], d["bpfi"], d.get("ftf", float("nan")), d.get("bsf", float("nan")))
    return BearingGeometry(d["name"], d["n_balls"], d["ball_diameter"], d["pitch_diameter"], d["contact_angle_deg"])


def check_speed(rpm: float, geom, bins: bool = True) -> None:
    """The 1 s window and 12 kHz rate bound the measurable speeds."""
    shaft = rpm / 60.0
    if shaft < 5:
        raise InputError(f"rpm={rpm:g} is too low for 1 s analysis windows (need >= 300 rpm)")
    top = max(15.5 if bins else 0.0, 3 * geom.bpfi + 1 if geom is not None else 0.0) * shaft
    if top > FS_MODEL / 2 * 0.9:
        raise InputError(f"rpm={rpm:g} is too high: the fault lines ({top:.0f} Hz) would pass the 12 kHz analysis range")


# ───────────────────────── features ─────────────────────────
def window_features(signal: np.ndarray, fs_hz: float, rpm: float, geom, max_windows: int | None, bins: bool = True):
    """-> ({"lines": [n, 6] if geometry known, "bins": [n, 135] if bins}, window starts, total windows, mean log RMS)."""
    if not np.isfinite(signal).all():
        raise InputError("signal contains NaN or infinite values")
    fs_i = int(round(fs_hz))
    sig = signal if fs_i == FS_MODEL else resample_to_rate(np.asarray(signal, dtype=np.float64), fs_i, FS_MODEL)
    starts = np.arange(0, len(sig) - WIN + 1, HOP)
    if len(starts) == 0:
        raise InputError(f"too short: {len(sig) / FS_MODEL:.2f} s after resampling to {FS_MODEL} Hz; need >= 1 s")
    total = len(starts)
    if max_windows and total > max_windows:
        starts = starts[np.linspace(0, total - 1, max_windows).round().astype(int)]
    lines, binv, rms = [], [], []
    for s in starts:
        w = sig[s:s + WIN]
        if bins:
            f = geofree_features(w, FS_MODEL, rpm)
            binv.append([f[n] for n in BIN_NAMES])
        if geom is not None:
            f = envelope_order_features(w, FS_MODEL, rpm, geom, **CPW_SES)
            lines.append([f[n] for n in LINES])
        x = w - w.mean()
        rms.append(np.sqrt((x ** 2).mean()) + 1e-12)
    X = {"lines": np.asarray(lines) if geom is not None else None, "bins": np.asarray(binv) if bins else None}
    return X, starts / FS_MODEL, total, float(np.mean(np.log(rms)))


def _cusum_max(u: np.ndarray) -> float:
    c, m = 0.0, 0.0
    for x in u:
        c = max(0.0, c + float(x) - CUSUM_K)
        m = max(m, c)
    return m


# ───────────────────────── the service ─────────────────────────
class BearingService(Service):
    name = "bearing"
    description = f"Rolling-element bearing monitoring per unit, own healthy baseline (method v{METHOD_VERSION})."
    extra_columns = ["unit", "detector", "fault_type", "dominant", "rpm", "fs_hz", "threshold", "n_windows_flagged",
                     "physics_z", "level_z", "cusum_physics", "cusum_level", "trend_alarm", "iso_velocity_mm_s", "iso_zone"]

    def default_config(self) -> dict:
        return {
            # The bearing of every unit unless `units` says otherwise. One of: a catalog
            # `designation`; n_balls + ball_diameter + pitch_diameter (+ contact_angle_deg);
            # bpfo + bpfi fault factors in orders. All empty -> geometry-free detection.
            "bearing": {"designation": None, "name": None, "n_balls": None, "ball_diameter": None,
                        "pitch_diameter": None, "contact_angle_deg": 0.0, "bpfo": None, "bpfi": None},
            "units": {},            # unit -> {bearing: {...}, rpm, fs_hz, detector} overrides
            # v1.1 (notes/22 §7a): geometry-free detection (won HUST and LDM); the six fault lines
            # type the race when the bearing is known. "six_line" = v1.0's detector.
            "detector": "geometry_free",
            # numeric sidecar keys describing the operating point (e.g. ["load_N"]); with log rpm they
            # condition the level trend statistic (notes/25: varying load/speed)
            "operating_state_keys": [],
            "default_unit": "unit1",  # files directly in train/ or in/ without a `unit` sidecar
            "fs_hz": None,          # sampling rate (Hz); a sidecar `fs_hz` overrides per file
            "rpm": None,            # shaft speed; a sidecar `rpm` overrides per file (REQUIRED one way or the other)
            "signal_key": None,     # array name inside .npz files, if they hold several
            "calibration": {"alpha": 0.10, "min_windows": 200, "recommended_windows": 1800,
                            "min_recordings_warn": 3, "max_windows_per_file": 600},
            # notes/14: watch 0.40 keeps ~10% of healthy recordings at watch; faulty 0.50 ~5%.
            "verdict": {"watch_fraction": 0.40, "faulty_fraction": 0.5},
            # L0 (ISO 20816-3): broadband velocity RMS zones A-D. Needs the machine group
            # (1: 300 kW-50 MW, 2: 15-300 kW) and the acceleration unit of the files.
            "iso20816": {"group": None, "foundation": "rigid", "signal_unit": None},
        }

    def validate_config(self, cfg: dict) -> list[str]:
        p = []
        try:
            resolve_bearing(cfg["bearing"])
            for u, uc in (cfg.get("units") or {}).items():
                if not isinstance(uc, dict):
                    p.append(f"units.{u} must be a mapping"); continue
                if "bearing" in uc:
                    resolve_bearing(uc["bearing"])
        except (ConfigError, ValueError) as e:
            p.append(str(e))
        c = cfg["calibration"]
        if not 0 < c["alpha"] < 0.5:
            p.append("calibration.alpha must be between 0 and 0.5")
        v = cfg["verdict"]
        if not 0 < v["watch_fraction"] < v["faulty_fraction"] <= 1:
            p.append("verdict: need 0 < watch_fraction < faulty_fraction <= 1")
        if cfg["detector"] not in ("geometry_free", "six_line") or any(
                (uc or {}).get("detector", "geometry_free") not in ("geometry_free", "six_line") for uc in (cfg.get("units") or {}).values()):
            p.append("detector must be geometry_free or six_line")
        iso = cfg["iso20816"]
        if iso.get("group") not in (None, 1, 2) or iso.get("foundation") not in ("rigid", "flexible") \
                or iso.get("signal_unit") not in (None, "g", "m/s2", "mm/s2"):
            p.append("iso20816: group 1|2, foundation rigid|flexible, signal_unit g|m/s2|mm/s2")
        return p

    # ---- helpers -------------------------------------------------------
    def _unit_cfg(self, cfg: dict, unit: str) -> dict:
        return (cfg.get("units") or {}).get(unit, {}) or {}

    def _file_params(self, path: Path, cfg: dict, unit: str, fallback: dict | None = None) -> tuple[float, float, str | None]:
        side, uc, fb = read_sidecar(path), self._unit_cfg(cfg, unit), fallback or {}
        pick = lambda k: side.get(k, uc.get(k, cfg.get(k) if cfg.get(k) is not None else fb.get(k)))
        fs, rpm, key = pick("fs_hz"), pick("rpm"), side.get("signal_key", cfg.get("signal_key") or fb.get("signal_key"))
        if fs is None:
            raise InputError("sampling rate unknown: set `fs_hz` in config/service.yaml or a sidecar file")
        if rpm is None:
            raise InputError("shaft speed unknown: set `rpm` (config, per unit, or a sidecar file); it is required")
        try:
            fs, rpm = float(fs), float(rpm)
        except (TypeError, ValueError):
            raise InputError(f"fs_hz/rpm must be numbers (got {fs!r}, {rpm!r})")
        if fs <= 0 or rpm <= 0:
            raise InputError("fs_hz and rpm must be positive")
        return fs, rpm, key

    def _state(self, path: Path, rpm: float, keys: list[str]) -> list[float]:
        """Operating point of one recording: log rpm and the configured numeric sidecar keys."""
        side = read_sidecar(path)
        try:
            return [float(np.log(rpm))] + [float(side[k]) for k in keys]
        except KeyError as e:
            raise InputError(f"operating_state_keys: sidecar of {path.name} lacks {e}")

    def _unit_of(self, path: Path, top: Path, default: str) -> str:
        side = read_sidecar(path)
        if side.get("unit"):
            return str(side["unit"])
        return path.parent.name if path.parent != top else default

    def _files_by_unit(self, top: Path, default: str) -> dict[str, list[Path]]:
        out: dict[str, list[Path]] = {}
        for f in self.list_inputs(top):
            out.setdefault(self._unit_of(f, top, default), []).append(f)
        return out

    def list_inputs(self, in_dir: Path) -> list[Path]:
        """Files in the folder and in one level of unit sub-folders (not processed/)."""
        files = list_signal_files(in_dir)
        if in_dir.is_dir():
            for d in sorted(p for p in in_dir.iterdir() if p.is_dir() and p.name != "processed" and not p.name.startswith(".")):
                files += list_signal_files(d)
        return files

    # ---- train: nothing to train --------------------------------------------
    def train(self, ctx: Context) -> Outcome:
        raise NotSupported("bearing: method v1.0 has no training step (no fault data is used). "
                           "Put each unit's healthy recordings in train/<unit>/ and run `calibrate`.")

    # ---- calibrate: one baseline per unit ------------------------------------
    def calibrate(self, ctx: Context) -> Outcome:
        cfg, c = ctx.config, ctx.config["calibration"]
        by_unit = self._files_by_unit(ctx.layout.train_dir, cfg["default_unit"])
        if not by_unit:
            raise InputError(f"no healthy recordings in {ctx.layout.train_dir} (or its unit sub-folders)")
        units, used, warns, summary, metrics = {}, [], [], [], {"units": {}}
        watch = cfg["verdict"]["watch_fraction"]
        for unit, files in sorted(by_unit.items()):
            uc = self._unit_cfg(cfg, unit)
            geom = resolve_bearing(uc.get("bearing", cfg["bearing"]))
            mode = uc.get("detector", cfg["detector"])
            if mode == "six_line" and geom is None:
                warns.append(f"unit {unit}: detector six_line needs the bearing geometry; using geometry_free")
                mode = "geometry_free"
            keys = list(cfg.get("operating_state_keys") or [])
            X, t0, fid, levels, states, skipped, ufiles = [], [], [], [], [], [], []
            for f in files:
                try:
                    fs, rpm, key = self._file_params(f, cfg, unit)
                    check_speed(rpm, geom, mode == "geometry_free")
                    Xi, ti, _, lv = window_features(load_signal(f, key), fs, rpm, geom, c["max_windows_per_file"],
                                                    bins=mode == "geometry_free")
                    st = self._state(f, rpm, keys)
                except InputError as e:
                    skipped.append(f"{f.name}: {e}"); ctx.log.warning("[%s] skipped %s: %s", unit, f.name, e); continue
                X.append(Xi); t0.append(ti); fid += [len(ufiles)] * len(ti); levels.append(lv); states.append(st)
                ufiles.append(f)
            n_win = len(fid)
            if n_win < c["min_windows"]:
                raise InputError(f"unit {unit}: only {n_win} usable healthy windows (need >= {c['min_windows']}; "
                                 "1 window = 1 s of signal)" + (f". Skipped: {skipped}" if skipped else ""))
            t0, fid = np.concatenate(t0), np.asarray(fid)
            groups = default_groups(fid, t0)
            det_key = "bins" if mode == "geometry_free" else "lines"
            base = fit_baseline(np.vstack([x[det_key] for x in X]), groups, meta={"unit": unit, "features": det_key})
            typ = fit_baseline(np.vstack([x["lines"] for x in X]), groups, meta={"unit": unit, "features": "lines"}) \
                if geom is not None else None
            thr = base.threshold(c["alpha"])
            if not np.isfinite(thr):
                raise InputError(f"unit {unit}: too few windows for a conformal threshold at alpha {c['alpha']}")
            # out-of-sample (cross-fitted) flagged share per healthy file: the honest healthy check
            fracs = {ufiles[i].name: float((base.cal_scores[fid == i] > thr).mean()) for i in range(len(ufiles))}
            rec = np.array([base.cal_scores[fid == i].mean() for i in range(len(ufiles))])
            if len(ufiles) >= 3:
                p_mu, p_sd = float(rec.mean()), float(rec.std(ddof=1)) + 1e-12
                lm = fit_level_model(np.asarray(levels), np.asarray(states))
            else:   # fewer than 3 recordings: spread of the windows stands in (stated in the warning)
                p_mu, p_sd = float(base.cal_scores.mean()), float(base.cal_scores.std(ddof=1)) + 1e-12
                lm = {"coef": [float(np.mean(levels))] + [0.0] * (1 + len(keys)), "resid_sd": 1.0, "conditioned": False}
            h_p = max(MIN_CUSUM_H, _cusum_max((rec - p_mu) / p_sd))
            h_l = max(MIN_CUSUM_H, _cusum_max([level_z(lm, lv, st) for lv, st in zip(levels, states)]))
            units[unit] = {"detector": mode, "geometry": geometry_to_dict(geom), "baseline": base.to_dict(),
                           "typing_baseline": typ.to_dict() if typ is not None else None,
                           "threshold": thr, "physics_rec": [p_mu, p_sd], "level_model": lm, "state_keys": keys,
                           "cusum_h": {"physics": h_p, "level": h_l}, "n_windows": n_win,
                           "n_recordings": len(ufiles), "default_rpm": uc.get("rpm"), "default_fs_hz": uc.get("fs_hz")}
            worst = max(fracs.values())
            metrics["units"][unit] = {"n_recordings": len(ufiles), "n_windows": n_win, "threshold": thr,
                                      "detector": mode, "healthy_flagged_by_file": fracs, "healthy_flagged_max": worst,
                                      "reads_clean": bool(worst < watch), "skipped": skipped}
            used += ufiles
            line = (f"unit {unit}: {len(ufiles)} recording(s), {n_win} windows, detector {mode}"
                    f"{'' if geom is None else ' (' + geom.name + f', BPFO {geom.bpfo:.3f} / BPFI {geom.bpfi:.3f})'}"
                    f", healthy flagged (cross-fitted) worst {worst:.0%} -> {'reads clean' if worst < watch else 'READS ABNORMAL'}")
            summary.append(line)
            if skipped:
                warns.append(f"unit {unit}: {len(skipped)} file(s) skipped: " + "; ".join(skipped[:3]))
            if geom is None:
                warns.append(f"unit {unit}: bearing unknown -> detection only, no fault typing. Give the designation, "
                             "geometry or fault factors if known.")
            if mode == "geometry_free" and n_win < c["recommended_windows"]:
                warns.append(f"unit {unit}: the geometry-free score sets a conservative threshold on short baselines "
                             "(notes/23); a longer baseline will raise its sensitivity")
            if n_win < c["recommended_windows"]:
                warns.append(f"unit {unit}: baseline is {n_win / 60:.0f} min; >= {c['recommended_windows'] / 60:.0f} min of "
                             "healthy operation per operating state is recommended (notes/26)")
            if len(ufiles) < c["min_recordings_warn"]:
                warns.append(f"unit {unit}: only {len(ufiles)} healthy recording(s); the trend statistics use window "
                             "spread instead of recording-to-recording spread -- add recordings from different days/conditions")
            if worst >= watch:
                bad = [n for n, f in fracs.items() if f >= watch]
                warns.append(f"unit {unit}: healthy recording(s) {bad} reach {worst:.0%} flagged against the rest of the "
                             "unit's baseline (cross-fitted). This unit's 'healthy' data is not self-consistent (the "
                             "per-unit failure of notes/14, 17, 24): add healthy data spanning its operating states first.")
        cal = {"method_version": METHOD_VERSION, "alpha": c["alpha"], "verdict": cfg["verdict"],
               "max_windows_per_file": c["max_windows_per_file"], "default_fs_hz": cfg["fs_hz"],
               "default_rpm": cfg["rpm"], "signal_key": cfg["signal_key"], "default_unit": cfg["default_unit"],
               "units": units}
        (ctx.version_dir / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf8")
        metrics["n_units"] = len(units)
        return Outcome("calibrate", metrics, [f"method v{METHOD_VERSION}: {len(units)} unit(s) calibrated"] + summary,
                       used, warns, extra_meta={"needs_calibration": False, "method_version": METHOD_VERSION})

    # ---- inference -----------------------------------------------------
    def promote_check(self, version_dir: Path, meta: dict) -> list[str]:
        p = version_dir / "calibration.json"
        if not p.is_file():
            return ["no calibration.json: run `calibrate`"]
        v = json.loads(p.read_text(encoding="utf8")).get("method_version")
        return [] if v == METHOD_VERSION else [f"calibrated with method {v!r}; this code runs {METHOD_VERSION} -- recalibrate"]

    def load_model(self, version_dir: Path, meta: dict):
        p = version_dir / "calibration.json"
        if not p.is_file():
            raise InputError("this version has no calibration (run `calibrate`)")
        cal = json.loads(p.read_text(encoding="utf8"))
        if cal.get("method_version") != METHOD_VERSION:
            raise InputError(f"model was calibrated with method {cal.get('method_version')!r}, this code is "
                             f"{METHOD_VERSION}: recalibrate (old versions stay available for rollback)")
        units = {}
        for u, d in cal["units"].items():
            typ = Baseline.from_dict(d["typing_baseline"]) if d.get("typing_baseline") else None
            units[u] = (Baseline.from_dict(d["baseline"]), geometry_from_dict(d["geometry"]), d, typ)
        return {"cal": cal, "units": units}

    def _trend(self, ctx: Context, model: dict, unit: str, file: str, pz: float, lz: float, h: dict) -> dict:
        path = ctx.layout.out_dir / TREND_FILE
        state = model.get("_trend_state")
        if state is None:       # read the trend file once per job, then keep the last state per unit in memory
            state = {}
            if path.is_file():
                with path.open(newline="", encoding="utf8") as f:
                    for r in csv.DictReader(f):
                        state[r["unit"]] = {"cusum_physics": float(r["cusum_physics"]), "cusum_level": float(r["cusum_level"])}
            model["_trend_state"] = state
        prev = state.get(unit, {"cusum_physics": 0.0, "cusum_level": 0.0})
        cp = max(0.0, prev["cusum_physics"] + pz - CUSUM_K)
        cl = max(0.0, prev["cusum_level"] + lz - CUSUM_K)
        alarm = bool(cl > h["level"] or cp > h["physics"])
        row = {"time": iso(), "unit": unit, "file": file, "physics_z": round(pz, 4), "level_z": round(lz, 4),
               "cusum_physics": round(cp, 4), "cusum_level": round(cl, 4), "trend_alarm": alarm}
        state[unit] = {"cusum_physics": row["cusum_physics"], "cusum_level": row["cusum_level"]}
        new = not path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=TREND_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        return row

    def infer_one(self, model, path: Path, ctx: Context) -> list[dict]:
        cal = model["cal"]
        unit = self._unit_of(path, ctx.layout.in_dir, cal.get("default_unit", "unit1"))
        if unit not in model["units"]:
            raise InputError(f"unit {unit!r} has no baseline in this model (calibrated units: {sorted(model['units'])}); "
                             "put its healthy recordings in train/<unit>/ and calibrate")
        base, geom, ud, typ = model["units"][unit]
        fb = {"fs_hz": ud.get("default_fs_hz") or cal.get("default_fs_hz"), "rpm": ud.get("default_rpm") or cal.get("default_rpm"),
              "signal_key": cal.get("signal_key")}
        fs, rpm, key = self._file_params(path, ctx.config, unit, fb)
        bins = ud["detector"] == "geometry_free"
        check_speed(rpm, geom, bins)
        signal = load_signal(path, key)
        X, _, total, level = window_features(signal, fs, rpm, geom, cal["max_windows_per_file"], bins=bins)
        iso = {**ctx.config["iso20816"], **(self._unit_cfg(ctx.config, unit).get("iso20816") or {})}
        iso_v = iso_z = None
        if iso.get("group") and iso.get("signal_unit"):
            iso_v = round(velocity_rms_mm_s(signal, fs, iso["signal_unit"], rpm), 3)
            iso_z = zone(iso_v, iso["group"], iso["foundation"])
        s = base.score(X["bins"] if bins else X["lines"])
        thr = ud["threshold"]
        flagged = s > thr
        frac = float(flagged.mean())
        v = cal["verdict"]
        verdict = "faulty" if frac >= v["faulty_fraction"] else "watch" if frac >= v["watch_fraction"] else "healthy"
        fault_type, dominant = "", ""
        if flagged.any():
            if typ is not None:
                lines = X["lines"][flagged]
                t = typ.typing(lines)
                # a hint, not a diagnosis (notes/25: unreliable when another defect type coexists)
                fault_type = "outer race (hint)" if (t == OUTER).mean() >= 0.5 else "inner race (hint)"
                med = np.median(typ.z(lines), 0)
                dominant = f"{LINES[int(np.argmax(med))]} z={med.max():.1f}"
            else:
                med = np.median(base.z(X["bins"][flagged]), 0)
                dominant = f"order {BIN_CENTRES[int(np.argmax(med))]:.2f} z={med.max():.1f} (bearing unknown: no typing)"
        pz = (float(s.mean()) - ud["physics_rec"][0]) / ud["physics_rec"][1]
        lz = level_z(ud["level_model"], level, self._state(path, rpm, ud.get("state_keys") or []))
        tr = self._trend(ctx, model, unit, path.name, pz, lz, ud["cusum_h"])
        msg = f"scored {len(s)} of {total} windows (evenly spaced)" if len(s) < total else ""
        if tr["trend_alarm"]:
            msg = (msg + "; " if msg else "") + "TREND ALARM: sustained rise over this unit's baseline"
        return [{"unit": unit, "detector": ud["detector"], "verdict": verdict, "score": float(np.median(s)),
                 "flagged_fraction": round(frac, 4), "n_windows": int(len(s)), "message": msg, "rpm": rpm, "fs_hz": fs,
                 "threshold": thr, "n_windows_flagged": int(flagged.sum()), "fault_type": fault_type, "dominant": dominant,
                 "physics_z": round(pz, 3), "level_z": round(lz, 3), "cusum_physics": tr["cusum_physics"],
                 "cusum_level": tr["cusum_level"], "trend_alarm": tr["trend_alarm"],
                 "iso_velocity_mm_s": iso_v, "iso_zone": iso_z}]

    def doctor(self, layout, cfg):
        out = []
        try:
            g = resolve_bearing(cfg["bearing"])
            out.append(("ok" if g is not None else "warn", "default bearing",
                        f"{g.name}: BPFO {g.bpfo:.3f}, BPFI {g.bpfi:.3f} (orders)" if g is not None
                        else "none -> geometry-free detection, no typing"))
        except ConfigError as e:
            out.append(("fail", "default bearing", str(e)))
        for u, uc in (cfg.get("units") or {}).items():
            try:
                g = resolve_bearing((uc or {}).get("bearing", cfg["bearing"]))
                out.append(("ok", f"unit {u}", g.name if g is not None else "geometry-free"))
            except ConfigError as e:
                out.append(("fail", f"unit {u}", str(e)))
        if cfg["rpm"] is None:
            out.append(("warn", "rpm", "not set: every file (or unit) then needs `rpm` (shaft speed is required)"))
        if cfg["fs_hz"] is None:
            out.append(("warn", "fs_hz", "not set (each file then needs a sidecar with fs_hz)"))
        return out
