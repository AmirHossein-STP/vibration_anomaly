"""The one command-line interface, identical for every service.

    python -m serving_core <service> [--data-root D] [--models-root M] <command> ...

Commands: init, doctor, infer, train, calibrate, models, show, promote, rollback.
Exit codes: 0 ok | 1 finished but some inputs failed / a check failed | 2 could not run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path

import yaml

from serving_core import __version__
from serving_core.layout import Layout
from serving_core.registry import Registry, RegistryError
from serving_core.service import ConfigError, Context, NotSupported, Outcome, Service
from serving_core.services import SERVICES, get_service
from serving_core.signals import InputError
from serving_core.util import current_user, iso, make_logger, sha256_file, sha256_json, utc_now, write_json

IMAGE_TAG = os.environ.get("SERVICE_IMAGE_TAG", "dev")


# ───────────────────────── config ─────────────────────────
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(service: Service, layout: Layout) -> dict:
    cfg = service.default_config()
    if layout.config_file.is_file():
        try:
            user = yaml.safe_load(layout.config_file.read_text(encoding="utf8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{layout.config_file} is not valid YAML: {e}") from e
        if not isinstance(user, dict):
            raise ConfigError(f"{layout.config_file} must hold a mapping at the top level")
        unknown = set(user) - set(cfg)
        if unknown:
            raise ConfigError(f"{layout.config_file}: unknown setting(s) {sorted(unknown)} (known: {sorted(cfg)})")
        cfg = deep_merge(cfg, user)
    problems = service.validate_config(cfg)
    if problems:
        raise ConfigError(f"{layout.config_file}: " + "; ".join(problems))
    return cfg


# ───────────────────────── jobs ─────────────────────────
class Job:
    def __init__(self, layout: Layout, kind: str):
        self.kind = kind
        self.id = f"{kind}-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.dir = layout.out_dir / utc_now().strftime("%Y-%m-%d") / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = make_logger(self.id, self.dir / "run.log")
        self.started = iso()


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})


def _context(service: Service, layout: Layout, cfg: dict, job: Job, args, reg: Registry) -> Context:
    active = reg.active()
    return Context(
        layout=layout, config=cfg, log=job.log, args=args, active_version=active,
        active_dir=(reg.path(active) if active else None), job_id=job.id,
    )


# ───────────────────────── commands ─────────────────────────
def cmd_init(service, layout, args) -> int:
    layout.ensure()
    example = layout.config_dir / "service.example.yaml"
    example.write_text(service.example_config_text(), encoding="utf8")
    created = False
    if not layout.config_file.is_file():
        shutil.copy(example, layout.config_file)
        created = True
    print(f"Folders ready under {layout.base} and {layout.models_dir}")
    print(f"  config:  {layout.config_file}" + (" (created from defaults -- EDIT IT)" if created else " (kept, not overwritten)"))
    print(f"  example: {example} (always rewritten; shows every setting)")
    return 0


def cmd_doctor(service, layout, args) -> int:
    checks: list[tuple[str, str, str]] = []

    def add(status, name, detail=""):
        checks.append((status, name, detail))

    for label, p in (("in", layout.in_dir), ("train", layout.train_dir), ("out", layout.out_dir),
                     ("config", layout.config_dir), ("models", layout.models_dir)):
        if not p.is_dir():
            add("fail", f"folder {label}", f"{p} does not exist -- run `init`")
            continue
        try:
            probe = p / f".write_test_{uuid.uuid4().hex[:6]}"
            probe.write_text("x"); probe.unlink()
            add("ok", f"folder {label}", str(p))
        except OSError as e:
            add("fail", f"folder {label}", f"{p} is not writable: {e}")
    try:
        free = shutil.disk_usage(layout.base if layout.base.exists() else layout.data_root).free / 1e9
        add("ok" if free >= 2 else "warn", "disk space", f"{free:.1f} GB free")
    except OSError:
        add("warn", "disk space", "could not be read")
    cfg = None
    try:
        cfg = load_config(service, layout)
        add("ok", "config", str(layout.config_file) if layout.config_file.is_file() else "defaults (no service.yaml)")
    except ConfigError as e:
        add("fail", "config", str(e))
    reg = Registry(layout.models_dir)
    active = reg.active()
    if not active:
        add("warn", "active model", "none yet -- `train` (and `calibrate`), then `promote`")
    else:
        probs = reg.verify(active)
        if probs:
            add("fail", "active model", f"{active}: " + "; ".join(probs))
        else:
            try:
                service.load_model(reg.path(active), reg.meta(active))
                add("ok", "active model", f"{active} loads and its hashes match")
            except Exception as e:
                add("fail", "active model", f"{active} does not load: {e}")
    if cfg is not None:
        checks.extend(service.doctor(layout, cfg))
    add("ok", "software", f"serving_core {__version__}, image tag {IMAGE_TAG}")
    mark = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    for s, n, d in checks:
        print(f"{mark[s]} {n}: {d}")
    return 1 if any(s == "fail" for s, _, _ in checks) else 0


def cmd_infer(service, layout, args) -> int:
    layout.ensure()
    cfg = load_config(service, layout)
    reg = Registry(layout.models_dir)
    version = args.model or reg.active()
    if not version:
        print("No active model. Create one with `train` (and `calibrate` if the service needs it), then `promote <version>`.",
              file=sys.stderr)
        return 2
    problems = reg.verify(version)
    if problems:
        print(f"Model {version} failed its integrity check: " + "; ".join(problems), file=sys.stderr)
        return 2
    inputs = service.list_inputs(layout.in_dir)
    if not inputs:
        print(f"No input files in {layout.in_dir}. Nothing to do.")
        return 0

    job = Job(layout, "infer")
    ctx = _context(service, layout, cfg, job, args, reg)
    meta = reg.meta(version)
    model = service.load_model(reg.path(version), meta)
    job.log.info("service=%s model=%s inputs=%d image=%s", service.name, version, len(inputs), IMAGE_TAG)

    rows: list[dict] = []
    ok_files: list[Path] = []
    for path in inputs:
        try:
            got = service.infer_one(model, path, ctx)
            for r in got:
                r.setdefault("file", path.name)
                r.setdefault("status", "ok")
            rows.extend(got)
            ok_files.append(path)
            job.log.info("%s -> %s", path.name, "; ".join(f"{r.get('verdict')}" for r in got))
        except InputError as e:
            rows.append({"file": path.name, "status": "error", "verdict": "", "message": str(e)})
            job.log.warning("%s: REJECTED: %s", path.name, e)
        except Exception as e:  # a bug must not lose the other files' results
            rows.append({"file": path.name, "status": "error", "verdict": "", "message": f"internal error: {e}"})
            job.log.error("%s: internal error\n%s", path.name, traceback.format_exc())
    for r in rows:
        r["model_version"] = version
        r["job_id"] = job.id

    n_err = sum(1 for r in rows if r["status"] == "error")
    result = {
        "job_id": job.id, "service": service.name, "kind": "infer",
        "started_at": job.started, "finished_at": iso(),
        "model_version": version, "model_created_at": meta.get("created_at"),
        "serving_core_version": __version__, "image_tag": IMAGE_TAG,
        "n_inputs": len(inputs), "n_error": n_err, "n_rows": len(rows), "rows": rows,
    }
    write_json(job.dir / "results.json", result)
    _write_csv(job.dir / "results.csv", service.columns, rows)
    write_json(job.dir / "manifest.json", {
        "job_id": job.id, "model_version": version, "model_config_hash": meta.get("config_hash"),
        "service_config_hash": sha256_json(cfg),
        "inputs": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in inputs},
    })
    _append_score_log(layout, job, version, rows)

    if args.archive and ok_files:
        dest = layout.in_dir / "processed" / job.id
        dest.mkdir(parents=True, exist_ok=True)
        for p in ok_files:
            shutil.move(str(p), dest / p.name)
            for side in layout.in_dir.glob(p.name + ".meta.*"):
                shutil.move(str(side), dest / side.name)
        job.log.info("archived %d processed file(s) to %s", len(ok_files), dest)

    print(f"job {job.id}: {len(inputs) - n_err}/{len(inputs)} files ok, model {version}")
    print(f"results: {job.dir}/results.csv  (and results.json, manifest.json, run.log)")
    for r in rows:
        if r["status"] == "error":
            print(f"  REJECTED {r['file']}: {r['message']}")
    return 1 if n_err else 0


def _append_score_log(layout: Layout, job: Job, version: str, rows: list[dict]) -> None:
    """One line per infer job in out/score_log.csv -- raw material for the
    daily drift check (a rising healthy score, or 'everything flagged')."""
    import numpy as np

    ok = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("score"), (int, float))]
    path = layout.out_dir / "score_log.csv"
    new = not path.exists()
    scores = np.array([r["score"] for r in ok], dtype=float) if ok else np.array([])
    flagged = [r for r in ok if r.get("verdict") not in ("healthy", "", None)]
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "job_id", "model_version", "n_ok", "n_error", "score_mean", "score_p95", "n_not_healthy"])
        w.writerow([iso(), job.id, version, len(ok), sum(1 for r in rows if r["status"] == "error"),
                    f"{scores.mean():.6g}" if len(scores) else "", f"{np.percentile(scores, 95):.6g}" if len(scores) else "",
                    len(flagged)])


def _make_candidate(service, layout, args, kind: str) -> int:
    layout.ensure()
    cfg = load_config(service, layout)
    reg = Registry(layout.models_dir)
    job = Job(layout, kind)
    ctx = _context(service, layout, cfg, job, args, reg)
    config_hash = sha256_json(cfg)
    version, vdir = reg.new_version_dir(config_hash, "")
    ctx.version_dir = vdir
    job.log.info("%s start: service=%s candidate=%s", kind, service.name, version)
    try:
        outcome: Outcome = service.train(ctx) if kind == "train" else service.calibrate(ctx)
    except NotSupported as e:
        shutil.rmtree(vdir, ignore_errors=True)
        print(str(e), file=sys.stderr)
        return 2
    except (InputError, ConfigError) as e:
        shutil.rmtree(vdir, ignore_errors=True)
        job.log.error("%s failed: %s", kind, e)
        print(f"{kind} failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        shutil.rmtree(vdir, ignore_errors=True)
        job.log.error("%s crashed\n%s", kind, traceback.format_exc())
        print(f"{kind} crashed: {e}\n(details: {job.dir}/run.log)", file=sys.stderr)
        return 2

    input_hashes = {p.name: sha256_file(p) for p in outcome.inputs}
    meta = {
        "version": version, "service": service.name, "kind": outcome.kind, "status": "candidate",
        "created_at": iso(), "created_by": current_user(), "parent": outcome.parent,
        "data_hash": sha256_json(input_hashes), "n_input_files": len(input_hashes),
        "serving_core_version": __version__, "image_tag": IMAGE_TAG,
        "summary": outcome.summary, "warnings": outcome.warnings, **outcome.extra_meta,
    }
    reg.finalize(vdir, meta, cfg, outcome.metrics)
    write_json(job.dir / f"{kind}_report.json", {"version": version, "summary": outcome.summary,
                                                  "warnings": outcome.warnings, "metrics": outcome.metrics})
    print(f"\nCandidate model: {version}  ({outcome.kind})")
    for line in outcome.summary:
        print("  " + line)
    for w in outcome.warnings:
        print("  WARNING: " + w)
    print(f"\nNot active yet. Review it, then:  promote {version}")
    return 0


def cmd_train(service, layout, args) -> int:
    return _make_candidate(service, layout, args, "train")


def cmd_calibrate(service, layout, args) -> int:
    return _make_candidate(service, layout, args, "calibrate")


def cmd_models(service, layout, args) -> int:
    reg = Registry(layout.models_dir)
    versions = reg.list()
    if not versions:
        print("No model versions yet.")
        return 0
    print(f"{'':2}{'version':24}{'kind':10}{'created':22}summary")
    for m in versions:
        first = (m.get("summary") or [""])[0]
        print(f"{'* ' if m['active'] else '  '}{m['version']:24}{m['kind']:10}{m['created_at']:22}{first}")
    print("\n* = active")
    return 0


def cmd_show(service, layout, args) -> int:
    reg = Registry(layout.models_dir)
    p = reg.path(args.version)
    meta = reg.meta(args.version)
    print(json.dumps({k: v for k, v in meta.items() if k != "artifacts"}, indent=2, ensure_ascii=False))
    print("\nmetrics.json:")
    print((p / "metrics.json").read_text(encoding="utf8"))
    probs = reg.verify(args.version)
    print("integrity: " + ("OK" if not probs else "; ".join(probs)))
    return 0


def _confirm(prompt_word: str, args) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("Refusing to continue without a terminal. Re-run with --yes to confirm.", file=sys.stderr)
        return False
    return input(f"Type '{prompt_word}' to confirm: ").strip() == prompt_word


def cmd_promote(service, layout, args) -> int:
    reg = Registry(layout.models_dir)
    try:
        meta = reg.meta(args.version)
        probs = reg.verify(args.version) + service.promote_check(reg.path(args.version), meta)
        print(f"Version {args.version}  ({meta['kind']}, created {meta['created_at']})")
        for line in meta.get("summary", []):
            print("  " + line)
        for w in meta.get("warnings", []):
            print("  WARNING: " + w)
        print(f"  currently active: {reg.active() or 'none'}")
        if probs:
            print("Integrity check FAILED: " + "; ".join(probs), file=sys.stderr)
            return 2
        if not _confirm("promote", args):
            print("Cancelled.")
            return 2
        reg.promote(args.version, by=current_user())
    except RegistryError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Active model is now {args.version}")
    return 0


def cmd_rollback(service, layout, args) -> int:
    reg = Registry(layout.models_dir)
    target = reg.rollback_target()
    if not target:
        print("Nothing to roll back to.", file=sys.stderr)
        return 2
    print(f"Active: {reg.active()}  ->  would become: {target}")
    if not _confirm("rollback", args):
        print("Cancelled.")
        return 2
    try:
        reg.rollback(by=current_user())
    except RegistryError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Active model is now {target}")
    return 0


# ───────────────────────── entry point ─────────────────────────
def build_parser(service: Service) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"serving_core {service.name}", description=service.description)
    p.add_argument("--data-root", default=None, help="default: $SERVING_DATA_ROOT or /data")
    p.add_argument("--models-root", default=None, help="default: $SERVING_MODELS_ROOT or /models")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create folders and a starter config/service.yaml")
    sub.add_parser("doctor", help="check folders, config, active model")
    pi = sub.add_parser("infer", help="score every file in in/ with the ACTIVE model")
    pi.add_argument("--model", default=None, help="score with this version instead of the active one")
    pi.add_argument("--archive", action="store_true", help="move successfully scored files to in/processed/<job>/")
    pt = sub.add_parser("train", help="train a CANDIDATE model from train/ (never changes the active model)")
    service.add_train_args(pt)
    pc = sub.add_parser("calibrate", help="build a CANDIDATE with a threshold set from healthy files in train/")
    service.add_calibrate_args(pc)
    sub.add_parser("models", help="list versions")
    ps = sub.add_parser("show", help="show a version's report"); ps.add_argument("version")
    pp = sub.add_parser("promote", help="make a version ACTIVE (the human approval step)")
    pp.add_argument("version"); pp.add_argument("--yes", action="store_true")
    pr = sub.add_parser("rollback", help="go back to the previously active version")
    pr.add_argument("--yes", action="store_true")
    return p


COMMANDS = {
    "init": cmd_init, "doctor": cmd_doctor, "infer": cmd_infer, "train": cmd_train,
    "calibrate": cmd_calibrate, "models": cmd_models, "show": cmd_show,
    "promote": cmd_promote, "rollback": cmd_rollback,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in SERVICES:
        print("usage: python -m serving_core <service> <command> [options]\n"
              f"services: {', '.join(SERVICES)}\n"
              "commands: init doctor infer train calibrate models show promote rollback\n"
              "Try:      python -m serving_core <service> --help")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    service = get_service(argv[0])
    args = build_parser(service).parse_args(argv[1:])
    layout = Layout.from_env(service.name, args.data_root, args.models_root)
    try:
        return COMMANDS[args.command](service, layout, args)
    except ConfigError as e:
        print(f"Configuration problem: {e}", file=sys.stderr)
        return 2
    except RegistryError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
