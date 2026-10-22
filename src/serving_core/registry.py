"""Versioned model registry on a mounted volume.

A version is a folder holding the model files plus `meta.json` (who/what/when,
hashes) and `metrics.json` (the report a human reads before approving).
`train`/`calibrate` only ever create *candidates*. The ACTIVE pointer is
changed by `promote` and `rollback` and nothing else -- that is the
human-approval step the deployment plan requires.

Integrity: `promote` refuses a version whose config or artifact hashes no
longer match what was recorded when it was created (a stale or hand-edited
folder), whose report is missing, or that was never fully written.
"""
from __future__ import annotations

import json
from pathlib import Path

from serving_core.util import atomic_write_text, iso, sha256_file, sha256_json, utc_now, write_json


class RegistryError(RuntimeError):
    pass


class Registry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.versions_dir = self.models_dir / "versions"
        self.active_file = self.models_dir / "ACTIVE"
        self.history_file = self.models_dir / "HISTORY.jsonl"

    # ---- versions ------------------------------------------------------
    def new_version_dir(self, config_hash: str, data_hash: str) -> tuple[str, Path]:
        stamp = utc_now().strftime("%Y%m%dT%H%M%S")
        base = f"{stamp}-{sha256_json([config_hash, data_hash])[:6]}"
        # Two calibrations of the same data within one second (seen on the
        # rented box's test run) would otherwise collide; a suffix keeps both
        # and still sorts after the first.
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        for n in range(100):
            version = base if n == 0 else f"{base}-{n}"
            path = self.versions_dir / version
            try:
                path.mkdir()
                return version, path
            except FileExistsError:
                continue
        raise RegistryError(f"could not allocate a version name after {base}")

    def path(self, version: str) -> Path:
        p = self.versions_dir / version
        if not (p / "meta.json").is_file():
            raise RegistryError(f"unknown or incomplete version {version!r} (no meta.json in {p})")
        return p

    def meta(self, version: str) -> dict:
        return json.loads((self.path(version) / "meta.json").read_text(encoding="utf8"))

    def list(self) -> list[dict]:
        out = []
        if self.versions_dir.is_dir():
            for p in sorted(self.versions_dir.iterdir()):
                if (p / "meta.json").is_file():
                    m = json.loads((p / "meta.json").read_text(encoding="utf8"))
                    m["active"] = m["version"] == self.active()
                    out.append(m)
        return out

    def finalize(self, version_dir: Path, meta: dict, config: dict, metrics: dict) -> dict:
        """Write config/metrics/meta LAST, so a crashed run has no meta.json
        and therefore is invisible to list/promote."""
        write_json(version_dir / "config.json", config)
        write_json(version_dir / "metrics.json", metrics)
        meta = dict(meta)
        meta["artifacts"] = {
            p.name: sha256_file(p) for p in sorted(version_dir.iterdir())
            if p.is_file() and p.name not in ("meta.json",)
        }
        meta["config_hash"] = sha256_json(config)
        write_json(version_dir / "meta.json", meta)
        return meta

    # ---- active pointer ------------------------------------------------
    def active(self) -> str | None:
        if self.active_file.is_file():
            v = self.active_file.read_text(encoding="utf8").strip()
            return v or None
        return None

    def verify(self, version: str) -> list[str]:
        """Problems that make `version` unsafe to activate (empty = fine)."""
        problems = []
        try:
            p = self.path(version)
        except RegistryError as e:
            return [str(e)]
        meta = json.loads((p / "meta.json").read_text(encoding="utf8"))
        for need in ("metrics.json", "config.json"):
            if not (p / need).is_file():
                problems.append(f"missing {need}")
        if (p / "config.json").is_file():
            cfg = json.loads((p / "config.json").read_text(encoding="utf8"))
            if sha256_json(cfg) != meta.get("config_hash"):
                problems.append("config.json does not match the recorded config hash")
        for name, digest in meta.get("artifacts", {}).items():
            f = p / name
            if not f.is_file():
                problems.append(f"artifact {name} is missing")
            elif sha256_file(f) != digest:
                problems.append(f"artifact {name} was modified after training (hash mismatch)")
        return problems

    def _log(self, event: dict) -> None:
        event = {"time": iso(), **event}
        with open(self.history_file, "a", encoding="utf8") as f:
            f.write(json.dumps(event) + "\n")

    def promote(self, version: str, by: str = "") -> None:
        problems = self.verify(version)
        if problems:
            raise RegistryError(f"refusing to promote {version}: " + "; ".join(problems))
        prev = self.active()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.active_file, version + "\n")
        self._log({"event": "promote", "version": version, "previous": prev, "by": by})

    def _stack(self) -> list[str]:
        """Replay history into the stack of previously active versions."""
        stack: list[str] = []
        if self.history_file.is_file():
            for line in self.history_file.read_text(encoding="utf8").splitlines():
                ev = json.loads(line)
                if ev["event"] == "promote":
                    if ev.get("previous"):
                        stack.append(ev["previous"])
                elif ev["event"] == "rollback" and stack:
                    stack.pop()
        return stack

    def rollback_target(self) -> str | None:
        stack = self._stack()
        return stack[-1] if stack else None

    def rollback(self, by: str = "") -> str:
        target = self.rollback_target()
        if not target:
            raise RegistryError("nothing to roll back to (no earlier active version in HISTORY)")
        problems = self.verify(target)
        if problems:
            raise RegistryError(f"cannot roll back to {target}: " + "; ".join(problems))
        cur = self.active()
        atomic_write_text(self.active_file, target + "\n")
        self._log({"event": "rollback", "version": target, "previous": cur, "by": by})
        return target
