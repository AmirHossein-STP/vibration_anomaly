"""Fetch, verify and seal the datasets in registry.yaml -- runs ON THE SERVER.

The server downloads straight from each publisher onto the vault; nothing
passes through the laptop (its uplink is ~500 KB/s). Usually driven through
`remote/box data ...`, which runs this as

    ~/work/venv/bin/python -I remote/datasets/fetch.py --vault /mnt/vault \
        --scratch ~/work/scratch --repo ~/work/<project> <command> ...

Commands:
    list                      what the registry holds and what the vault has
    fetch NAME|all [--jobs N] download, check, (condense), record MANIFEST.json
    verify NAME|all [--quick] [--against DIR]
                              re-hash vault files against MANIFEST.json, and
                              optionally against the manifests committed in git
    rehash NAME               re-record the condensed/ files (after a manual add)
    unseal NAME --commit SHA --confirm WP11
                              extract a sealed archive onto the SSD, logged

Every file is checked against the publisher's checksum where the host offers
one (sha256 on Mendeley/MediaFire, md5 on Zenodo, git blob sha1 on GitHub),
and its sha256 is recorded either way. Downloads resume from `.part` files,
so a rental that ends mid-fetch loses nothing but the file in flight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.yaml"
UA = "Mozilla/5.0 (X11; Linux x86_64) vault-fetch"
# Never fill the vault to the last byte: ext4 degrades badly near full, and
# the sync daemon still needs room for run outputs.
RESERVE_BYTES = 1 * 1024**3


class FetchError(RuntimeError):
    pass


@dataclass
class FileSpec:
    name: str                   # path relative to raw/
    url: str                    # "mediafire:<page>" is resolved just before download
    size: int | None = None
    checksum: str | None = None  # "<algo>:<hex>", algo in sha256 | md5 | gitsha1
    mirrors: tuple[str, ...] = ()  # same file elsewhere; tried in turn when `url` fails


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_json(url: str) -> dict:
    return json.loads(http_get(url))


# --- source resolvers: registry entry -> list of FileSpec ------------------

def resolve_github_tree(src: dict) -> list[FileSpec]:
    tree = http_json(f"https://api.github.com/repos/{src['repo']}/git/trees/{src['ref']}?recursive=1")
    if tree.get("truncated"):
        raise FetchError(f"GitHub tree listing for {src['repo']} was truncated")
    pat = re.compile(src["include"])
    return [
        FileSpec(e["path"],
                 f"https://raw.githubusercontent.com/{src['repo']}/{src['ref']}/{urllib.parse.quote(e['path'])}",
                 e["size"], f"gitsha1:{e['sha']}")
        for e in tree["tree"] if e["type"] == "blob" and pat.search(e["path"])
    ]


def resolve_mendeley(src: dict) -> list[FileSpec]:
    # The public API lists files only for the current version (a ?version=
    # query returns metadata without them), so pin by checking instead.
    meta = http_json(f"https://data.mendeley.com/public-api/datasets/{src['dataset']}")
    if int(meta["version"]) != int(src["version"]):
        raise FetchError(f"Mendeley {src['dataset']}: registry pins version {src['version']}, "
                         f"current is {meta['version']} -- review the new version, then update the registry")
    pat = re.compile(src["include"])
    out = []
    for f in meta["files"]:
        if not pat.search(f["filename"]):
            continue
        cd = f["content_details"]
        out.append(FileSpec(f["filename"], cd["download_url"], cd["size"], f"sha256:{cd['sha256_hash']}"))
    return out


def resolve_zenodo(src: dict) -> list[FileSpec]:
    rec = http_json(f"https://zenodo.org/api/records/{src['record']}")
    pat = re.compile(src["include"])
    return [FileSpec(f["key"], f["links"]["self"], f["size"], f["checksum"])  # checksum is "md5:<hex>"
            for f in rec["files"] if pat.search(f["key"])]


def resolve_mediafire_folder(src: dict) -> list[FileSpec]:
    url = ("https://www.mediafire.com/api/1.5/folder/get_content.php?"
           f"folder_key={src['folder']}&content_type=files&response_format=json")
    files = http_json(url)["response"]["folder_content"]["files"]
    pat = re.compile(src["include"])
    return [FileSpec(f["filename"], "mediafire:" + f["links"]["normal_download"], int(f["size"]), f"sha256:{f['hash']}")
            for f in files if pat.search(f["filename"])]


def resolve_url(src: dict) -> list[FileSpec]:
    return [FileSpec(f["name"], f["url"], f.get("size"), f.get("checksum"), tuple(f.get("mirrors", ())))
            for f in src["files"]]


RESOLVERS = {
    "github_tree": resolve_github_tree,
    "mendeley": resolve_mendeley,
    "zenodo": resolve_zenodo,
    "mediafire_folder": resolve_mediafire_folder,
    "url": resolve_url,
}

MEDIAFIRE_DIRECT = re.compile(r'https?://download\d*\.mediafire\.com/[^"\'\s<>]+')


def direct_url(url: str) -> str:
    """MediaFire hands out short-lived direct links on the file's page, so
    resolve one per download attempt instead of storing it."""
    if not url.startswith("mediafire:"):
        return url
    page = http_get(url[len("mediafire:"):]).decode("utf-8", "replace")
    m = MEDIAFIRE_DIRECT.search(page)
    if not m:
        raise FetchError(f"no direct download link on {url}")
    return m.group(0)


# --- hashing and download ---------------------------------------------------

def hash_file(path: Path) -> dict[str, str]:
    """sha256, md5 and git-blob sha1 in one pass over the file."""
    size = path.stat().st_size
    h = {"sha256": hashlib.sha256(), "md5": hashlib.md5(), "gitsha1": hashlib.sha1(b"blob %d\0" % size)}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for d in h.values():
                d.update(chunk)
    return {k: d.hexdigest() for k, d in h.items()}


def check_publisher(digests: dict[str, str], checksum: str | None, name: str) -> None:
    if not checksum:
        return
    algo, _, want = checksum.partition(":")
    if algo not in digests:
        raise FetchError(f"{name}: unknown checksum algorithm {algo!r}")
    if digests[algo].lower() != want.lower():
        raise FetchError(f"{name}: {algo} mismatch (got {digests[algo]}, publisher says {want})")


def download(spec: FileSpec, dest: Path, attempts: int = 4) -> dict:
    """Fetch one file with resume, check size and checksum, then move it into
    place. Returns the MANIFEST record."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urls = (spec.url, *spec.mirrors)
    used = spec.url
    for attempt in range(1, attempts + 1):
        if spec.size is not None and part.exists() and part.stat().st_size == spec.size:
            break  # finished on a previous attempt, only the move was missing
        if attempt > 1 and len(urls) > 1 and used != urls[(attempt - 1) % len(urls)]:
            part.unlink(missing_ok=True)  # a mirror's bytes may differ; never splice two sources
        used = urls[(attempt - 1) % len(urls)]
        try:
            url = direct_url(used)
        except Exception as exc:  # noqa: BLE001 -- network errors come in many types
            print(f"    {spec.name}: resolve failed ({exc}), attempt {attempt}/{attempts}", flush=True)
            time.sleep(5 * attempt)
            continue
        cmd = ["curl", "-fL", "--silent", "--show-error", "--retry", "3", "--retry-delay", "5",
               "--connect-timeout", "30", "-A", UA, "-C", "-", "-o", str(part), url]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            break
        if rc == 33:  # server refuses ranges: start over
            part.unlink(missing_ok=True)
        print(f"    {spec.name}: curl exit {rc}, attempt {attempt}/{attempts}", flush=True)
        time.sleep(5 * attempt)
    else:
        raise FetchError(f"{spec.name}: download failed after {attempts} attempts")

    got = part.stat().st_size
    if spec.size is not None and got != spec.size:
        part.unlink()
        raise FetchError(f"{spec.name}: size {got}, publisher says {spec.size}")
    digests = hash_file(part)
    try:
        check_publisher(digests, spec.checksum, spec.name)
    except FetchError:
        part.unlink()
        raise
    os.replace(part, dest)
    return {"size": got, "sha256": digests["sha256"], "url": used, "publisher_checksum": spec.checksum}


# --- manifest -----------------------------------------------------------------

def dataset_dir(vault: Path, name: str, cfg: dict) -> Path:
    return vault / "datasets" / name / cfg["version"]


def load_manifest(ddir: Path) -> dict | None:
    p = ddir / "MANIFEST.json"
    return json.loads(p.read_text()) if p.is_file() else None


def save_manifest(ddir: Path, m: dict) -> None:
    ddir.mkdir(parents=True, exist_ok=True)
    tmp = ddir / "MANIFEST.json.tmp"
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, ddir / "MANIFEST.json")


def new_manifest(name: str, cfg: dict) -> dict:
    return {
        "dataset": name, "version": cfg["version"], "role": cfg["role"], "stage": cfg["stage"],
        "license": cfg.get("license"), "citation": cfg.get("citation"), "notes": cfg.get("notes"),
        "status": "partial", "source": None, "raw": {}, "condensed": {},
        "raw_on_vault": cfg["stage"] == "vault", "sealed": False, "history": [],
    }


def safe_relname(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise FetchError(f"refusing unsafe file name from publisher: {name!r}")
    return str(p)


def write_source_md(ddir: Path, m: dict) -> None:
    lines = [f"# {m['dataset']} ({m['version']})", "",
             f"- role: {m['role']}", f"- license: {m.get('license')}", f"- cite: {m.get('citation')}",
             f"- source: `{json.dumps(m.get('source'))}`", f"- status: {m['status']}"]
    if m.get("notes"):
        lines.append(f"- notes: {m['notes']}")
    lines += ["", "Written by remote/datasets/fetch.py. MANIFEST.json has the per-file sizes and sha256."]
    (ddir / "SOURCE.md").write_text("\n".join(lines) + "\n")


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def require_space(path: Path, need: int, what: str) -> None:
    free = free_bytes(path)
    if need + RESERVE_BYTES > free:
        raise FetchError(f"{what}: needs {need / 1e9:.2f} GB (+{RESERVE_BYTES / 1e9:.0f} GB reserve) "
                         f"but {path} has {free / 1e9:.2f} GB free -- refusing to start")


# --- commands -----------------------------------------------------------------

def resolve_sources(name: str, cfg: dict) -> tuple[dict, list[FileSpec]]:
    errors = []
    for src in cfg["sources"]:
        try:
            specs = RESOLVERS[src["kind"]](src)
            if not specs:
                raise FetchError("listing matched no files")
            return src, specs
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src['kind']}: {exc}")
            print(f"  source {src['kind']} unusable ({exc}); trying the next one", flush=True)
    raise FetchError(f"{name}: every source failed: {errors}")


def seal(ddir: Path, raw_dir: Path) -> None:
    for p in raw_dir.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    raw_dir.chmod(0o555)
    (ddir / "SEALED").write_text(
        "SEALED (MASTER_PLAN §4.3). Do not open, list the contents of, or score this data before WP11.\n"
        "Unsealing: fetch.py unseal <name> --commit <frozen method sha> --confirm WP11\n")


def run_condenser(name: str, cfg: dict, ddir: Path, raw_dir: Path, scratch: Path, repo: Path) -> None:
    out_tmp = ddir / "condensed.partial"
    if out_tmp.exists():
        shutil.rmtree(out_tmp)
    work = scratch / name / "work"
    if work.exists():
        shutil.rmtree(work)
    cmd = [sys.executable, str(HERE / f"{cfg['condense']}.py"), "--raw-dir", str(raw_dir),
           "--out", str(out_tmp), "--work", str(work), "--src", str(repo / "src"),
           "--jobs", str(max(1, (os.cpu_count() or 2) - 1))]
    print(f"  condensing: {' '.join(cmd)}", flush=True)
    # cwd = repo, never a data directory (a planted module there would be imported)
    subprocess.run(cmd, check=True, cwd=repo)
    final = ddir / "condensed"
    if final.exists():
        shutil.rmtree(final)
    os.replace(out_tmp, final)
    shutil.rmtree(work, ignore_errors=True)


def record_condensed(ddir: Path, m: dict) -> None:
    root = ddir / "condensed"
    m["condensed"] = {
        str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": hash_file(p)["sha256"]}
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def fetch(name: str, cfg: dict, vault: Path, scratch: Path, repo: Path, jobs: int,
          force: bool = False, keep_raw: bool = False) -> dict:
    ddir = dataset_dir(vault, name, cfg)
    m = load_manifest(ddir) or new_manifest(name, cfg)
    if m["status"] == "complete" and not force:
        print(f"{name}: complete on the vault, skipping (--force to redo)")
        return m

    print(f"== {name} ({cfg['version']}, role={cfg['role']}, stage={cfg['stage']})", flush=True)
    src, specs = resolve_sources(name, cfg)
    if cfg.get("layout") == "flat":
        for s in specs:
            s.name = PurePosixPath(s.name).name
    names = [safe_relname(s.name) for s in specs]
    if len(set(names)) != len(names):
        raise FetchError(f"{name}: duplicate file names after layout={cfg.get('layout')}")
    m["source"] = src

    raw_dir = ddir / "raw" if cfg["stage"] == "vault" else scratch / name / cfg["version"] / "raw"
    if raw_dir.exists() and not os.access(raw_dir, os.W_OK):
        raise FetchError(f"{raw_dir} is read-only (sealed?) -- refusing to write into it")

    def have(s: FileSpec) -> bool:
        rec, p = m["raw"].get(s.name), raw_dir / s.name
        return rec is not None and p.is_file() and p.stat().st_size == rec["size"]

    todo = [s for s in specs if not have(s)]
    raw_total = sum(s.size or 0 for s in specs)
    pending = sum(s.size or 0 for s in todo)
    print(f"  {len(specs)} files, {raw_total / 1e9:.2f} GB; {len(todo)} to download ({pending / 1e9:.2f} GB)", flush=True)
    if cfg["stage"] == "vault":
        require_space(vault, pending, name)
    else:
        require_space(scratch, pending, f"{name} (raw on SSD)")
        require_space(vault, int(cfg["expected_gb"] * 1e9), f"{name} (condensed)")

    failures = []
    t0, done_bytes = time.time(), 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(download, s, raw_dir / s.name): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                m["raw"][s.name] = fut.result()
                done_bytes += m["raw"][s.name]["size"]
                rate = done_bytes / max(time.time() - t0, 1e-6) / 1e6
                print(f"  [{i}/{len(todo)}] ok {s.name} ({rate:.1f} MB/s overall)", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{s.name}: {exc}")
                print(f"  [{i}/{len(todo)}] FAILED {s.name}: {exc}", flush=True)
            save_manifest(ddir, m)
    if failures:
        m["status"] = "partial"
        save_manifest(ddir, m)
        raise FetchError(f"{name}: {len(failures)} files failed (rerun to resume): {failures[:3]}")

    m["status"] = "raw-complete"
    m.setdefault("history", []).append({"at": now(), "by": socket.gethostname(), "event": "raw verified"})
    save_manifest(ddir, m)

    if cfg["role"] == "sealed":
        seal(ddir, raw_dir)
        m["sealed"] = True
    if cfg.get("condense"):
        run_condenser(name, cfg, ddir, raw_dir, scratch, repo)
        record_condensed(ddir, m)
        if cfg["stage"] == "ssd" and not keep_raw:
            shutil.rmtree(raw_dir.parent, ignore_errors=True)
    m["status"] = "complete"
    m["history"].append({"at": now(), "by": socket.gethostname(), "event": "complete"})
    save_manifest(ddir, m)
    write_source_md(ddir, m)
    print(f"{name}: complete -> {ddir}", flush=True)
    return m


def verify(name: str, cfg: dict, vault: Path, quick: bool, against: Path | None) -> list[str]:
    ddir = dataset_dir(vault, name, cfg)
    m = load_manifest(ddir)
    if m is None:
        return [f"{name}: not on the vault"]
    problems = [] if m["status"] == "complete" else [f"{name}: status {m['status']}"]
    groups = [("condensed", ddir / "condensed")]
    if m.get("raw_on_vault"):
        groups.append(("raw", ddir / "raw"))
    for key, root in groups:
        for rel, rec in m.get(key, {}).items():
            p = root / rel
            if not p.is_file():
                problems.append(f"{name}: missing {key}/{rel}")
            elif p.stat().st_size != rec["size"]:
                problems.append(f"{name}: size changed {key}/{rel}")
            elif not quick and hash_file(p)["sha256"] != rec["sha256"]:
                problems.append(f"{name}: sha256 changed {key}/{rel}")
    if against is not None:
        ref_path = against / f"{name}.json"
        if not ref_path.is_file():
            problems.append(f"{name}: no committed manifest at {ref_path}")
        else:
            ref = json.loads(ref_path.read_text())
            for key in ("raw", "condensed"):
                mine = {k: v["sha256"] for k, v in m.get(key, {}).items()}
                theirs = {k: v["sha256"] for k, v in ref.get(key, {}).items()}
                if mine != theirs:
                    diff = sorted(set(mine.items()) ^ set(theirs.items()))
                    problems.append(f"{name}: {key} differs from the committed manifest ({len(diff)} entries)")
    print(f"{name}: {'OK' if not problems else 'PROBLEMS'}"
          + ("" if not problems else "\n  " + "\n  ".join(problems)))
    return problems


def unseal(name: str, cfg: dict, vault: Path, scratch: Path, commit: str) -> None:
    """WP11 only. The vault copy stays sealed; an extracted copy goes to the SSD."""
    sys.path.insert(0, str(HERE))
    from safe_extract import extract_zip

    ddir = dataset_dir(vault, name, cfg)
    if not (ddir / "SEALED").is_file():
        raise FetchError(f"{name} is not sealed")
    dest = scratch / f"{name}_unsealed"
    for p in sorted((ddir / "raw").iterdir()):
        if p.suffix == ".zip":
            extract_zip(p, dest / p.stem)
    with open(ddir / "UNSEAL_LOG", "a") as f:
        f.write(f"{now()} unsealed by {socket.gethostname()} for frozen method commit {commit} -> {dest}\n")
    print(f"unsealed copy at {dest} (logged in {ddir / 'UNSEAL_LOG'})")


def cmd_list(registry: dict, vault: Path) -> None:
    print(f"{'dataset':<13}{'role':<8}{'stage':<7}{'version':<26}{'vault status':<14}{'GB on vault':>11}")
    for name, cfg in registry.items():
        ddir = dataset_dir(vault, name, cfg)
        m = load_manifest(ddir)
        size = sum(p.stat().st_size for p in ddir.rglob("*") if p.is_file()) / 1e9 if ddir.exists() else 0.0
        print(f"{name:<13}{cfg['role']:<8}{cfg['stage']:<7}{cfg['version']:<26}{(m or {}).get('status', '-'):<14}{size:>11.2f}")
    if vault.exists():
        print(f"\nvault free: {free_bytes(vault) / 1e9:.1f} GB")


def load_registry(path: Path = REGISTRY) -> dict:
    return yaml.safe_load(path.read_text())["datasets"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", type=Path, default=Path("/mnt/vault"))
    ap.add_argument("--scratch", type=Path, default=Path.home() / "work" / "scratch")
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="+")
    f.add_argument("--jobs", type=int, default=6)
    f.add_argument("--force", action="store_true")
    f.add_argument("--keep-raw", action="store_true", help="keep stage=ssd raw files on the SSD")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="+")
    v.add_argument("--quick", action="store_true", help="sizes only, no hashing")
    v.add_argument("--against", type=Path, help="directory of committed manifests (remote/datasets/manifests)")
    r = sub.add_parser("rehash")
    r.add_argument("name")
    u = sub.add_parser("unseal")
    u.add_argument("name")
    u.add_argument("--commit", required=True)
    u.add_argument("--confirm", required=True)
    args = ap.parse_args(argv)

    registry = load_registry(args.registry)

    def pick(names: list[str]) -> list[str]:
        if names == ["all"]:
            return list(registry)
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; registry has {list(registry)}")
        return names

    if args.cmd == "list":
        cmd_list(registry, args.vault)
        return 0
    if args.cmd == "fetch":
        failed = []
        for n in pick(args.names):
            try:
                fetch(n, registry[n], args.vault, args.scratch, args.repo, args.jobs, args.force, args.keep_raw)
            except (FetchError, subprocess.CalledProcessError) as exc:
                failed.append(n)
                print(f"!! {n}: {exc}", flush=True)
        if failed:
            print(f"\nFAILED: {failed} (rerun the same command to resume)")
        return 1 if failed else 0
    if args.cmd == "verify":
        problems = [p for n in pick(args.names)
                    for p in verify(n, registry[n], args.vault, args.quick, args.against)]
        return 1 if problems else 0
    if args.cmd == "rehash":
        cfg = registry[args.name]
        ddir = dataset_dir(args.vault, args.name, cfg)
        m = load_manifest(ddir)
        if m is None:
            raise SystemExit(f"{args.name}: no MANIFEST.json on the vault")
        record_condensed(ddir, m)
        m["history"].append({"at": now(), "by": socket.gethostname(), "event": "rehash condensed"})
        save_manifest(ddir, m)
        print(f"{args.name}: {len(m['condensed'])} condensed files recorded")
        return 0
    if args.cmd == "unseal":
        if args.confirm != "WP11":
            raise SystemExit("unsealing is for WP11 only: pass --confirm WP11")
        unseal(args.name, registry[args.name], args.vault, args.scratch, args.commit)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
