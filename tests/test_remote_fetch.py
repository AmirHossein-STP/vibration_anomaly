"""remote/datasets/fetch.py and safe_extract.py on tiny local files.

No network: downloads use file:// URLs (curl handles them like http,
including resume). Covers what protects the vault: checksums, resume,
the free-space refusal, sealing, and archive path traversal.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

DATASETS = Path(__file__).resolve().parents[1] / "remote" / "datasets"
pytestmark = pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DATASETS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch")
safe_extract = _load("safe_extract")


def _src_file(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / "publisher" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


def _cfg(files: list[dict], role: str = "dev", expected_gb: float = 0.0001) -> dict:
    return {"version": "v1", "role": role, "stage": "vault", "expected_gb": expected_gb,
            "sources": [{"kind": "url", "files": files}]}


def test_registry_entries_are_complete():
    reg = fetch.load_registry()
    assert {"cwru", "paderborn", "xjtu_sy", "ottawa_2018", "ottawa_2023", "mfpt", "ims"} <= set(reg)
    for name, cfg in reg.items():
        for key in ("version", "role", "stage", "expected_gb", "sources", "license", "citation"):
            assert key in cfg, f"{name} lacks {key}"
        assert cfg["role"] in ("dev", "sealed")
        assert cfg["stage"] in ("vault", "ssd", "stream")
        assert all(s["kind"] in fetch.RESOLVERS for s in cfg["sources"])
        if cfg["stage"] == "stream":
            assert cfg.get("group") and cfg.get("condense"), f"{name}: a stream needs a group regex and a condenser"
        if cfg["stage"] in ("ssd", "stream"):
            assert cfg.get("condense"), f"{name}: an ssd-staged dataset must be condensed onto the vault"
            assert (DATASETS / f"{cfg['condense']}.py").is_file()
    assert reg["ims"]["role"] == "sealed"


def test_hash_file_matches_git_blob_sha1(tmp_path):
    p = _src_file(tmp_path, "a.bin", b"hello\n")
    d = fetch.hash_file(p)
    assert d["sha256"] == hashlib.sha256(b"hello\n").hexdigest()
    assert d["gitsha1"] == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object` of "hello\n"


def test_fetch_records_manifest_and_skips_when_complete(tmp_path):
    payload = b"x" * 5000
    src = _src_file(tmp_path, "data.bin", payload)
    cfg = _cfg([{"name": "data.bin", "url": src.as_uri(), "size": len(payload),
                 "checksum": "md5:" + hashlib.md5(payload).hexdigest()}])
    vault = tmp_path / "vault"
    m = fetch.fetch("toy", cfg, vault, tmp_path / "scratch", tmp_path, jobs=2)
    assert m["status"] == "complete"
    assert m["raw"]["data.bin"]["sha256"] == hashlib.sha256(payload).hexdigest()
    on_disk = json.loads((vault / "datasets/toy/v1/MANIFEST.json").read_text())
    assert on_disk["raw"] == m["raw"]
    assert (vault / "datasets/toy/v1/raw/data.bin").read_bytes() == payload
    src.unlink()  # a second fetch must not touch the network
    assert fetch.fetch("toy", cfg, vault, tmp_path / "scratch", tmp_path, jobs=2)["status"] == "complete"
    assert fetch.verify("toy", cfg, vault, quick=False, against=None) == []


def test_checksum_mismatch_is_rejected_and_not_kept(tmp_path):
    src = _src_file(tmp_path, "data.bin", b"real bytes")
    cfg = _cfg([{"name": "data.bin", "url": src.as_uri(), "checksum": "sha256:" + "0" * 64}])
    with pytest.raises(fetch.FetchError):
        fetch.fetch("toy", cfg, tmp_path / "vault", tmp_path / "scratch", tmp_path, jobs=1)
    raw = tmp_path / "vault/datasets/toy/v1/raw"
    assert not (raw / "data.bin").exists() and not (raw / "data.bin.part").exists()
    m = json.loads((tmp_path / "vault/datasets/toy/v1/MANIFEST.json").read_text())
    assert m["status"] == "partial"


def test_download_resumes_from_part(tmp_path):
    payload = bytes(range(256)) * 40
    src = _src_file(tmp_path, "big.bin", payload)
    dest = tmp_path / "out" / "big.bin"
    dest.parent.mkdir()
    (dest.parent / "big.bin.part").write_bytes(payload[:3000])  # a rental ended mid-file
    spec = fetch.FileSpec("big.bin", src.as_uri(), len(payload), "sha256:" + hashlib.sha256(payload).hexdigest())
    rec = fetch.download(spec, dest)
    assert dest.read_bytes() == payload and rec["size"] == len(payload)


def test_mirror_is_used_when_primary_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    payload = b"mirror copy"
    src = _src_file(tmp_path, "m.bin", payload)
    spec = fetch.FileSpec("m.bin", (tmp_path / "missing.bin").as_uri(), len(payload), None, (src.as_uri(),))
    rec = fetch.download(spec, tmp_path / "out" / "m.bin")
    assert rec["url"] == src.as_uri()


def test_free_space_refusal(tmp_path):
    src = _src_file(tmp_path, "data.bin", b"tiny")
    cfg = _cfg([{"name": "data.bin", "url": src.as_uri(), "size": 10**15}])  # a petabyte
    with pytest.raises(fetch.FetchError, match="refusing to start"):
        fetch.fetch("toy", cfg, tmp_path / "vault", tmp_path / "scratch", tmp_path, jobs=1)


def test_unsafe_publisher_names_are_refused(tmp_path):
    src = _src_file(tmp_path, "data.bin", b"tiny")
    cfg = _cfg([{"name": "../escape.bin", "url": src.as_uri()}])
    with pytest.raises(fetch.FetchError, match="unsafe"):
        fetch.fetch("toy", cfg, tmp_path / "vault", tmp_path / "scratch", tmp_path, jobs=1)


def test_sealed_dataset_is_read_only_and_refuses_refetch(tmp_path):
    payload = b"sealed exam"
    src = _src_file(tmp_path, "exam.zip", payload)
    cfg = _cfg([{"name": "exam.zip", "url": src.as_uri()}], role="sealed")
    vault = tmp_path / "vault"
    m = fetch.fetch("exam", cfg, vault, tmp_path / "scratch", tmp_path, jobs=1)
    ddir = vault / "datasets/exam/v1"
    assert m["sealed"] and (ddir / "SEALED").is_file()
    raw = ddir / "raw"
    try:
        assert not (raw.stat().st_mode & 0o222) and not ((raw / "exam.zip").stat().st_mode & 0o222)
        with pytest.raises(fetch.FetchError, match="read-only"):
            fetch.fetch("exam", cfg, vault, tmp_path / "scratch", tmp_path, jobs=1, force=True)
    finally:
        raw.chmod(0o755)  # let pytest clean up


def test_against_committed_manifest_detects_drift(tmp_path):
    payload = b"abc"
    src = _src_file(tmp_path, "d.bin", payload)
    cfg = _cfg([{"name": "d.bin", "url": src.as_uri()}])
    vault = tmp_path / "vault"
    m = fetch.fetch("toy", cfg, vault, tmp_path / "scratch", tmp_path, jobs=1)
    committed = tmp_path / "manifests"
    committed.mkdir()
    (committed / "toy.json").write_text(json.dumps(m))
    assert fetch.verify("toy", cfg, vault, quick=True, against=committed) == []
    m["raw"]["d.bin"]["sha256"] = "f" * 64
    (committed / "toy.json").write_text(json.dumps(m))
    assert fetch.verify("toy", cfg, vault, quick=True, against=committed)


def test_zip_traversal_is_refused(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("ok.txt", "fine")
        zf.writestr("../../evil.txt", "escaped")
    with pytest.raises(safe_extract.UnsafeArchive):
        safe_extract.extract_zip(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_needs_a_fresh_directory(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("a/b.txt", "fine")
    out = tmp_path / "out"
    out.mkdir()
    (out / "leftover").write_text("x")
    with pytest.raises(safe_extract.UnsafeArchive, match="not empty"):
        safe_extract.extract_zip(good, out)
    safe_extract.extract_zip(good, tmp_path / "fresh")
    assert (tmp_path / "fresh/a/b.txt").read_text() == "fine"


def test_member_name_checks():
    for name in ["/etc/passwd", "a/../../b", "C:\\x", "..\\up"]:
        with pytest.raises(safe_extract.UnsafeArchive):
            safe_extract.check_member_names([name], Path("x.rar"))
    safe_extract.check_member_names(["K001/N09_M07_F10_K001_1.mat", "dir/file..name.mat"], Path("x.rar"))
