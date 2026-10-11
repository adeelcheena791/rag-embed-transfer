"""Pinned-SHA downloader for the five Hugging Face dataset repos (design v0.2 §2).

For each dataset in cfg['datasets']:
  1. ask the Hub for the commit at the pinned revision and require it to equal the pin;
  2. snapshot_download exactly that revision (allow_patterns restrict files);
  3. write a manifest with size + sha256 of every downloaded file.

Pilot step P1 passes if every dataset's resolved sha equals its pin.

    python -m src.data.download --config configs/pilot.yaml [--only nq_open,scifact]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, write_new_json
from src.common.log import get_logger


class ShaMismatch(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_manifest(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            out.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return out


def download_one(name: str, spec: dict, raw_dir: Path, api, snapshot_fn) -> dict:
    repo, pin = spec["repo"], spec["revision"]
    resolved_now = False
    if pin == "resolve":
        # addendum v0.3: datasets added in v0.3 are pinned at first download; the resolved sha is recorded
        # in the manifest and download.jsonl and then used for every later session (config `revision_file`).
        pin, resolved_now = api.dataset_info(repo).sha, True
    if not pin or len(pin) != 40:
        raise ValueError(f"{name}: revision must be a full 40-char commit sha, got {pin!r}")
    info = api.dataset_info(repo, revision=pin)
    if info.sha != pin:
        raise ShaMismatch(f"{name}: hub resolved {info.sha} for pin {pin}")
    local = raw_dir / name
    snapshot_fn(repo_id=repo, repo_type="dataset", revision=pin, local_dir=str(local),
                allow_patterns=spec.get("allow_patterns"))
    files = file_manifest(local)
    if not files:
        raise RuntimeError(f"{name}: no files downloaded (check allow_patterns)")
    return {"name": name, "repo": repo, "pinned_revision": pin, "resolved_revision": info.sha,
            "sha_match": info.sha == pin, "resolved_at_first_use": resolved_now, "n_files": len(files), "files": files}


def main(argv=None, api=None, snapshot_fn=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default="", help="comma-separated dataset keys")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("download", cfg, run_id)
    if api is None or snapshot_fn is None:
        from huggingface_hub import HfApi, snapshot_download
        api, snapshot_fn = HfApi(), snapshot_download

    keys = [k for k in args.only.split(",") if k] or list(cfg["datasets"])
    raw_dir = cpath(cfg, "data_raw")
    sink = AppendOnlyJSONL(cpath(cfg, "results_raw") / "download.jsonl")
    results, ok = [], True
    for k in keys:
        log.info("downloading %s @ %s", cfg["datasets"][k]["repo"], cfg["datasets"][k]["revision"])
        try:
            r = download_one(k, cfg["datasets"][k], raw_dir, api, snapshot_fn)
        except Exception as e:
            log.error("FAILED %s: %s", k, e)
            r = {"name": k, "error": repr(e), "sha_match": False}
        ok &= bool(r.get("sha_match"))
        r["run_id"] = run_id
        sink.append({kk: vv for kk, vv in r.items() if kk != "files"})
        results.append(r)
    write_new_json(raw_dir / f"manifest_{run_id}.json", {"run_id": run_id, "datasets": results})
    log.info("P1 %s: all pinned shas matched = %s", "PASS" if ok else "FAIL", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
