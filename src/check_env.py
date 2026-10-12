"""Compare installed versions of the packages that matter with a lock file from the pilot.

Prints one line per package and exits 1 if any differs, so a changed Kaggle image is noticed
before a main run starts (the pinned requirements files should make this pass).

    python -m src.check_env --lock requirements-generation.lock
    python -m src.check_env --lock requirements-retrieval.lock --report-only results/v03_hp/raw/check_env.json   (v0.3)
"""
from __future__ import annotations

import argparse
import re
from importlib import metadata

KEY = ["torch", "transformers", "sentence-transformers", "tokenizers", "huggingface_hub", "faiss-cpu", "bm25s",
       "numpy", "pyarrow", "vllm", "triton"]


def parse_lock(path: str) -> dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def installed(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def compare(lock: dict[str, str], packages: list[str]) -> list[tuple[str, str | None, str | None, bool]]:
    rows = []
    for p in packages:
        want = lock.get(p.lower().replace("_", "-"))
        if want is None:
            continue
        have = installed(p)
        rows.append((p, want, have, have == want))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--report-only", metavar="OUT_JSON", default=None,
                    help="v0.3 (addendum v0.3.1): write the comparison to OUT_JSON and exit 0 even if versions differ")
    args = ap.parse_args(argv)
    rows = compare(parse_lock(args.lock), KEY)
    for p, want, have, ok in rows:
        print(f"{'OK ' if ok else 'DIFF'} {p:24s} lock {want:20s} installed {have}")
    bad = [r for r in rows if not r[3]]
    print(f"{len(rows) - len(bad)}/{len(rows)} key packages match {args.lock}")
    if args.report_only:
        import json
        import sys
        from pathlib import Path
        out = Path(args.report_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"lock": args.lock, "python": sys.version.split()[0],
                                   "rows": [{"package": p, "lock": w, "installed": h, "match": ok} for p, w, h, ok in rows],
                                   "n_diff": len(bad)}, indent=1))
        print(f"report-only: {len(bad)} difference(s) recorded in {out} (deviation, addendum v0.3.1)")
        return 0
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
