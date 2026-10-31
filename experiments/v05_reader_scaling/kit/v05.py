"""Phase 4 run kit: reader scaling of the transfer ratio (design/addendum_v0.5.md).

One file, Python 3.10+, runs on Windows, macOS or Linux. Commands (run them in this folder):

    python v05.py check                 # Python packages, Ollama, GPU
    python v05.py prepare               # download the two pinned corpora, verify, build all prompts (once)
    python v05.py run qwen3:1.7b-q4_K_M --plan core
    python v05.py run qwen3:4b-q4_K_M   --plan core
    python v05.py run qwen3:8b-q4_K_M   --plan core
    python v05.py run qwen3:14b-q4_K_M  --plan 14b
    python v05.py run gemma3:12b-it-q4_K_M --plan large   # Part B (second model family), addendum v0.5.1
    python v05.py run qwen2.5:7b-instruct-q4_K_M --plan bridge
    python v05.py determinism           # rerun 100 prompts with qwen3:8b, compare
    python v05.py gemini --model <id>   # optional anchor; needs GEMINI_API_KEY in the environment
    python v05.py status                # progress of every run
    python v05.py pack                  # zip everything to send back

Every run can be stopped (Ctrl+C, shutdown) and started again with the same command; finished items are skipped.
No API key is ever written to disk by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
INP = KIT / "inputs"
WORK = KIT / "work"            # corpus downloads (can be deleted after prepare)
PROMPTS = KIT / "prompts"
RES = KIT / "results"
VERSION = (KIT / "VERSION").read_text().strip() if (KIT / "VERSION").exists() else "dev"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
GEMINI = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

# ---- prompts: verbatim copy of src/generation/prompts.py (Appendix A); hashes checked below ----
A1_SYSTEM = ("You are a question answering assistant. Answer the question using the passages provided. "
             "If the passages do not contain the answer, give your best short answer anyway. "
             "Reply with the answer only, as a short phrase of a few words, with no explanation.")
A2_SYSTEM = ("You are a question answering assistant. "
             "Reply with the answer only, as a short phrase of a few words, with no explanation.")
PROMPT_HASH = {"A1": "c50efcc15cc6"}


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


assert _h(A1_SYSTEM) == PROMPT_HASH["A1"], "prompt A1 changed"
PROMPT_HASH["A2"] = _h(A2_SYSTEM)


def messages(question: str, passages: list[tuple[str, str]], variant: str) -> list[dict]:
    if variant == "A2":
        assert not passages
        return [{"role": "system", "content": A2_SYSTEM}, {"role": "user", "content": f"Question: {question}\nAnswer:"}]
    assert variant == "A1" and passages
    body = "\n\n".join(f"[{i}] {t}\n{x}" for i, (t, x) in enumerate(passages, start=1))
    return [{"role": "system", "content": A1_SYSTEM},
            {"role": "user", "content": f"Passages:\n{body}\n\nQuestion: {question}\nAnswer:"}]


# ---- plans (addendum v0.5) ----
ALL = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "GOLD", "CB"]
PLANS = {
    "core": {"sample": "all", "conds": ALL},
    "14b": {"sample": "first1000", "conds": ["R1", "R4", "R5", "GOLD", "CB"], "fallback": "first500",
            "fallback_after": 200, "fallback_median_s": 8.0},
    "bridge": {"sample": "first500", "conds": ["R1", "R5"], "datasets": ["nq"]},
}
PLANS["large"] = PLANS["14b"]          # addendum v0.5.1: same plan for gemma3:12b
GEMINI_ORDER = [["R1", "R5"], ["R4"], ["GOLD"], ["CB"]]
OPTIONS = {"temperature": 0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0, "seed": 0, "num_ctx": 4096,
           "num_predict": 32, "stop": ["\n"]}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
THINK_MODELS: dict[str, bool] = {}
KIT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # a line cut by a power loss: ignored, the item is redone
                pass
    return out


def repair_tail(p: Path) -> None:
    """Cut a last line left incomplete by a power loss or kill, so new records start on a fresh line."""
    if not p.exists() or p.stat().st_size == 0:
        return
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    with open(p, "r+b") as f:
        f.truncate(cut)


def safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def log(msg: str):
    RES.mkdir(exist_ok=True)
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(RES / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def http(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def verify_inputs():
    man = json.loads((INP / "MANIFEST.json").read_text())
    for name, h in man.items():
        if sha_file(INP / name) != h:
            sys.exit(f"STOP: inputs/{name} differs from MANIFEST.json. Use the kit exactly as received.")


# ---------------- check ----------------
def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi not available ({e.__class__.__name__})"


def cmd_check(_):
    print("Kit version:", VERSION)
    ref = (KIT / "KIT_SHA256.txt").read_text().split()[0] if (KIT / "KIT_SHA256.txt").exists() else None
    if ref and ref != KIT_SHA:
        print("WARNING: v05.py has been modified. Use the kit exactly as received, or tell the author.")
    else:
        print("Kit file: unmodified")
    print("Python:", sys.version.split()[0], platform.platform())
    ok = True
    for mod in ("pyarrow", "huggingface_hub"):
        try:
            __import__(mod)
            print(f"  {mod}: ok")
        except ImportError:
            ok = False
            print(f"  {mod}: MISSING  ->  python -m pip install {mod}")
    print("GPU:", gpu_info())
    try:
        v = http("GET", f"{OLLAMA}/api/version", timeout=10)["version"]
        print("Ollama:", v)
        major, minor = (int(x) for x in re.findall(r"\d+", v)[:2])
        if (major, minor) < (0, 9):
            ok = False
            print("  Ollama is too old (needs 0.9 or newer for think=false). Update it.")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("Ollama: NOT REACHABLE at", OLLAMA, f"({e.__class__.__name__}). Start the Ollama app.")
    verify_inputs()
    print("Inputs: verified against MANIFEST.json")
    print("RESULT:", "ready" if ok else "fix the items marked above")
    if not ok:
        sys.exit(1)


# ---------------- prepare ----------------
def cmd_prepare(args):
    verify_inputs()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download

    spec = json.loads((INP / "corpus_spec.json").read_text())
    PROMPTS.mkdir(exist_ok=True)
    meta = {"kit_version": VERSION, "time_utc": now(), "datasets": {}}
    for ds, s in spec.items():
        done = PROMPTS / f"prompts_{ds}.jsonl"
        if done.exists() and (PROMPTS / f"prompts_{ds}.ok").exists():
            log(f"prepare {ds}: already done")
            meta["datasets"][ds] = json.loads((PROMPTS / f"prompts_{ds}.ok").read_text())
            continue
        info = HfApi().dataset_info(s["repo"], revision=s["revision"])
        if info.sha != s["revision"]:
            sys.exit(f"STOP: {s['repo']} resolved to {info.sha}, expected {s['revision']}")
        local = WORK / ds
        log(f"prepare {ds}: downloading {s['repo']}@{s['revision'][:7]} (corpus files only; a few GB)")
        snapshot_download(repo_id=s["repo"], repo_type="dataset", revision=s["revision"], local_dir=str(local),
                          allow_patterns=s["allow_patterns"])
        files = sorted((local / "corpus").glob("*.parquet"))
        if not files:
            sys.exit(f"STOP: no corpus parquet files in {local / 'corpus'}")
        need = pa.array(s["needed_pids"], type=pa.string())
        text: dict[str, tuple[str, str]] = {}
        for f in files:
            t = pq.read_table(f, columns=["_id", "title", "text"])
            ids = pc.cast(t.column("_id"), pa.string())
            sub = t.filter(pc.is_in(ids, value_set=need))
            for i, a, b in zip(pc.cast(sub.column("_id"), pa.string()).to_pylist(), sub.column("title").to_pylist(),
                               sub.column("text").to_pylist()):
                text[i] = (a or "", b or "")
        missing = set(s["needed_pids"]) - set(text)
        if missing:
            sys.exit(f"STOP: {len(missing)} passages missing from the corpus, e.g. {sorted(missing)[:3]}")
        if ds == "nq":
            chk = json.loads((INP / "spotcheck_nq.json").read_text())
            bad = [p for p, h in chk.items()
                   if hashlib.sha256((text[p][0] + "\x1f" + text[p][1]).encode()).hexdigest() != h]
            if bad:
                sys.exit(f"STOP: {len(bad)} of {len(chk)} spot-check passages differ from the committed texts")
            log(f"prepare nq: {len(chk)} spot-check passages identical to the committed texts")
        n = 0
        with open(PROMPTS / f"passages_{ds}.jsonl", "w", encoding="utf-8") as fp:
            for p in sorted(text):
                fp.write(json.dumps({"pid": p, "title": text[p][0], "text": text[p][1]}, ensure_ascii=False) + "\n")
        with open(done, "w", encoding="utf-8") as fo:
            for it in jl(INP / f"items_{ds}.jsonl"):
                msgs = messages(it["question"], [text[p] for p in it["pids"]], it["prompt"])
                fo.write(json.dumps({"item_id": f"{ds}|{it['condition']}|{it['qid']}", "dataset": ds,
                                     "condition": it["condition"], "qid": it["qid"], "prompt": it["prompt"],
                                     "prompt_hash": PROMPT_HASH[it["prompt"]], "messages": msgs},
                                    ensure_ascii=False) + "\n")
                n += 1
        rec = {"repo": s["repo"], "revision": s["revision"], "n_prompts": n, "n_passages": len(text),
               "prompts_sha256": sha_file(done), "passages_sha256": sha_file(PROMPTS / f"passages_{ds}.jsonl"),
               "corpus_files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha_file(f)} for f in files]}
        (PROMPTS / f"prompts_{ds}.ok").write_text(json.dumps(rec, indent=1))
        meta["datasets"][ds] = rec
        log(f"prepare {ds}: {n} prompts, {len(text)} passages")
    RES.mkdir(exist_ok=True)
    (RES / "prepare_meta.json").write_text(json.dumps(meta, indent=1))
    log("prepare: done. You may delete the 'work' folder to free disk space.")


# ---------------- run (Ollama) ----------------
def load_prompts() -> dict[str, list[dict]]:
    out = {}
    for ds in ("nq", "hp"):
        p = PROMPTS / f"prompts_{ds}.jsonl"
        if not (PROMPTS / f"prompts_{ds}.ok").exists():
            sys.exit("STOP: run 'python v05.py prepare' first.")
        out[ds] = jl(p)
    return out


def plan_items(plan: str, sample: str | None = None) -> list[dict]:
    pl = PLANS[plan]
    samples = json.loads((INP / "samples.json").read_text())
    prompts = load_prompts()
    items = []
    for ds in pl.get("datasets", ["nq", "hp"]):
        by = {(p["condition"], p["qid"]): p for p in prompts[ds]}
        for q in samples[f"{ds}_{sample or pl['sample']}"]:          # query-major: every query gets all its conditions
            for c in pl["conds"]:
                items.append(by[(c, q)])
    return items


def ollama_meta(model: str) -> dict:
    tags = http("GET", f"{OLLAMA}/api/tags")["models"]
    hit = [m for m in tags if m["name"] == model or m.get("model") == model]
    if not hit:
        sys.exit(f"STOP: model {model} is not pulled. Run:  ollama pull {model}")
    show = http("POST", f"{OLLAMA}/api/show", {"model": model})
    caps = show.get("capabilities") or []
    THINK_MODELS[model] = "thinking" in caps or model.startswith("qwen3")
    return {"model": model, "digest": hit[0].get("digest"), "size": hit[0].get("size"),
            "details": show.get("details"), "parameters": show.get("parameters"), "template": show.get("template"),
            "capabilities": caps, "ollama_version": http("GET", f"{OLLAMA}/api/version")["version"], "gpu": gpu_info(),
            "platform": platform.platform(), "kit_version": VERSION, "kit_sha256": KIT_SHA, "options": OPTIONS,
            "think_sent": (False if THINK_MODELS[model] else None)}


def call_ollama(model: str, msgs: list[dict]) -> dict:
    t0 = time.perf_counter()
    body = {"model": model, "messages": msgs, "stream": False, "options": OPTIONS, "keep_alive": "30m"}
    if THINK_MODELS.get(model, True):          # think=false only for models that can think (Qwen3)
        body["think"] = False
    r = http("POST", f"{OLLAMA}/api/chat", body)
    lat = time.perf_counter() - t0
    raw = r.get("message", {}).get("content", "")
    clean = THINK_RE.sub("", raw)
    clean = clean.split("<think>")[0]
    out = clean.strip().split("\n")[0].strip()
    return {"output_raw": raw, "output": out, "think_removed": int(clean != raw), "done_reason": r.get("done_reason"),
            "thinking_present": int(bool(r.get("message", {}).get("thinking"))),
            "prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count"), "latency_s": round(lat, 4)}


def run_items(model: str, items: list[dict], out_path: Path, label: str, pl: dict | None = None,
              sample_switch: tuple | None = None) -> None:
    repair_tail(out_path)
    done = {r["item_id"] for r in jl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    log(f"{label}: {len(items)} items, {len(done & {i['item_id'] for i in items})} already done, {len(todo)} to run")
    lat, n_new, checked = [], 0, False
    n_prev = len(done & {i["item_id"] for i in items})
    with open(out_path, "a", encoding="utf-8") as f:
        for it in todo:
            try:
                r = call_ollama(model, it["messages"])
            except urllib.error.URLError as e:
                log(f"{label}: Ollama not reachable ({e}). Start Ollama and run the same command again.")
                sys.exit(1)
            rec = {"item_id": it["item_id"], "dataset": it["dataset"], "condition": it["condition"], "qid": it["qid"],
                   "prompt": it["prompt"], "prompt_hash": it["prompt_hash"], "model": model, "time_utc": now(), **r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
            lat.append(r["latency_s"])
            if n_new % 25 == 0:
                f.flush()
                os.fsync(f.fileno())
            if n_new <= 5:
                print(f"   example {n_new}: {it['dataset']} {it['condition']} -> {r['output']!r}", flush=True)
            if not checked and n_new >= 50:
                checked = True
                f.flush()
                recent = jl(out_path)[-50:]
                bad = sum(1 for x in recent if not x["output"] or "<think" in x["output_raw"] or x["thinking_present"])
                if bad > 12:
                    log(f"{label}: STOP: {bad} of the first 50 outputs are empty or contain thinking text. "
                        "Send results/log.txt and the results file to Claude.")
                    sys.exit(2)
            if n_new % 500 == 0:
                med = statistics.median(lat[-500:])
                eta_h = med * (len(todo) - n_new) / 3600
                log(f"{label}: {n_new + len(done)}/{len(items)} done, median {med:.2f} s/item, about {eta_h:.1f} h left")
            if sample_switch and n_prev + n_new == sample_switch[0]:
                first = [x["latency_s"] for x in jl(out_path)][: sample_switch[0]]
                med = statistics.median(first)
                decision = "first500" if med > sample_switch[1] else "first1000"
                out_path.with_suffix(".decision.json").write_text(json.dumps(
                    {"median_latency_first_prompts": med, "n": len(first), "threshold_s": sample_switch[1],
                     "sample": decision, "time_utc": now()}))
                log(f"{label}: median latency {med:.2f} s over the first {len(first)} prompts -> sample {decision}")
                if decision == "first500":
                    out_path.with_suffix(".fallback").write_text("first500")
                    return
    log(f"{label}: finished ({n_new} new items)")


def cmd_run(args):
    verify_inputs()
    model, plan = args.model, args.plan
    RES.mkdir(exist_ok=True)
    out = RES / f"gen__{safe(model)}__{plan}.jsonl"
    meta_p = RES / f"gen__{safe(model)}__{plan}.meta.json"
    meta = ollama_meta(model)
    ref = (KIT / "KIT_SHA256.txt").read_text().split()[0] if (KIT / "KIT_SHA256.txt").exists() else None
    if ref and ref != KIT_SHA:
        log("WARNING: v05.py differs from KIT_SHA256.txt (kit modified); recorded in the meta file")
    old = json.loads(meta_p.read_text()) if meta_p.exists() else None
    if old and old.get("digest") != meta["digest"]:
        sys.exit(f"STOP: model digest changed since this run started ({old.get('digest')} -> {meta['digest']}). "
                 "Do not re-pull models in the middle of a run; tell Claude.")
    meta.setdefault("started_utc", (old or {}).get("started_utc", now()))
    meta["plan"] = plan
    meta_p.write_text(json.dumps(meta, indent=1))
    pl = PLANS[plan]
    sample = pl["sample"]
    if pl.get("fallback") and out.with_suffix(".fallback").exists():
        sample = out.with_suffix(".fallback").read_text().strip()
    items = plan_items(plan, sample)
    decided = out.with_suffix(".decision.json").exists()
    switch = ((pl["fallback_after"], pl["fallback_median_s"])
              if pl.get("fallback") and sample == pl["sample"] and not decided else None)
    run_items(model, items, out, f"{model} [{plan}, {sample}]", pl, switch)
    if switch and out.with_suffix(".fallback").exists():
        items = plan_items(plan, "first500")
        run_items(model, items, out, f"{model} [{plan}, first500]")
    meta["finished_utc"] = now()
    meta_p.write_text(json.dumps(meta, indent=1))


def cmd_determinism(args):
    model = args.model
    RES.mkdir(exist_ok=True)
    (RES / f"det__{safe(model)}.meta.json").write_text(json.dumps(ollama_meta(model), indent=1))
    samples = json.loads((INP / "samples.json").read_text())
    prompts = {(p["condition"], p["qid"]): p for p in load_prompts()["nq"]}
    items = [dict(prompts[("R5", q)], item_id="det|" + prompts[("R5", q)]["item_id"]) for q in samples["nq_all"][:100]]
    out = RES / f"det__{safe(model)}.jsonl"
    run_items(model, items, out, f"determinism {model}")
    first = {r["item_id"]: r["output"] for r in jl(RES / f"gen__{safe(model)}__core.jsonl")}
    rer = {r["item_id"][4:]: r["output"] for r in jl(out)}
    same = sum(1 for k, v in rer.items() if first.get(k) == v)
    log(f"determinism {model}: {same} of {len(rer)} outputs identical to the first run")


# ---------------- Gemini (optional anchor) ----------------
def gemini_call(model: str, key: str, msgs: list[dict], thinking_off: bool) -> dict:
    gc = {"temperature": 0, "stopSequences": ["\n"], "maxOutputTokens": 32 if thinking_off else 2048}
    if thinking_off:
        gc["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"systemInstruction": {"parts": [{"text": msgs[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": msgs[1]["content"]}]}], "generationConfig": gc}
    t0 = time.perf_counter()
    r = http("POST", f"{GEMINI}/models/{model}:generateContent", body,
             headers={"x-goog-api-key": key}, timeout=120)
    lat = time.perf_counter() - t0
    cand = (r.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", []) or []
    raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return {"output_raw": raw, "output": raw.strip().split("\n")[0].strip(), "finish_reason": cand.get("finishReason"),
            "model_version": r.get("modelVersion"), "usage": r.get("usageMetadata"), "latency_s": round(lat, 4),
            "thinking_off": thinking_off}


def cmd_gemini(args):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("Set the key first (it is not saved anywhere):\n  Windows PowerShell:  $env:GEMINI_API_KEY=\"your-key\"\n"
                 "  Command Prompt:      set GEMINI_API_KEY=your-key")
    if args.list_models:
        r = http("GET", f"{GEMINI}/models?pageSize=200",
                 headers={"x-goog-api-key": key})
        for m in r.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]:
                print(m["name"].split("/")[-1], "-", m.get("displayName"))
        return
    if not args.model:
        sys.exit("Give the model id:  python v05.py gemini --model <id>   (list them with --list-models)")
    verify_inputs()
    samples = json.loads((INP / "samples.json").read_text())
    prompts = load_prompts()
    RES.mkdir(exist_ok=True)
    out = RES / f"gemini__{safe(args.model)}.jsonl"
    repair_tail(out)
    done = {r["item_id"] for r in jl(out)} if out.exists() else set()
    thinking_off = True
    gap = 60.0 / args.rpm
    with open(out, "a", encoding="utf-8") as f:
        for group in GEMINI_ORDER:
            for ds in ("nq", "hp"):
                by = {(p["condition"], p["qid"]): p for p in prompts[ds]}
                for q in samples[f"{ds}_first500"]:
                    for c in group:
                        it = by[(c, q)]
                        if it["item_id"] in done:
                            continue
                        for attempt in range(6):
                            t_start = time.time()
                            try:
                                r = gemini_call(args.model, key, it["messages"], thinking_off)
                                break
                            except urllib.error.HTTPError as e:
                                msg = e.read().decode("utf-8", "replace")[:500]
                                if e.code == 400 and thinking_off and "thinking" in msg.lower():
                                    thinking_off = False
                                    log("gemini: this model does not accept thinkingBudget 0; continuing with default "
                                        "thinking (recorded per item)")
                                    continue
                                if e.code in (429, 500, 503):
                                    wait = 60 * (attempt + 1)
                                    log(f"gemini: HTTP {e.code}, waiting {wait} s")
                                    time.sleep(wait)
                                    continue
                                log(f"gemini: HTTP {e.code}: {msg}")
                                sys.exit(1)
                            except urllib.error.URLError as e:
                                log(f"gemini: network error {e}; waiting 60 s")
                                time.sleep(60)
                        else:
                            log("gemini: still refused after several tries; the daily free quota is probably used up. "
                                "Run the same command again tomorrow; finished items are kept.")
                            sys.exit(0)
                        rec = {"item_id": it["item_id"], "dataset": ds, "condition": c, "qid": q, "prompt": it["prompt"],
                               "prompt_hash": it["prompt_hash"], "model": args.model, "time_utc": now(), **r}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        done.add(it["item_id"])
                        if len(done) % 100 == 0:
                            log(f"gemini: {len(done)} items done")
                        time.sleep(max(0.0, gap - (time.time() - t_start)))
            log(f"gemini: group {group} finished")
    log("gemini: all groups finished")


# ---------------- status / pack ----------------
def cmd_status(_):
    if not RES.exists():
        print("No results yet.")
        return
    for p in sorted(RES.glob("*.jsonl")):
        rows = jl(p)
        lat = [r["latency_s"] for r in rows if "latency_s" in r]
        print(f"{p.name}: {len(rows)} items" + (f", median {statistics.median(lat):.2f} s/item" if lat else ""))


def cmd_pack(_):
    RES.mkdir(exist_ok=True)
    files = sorted(RES.glob("*")) + [p for p in PROMPTS.glob("*.ok")] + [p for p in PROMPTS.glob("passages_*.jsonl")]
    files = [p for p in files if p.is_file() and p.suffix != ".zip"]
    man = {str(p.relative_to(KIT)).replace("\\", "/"): sha_file(p) for p in files}
    (RES / "MANIFEST_results.json").write_text(json.dumps(man, indent=1))
    name = KIT / f"v05_results_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + [RES / "MANIFEST_results.json"]:
            z.write(p, str(p.relative_to(KIT)))
    print("Wrote", name, f"({name.stat().st_size / 1e6:.1f} MB). Send this file to Claude.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--plan", required=True, choices=sorted(PLANS))
    d = sub.add_parser("determinism")
    d.add_argument("--model", default="qwen3:8b-q4_K_M")
    g = sub.add_parser("gemini")
    g.add_argument("--model")
    g.add_argument("--list-models", action="store_true")
    g.add_argument("--rpm", type=float, default=8.0, help="requests per minute (free tier is limited)")
    sub.add_parser("status")
    sub.add_parser("pack")
    a = ap.parse_args()
    {"check": cmd_check, "prepare": cmd_prepare, "run": cmd_run, "determinism": cmd_determinism, "gemini": cmd_gemini,
     "status": cmd_status, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
