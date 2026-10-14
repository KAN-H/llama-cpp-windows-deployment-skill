#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q6 multi-turn stability probe (long-agent proxy).

One llama-server load, then THREE turns on the same slot:

  turn 1  the ~45K-token haystack + "summarize"
  turn 2  appended "go deeper on the middle section"      <- must hit prefix cache
  turn 3  appended "one sentence summary in English"      <- must hit prefix cache

What it verifies (the long-agent failure modes seen in August's Hermes run):
  * prefix cache reuse on turns 2/3: the server should report
    `n_prompt_tokens_cache` near the previous total, and prefill time should
    collapse to the new tokens only - NOT a full re-processing.
  * decode speed stability across turns (context growing 45K -> ~46K).
  * the log must NOT contain "forcing full prompt re-processing" (issue #30357,
    Gated DeltaNet state cannot roll back, so any divergence = full re-prefill).

Usage (script in <llama-cpp-dir>/plan/, or anywhere with LLAMA_DIR set):
    .\\.venv\\Scripts\\python.exe plan\\_q6_multiturn.py [--ngl 64] [--mtp]

Environment: LLAMA_DIR (llama.cpp dir; default = parent.parent),
LAUNCHER_DIR (launcher data; default <LLAMA_DIR>/launcher), CHAT_DIR
(models root; else the registry's chat_dir), LLAMA_API_KEY (default
sk-local-001).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or HERE.parent)
LAUNCHER_DIR = Path(os.environ.get("LAUNCHER_DIR") or (LLAMA_DIR / "launcher"))
DIRNAME = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"


def _chat_root():
    """CHAT_DIR, else the chat_dir the launcher registry remembers."""
    v = os.environ.get("CHAT_DIR")
    if v:
        return Path(v)
    try:
        reg = json.loads(
            (LAUNCHER_DIR / "launcher-models.json").read_text(encoding="utf-8"))
        cd = reg.get("chat_dir")
    except Exception:
        cd = None
    return Path(cd) if cd else None


CHAT_ROOT = _chat_root()
MODEL_DIR = (CHAT_ROOT / DIRNAME) if CHAT_ROOT else Path("<models-dir>") / DIRNAME
MODEL = MODEL_DIR / "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
# Local placeholder only - set LLAMA_API_KEY to match your own deployment.
API_KEY = os.environ.get("LLAMA_API_KEY", "sk-local-001")
LOGS = LLAMA_DIR / "logs"
HAYSTACK = LOGS / "q6-long56k.txt"   # ~45K-token material (matches the main ladder)


def vram():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = [int(x) for x in out.splitlines()[0].split(",")]
        return used, total
    except Exception:
        return None, None


def free_port(start=18470):
    for p in range(start, start + 40):
        s = socket.socket()
        try:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
        finally:
            s.close()
    return start


def chat(port, messages, max_tokens, timeout_s=1200):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "stream": False, "temperature": 0}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:{}/v1/chat/completions".format(port), data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=141312)
    ap.add_argument("--ngl", type=int, default=64)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--nmax", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    if CHAT_ROOT is None:
        print("[X] chat dir unknown - set CHAT_DIR, or LLAMA_DIR so the "
              "launcher registry's chat_dir can be read")
        return 2
    if not MODEL.is_file():
        print("[X] model not found: %s" % MODEL)
        return 2

    used0, total = vram()
    print("VRAM before: %s / %s MiB" % (used0, total))
    if used0 and used0 > 1500:
        print("[!] GPU busy - stop the other server first")
        return 2

    text = HAYSTACK.read_text(encoding="utf-8", errors="replace")
    port = free_port()
    argv = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(MODEL),
            "--jinja", "-fa", "on", "-np", "1", "-t", str(args.threads),
            "-c", str(args.ctx), "-ngl", str(args.ngl),
            "--batch-size", "256", "--ubatch-size", "256",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--keep", "-1", "--load-mode", "none",
            "--host", "127.0.0.1", "--port", str(port), "--api-key", API_KEY,
            "--no-warmup", "--metrics"]
    if args.mtp:
        argv += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(args.nmax)]
    log_lines = []
    import threading
    print("launching port=%d ngl=%d mtp=%s" % (port, args.ngl, bool(args.mtp)))
    p = subprocess.Popen(argv, cwd=str(LLAMA_DIR), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")

    def drain():
        try:
            for ln in p.stdout:
                log_lines.append(ln.rstrip())
        except Exception:
            pass

    threading.Thread(target=drain, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            print("[X] server died rc=%s" % p.returncode)
            print("\n".join(log_lines[-20:]))
            return 2
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:{}/health".format(port), timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(1.5)
    print("healthy at %.1fs" % (time.time() - t0))
    time.sleep(2)

    turns = [
        ("T1 summarize", [{"role": "user",
                           "content": text + "\n\nSummarize the main topic "
                           "of the text above in one paragraph."}], 200),
        ("T2 go deeper", None, 200),   # appended below
        ("T3 one-liner", None, 100),
    ]
    msgs = None
    results = []
    try:
        for name, m, npred in turns:
            if m is None:
                # append the previous answer + the new ask to the SAME history
                prev = results[-1]["answer"]
                msgs = msgs + [
                    {"role": "assistant", "content": prev},
                    {"role": "user",
                     "content": ("Go one level deeper on the middle section. "
                                 "Keep it under 100 words.") if name.startswith("T2")
                     else "Now give me a single-sentence summary in English."}]
            else:
                msgs = m
            t1 = time.time()
            data = chat(port, msgs, npred)
            wall = time.time() - t1
            tm = data.get("timings") or {}
            ans = ((data.get("choices") or [{}])[0].get("message") or {}) \
                .get("content") or ""
            rec = {"turn": name, "wall_s": round(wall, 1),
                   "prompt_n": tm.get("prompt_n"),
                   "cached_n": tm.get("prompt_n_cached") or tm.get("cache_n"),
                   "prompt_tps": round(tm.get("prompt_per_second") or 0, 1),
                   "decode_tps": round(tm.get("predicted_per_second") or 0, 2),
                   "gen": tm.get("predicted_n"),
                   "answer": ans[:220]}
            used1, _ = vram()
            rec["vram_mib"] = used1
            results.append(rec)
            print("== {name} == prompt={p} cached={c} prefill={pp} t/s "
                  "decode={d} t/s gen={g} wall={w}s".format(
                      name=name, p=rec["prompt_n"], c=rec["cached_n"],
                      pp=rec["prompt_tps"], d=rec["decode_tps"],
                      g=rec["gen"], w=rec["wall_s"]))
        log = "\n".join(log_lines)
        reprefill = log.count("forcing full prompt re-processing")
        print("reprefill events in log: %d" % reprefill)
        out = Path(__file__).resolve().parent / "_q6_multiturn.json"
        out.write_text(json.dumps({"turns": results, "reprefill": reprefill},
                                  indent=2, ensure_ascii=False),
                       encoding="utf-8")
        (LOGS / "q6-multiturn.log").write_text(log, encoding="utf-8")
        print("written to %s" % out)
        return 0
    finally:
        try:
            p.terminate()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        for _ in range(20):
            u, _t = vram()
            if u is not None and u < 800:
                break
            time.sleep(1.5)


if __name__ == "__main__":
    sys.exit(main())
