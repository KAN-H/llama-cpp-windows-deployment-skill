#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q6 probe - run ONE llama-server configuration and measure it end to end.

Phase-2 workhorse for Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp (qwen35 dense 27B) on the
RTX 5060 Ti 16 GiB card.

What it does, in order:
  1. refuse to run if the GPU is already busy (mixed numbers are worse than none)
  2. launch llama-server with a LIST argv (no shell quoting to get wrong)
  3. wait for /health, watching for the process to die first
  4. warm up with a 1-token request, then measure a real decode
  5. read VRAM after the request, parse the log for draft acceptance / layer
     split / n_ctx_slot / warnings, write everything to logs/ + JSON record
  6. terminate the server and wait for VRAM to come back before returning

Measurement conventions (kept identical across arms so numbers are comparable):
  * decode t/s      = the server's own predicted_per_second from response timings
  * prompt          = fixed English paragraph, temperature 0
  * warmup excluded = the first request is thrown away
  * VRAM            = nvidia-smi memory.used right after the measured request

The BSOD caveat (#30302: GSQ-RCO + draft-mtp + Windows CUDA) is why --ctx and
--n-predict can be made TINY: run the smoke test first (--smoke), scale up only
if it survives.

Examples:
    # MTP smoke test - minimal risk, do this FIRST
    .\\.venv\\Scripts\\python.exe plan\\_q6_probe.py --smoke --mtp --tag mtp-smoke

    # no-MTP baseline at 138K, fit auto-layering
    .\\.venv\\Scripts\\python.exe plan\\_q6_probe.py --ctx 141312 --tag base-fit-141k

    # manual ngl + KV variant
    .\\.venv\\Scripts\\python.exe plan\\_q6_probe.py --ctx 141312 --ngl 57 --kv-v q4_0 --tag ngl57-q8kq4v

    # long-prompt arm (uses logs/p1b-text-56k.txt concatenated to --long-extra)
    .\\.venv\\Scripts\\python.exe plan\\_q6_probe.py --ctx 141312 --prompt-file logs\\p1b-text-56k.txt

    # dry: print the argv only
    .\\.venv\\Scripts\\python.exe plan\\_q6_probe.py --dry --mtp

Environment (all optional when the script sits in <llama-cpp-dir>/plan/ and
the launcher registry already remembers chat_dir):
    LLAMA_DIR      llama.cpp deployment dir (llama-server.exe + logs/); default
                   = this script's parent.parent
    LAUNCHER_DIR   launcher data dir (launcher-models.json); default =
                   <LLAMA_DIR>/launcher
    CHAT_DIR       models root (overrides the registry's remembered chat_dir)
    LLAMA_API_KEY  server API key (default: sk-local-001)
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# No machine-specific paths in this file: the defaults reproduce the original
# layout (script inside <llama-cpp-dir>/plan/), and the env vars make it run
# from anywhere (see the Environment block in the docstring).
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
MMPROJ = MODEL_DIR / "mmproj-Qwen3.8-27B-BF16.gguf"
# Local placeholder only - set LLAMA_API_KEY to match your own deployment.
API_KEY = os.environ.get("LLAMA_API_KEY", "sk-local-001")
LOGS = LLAMA_DIR / "logs"
RESULTS = HERE / "_q6_results.json"

PROMPT = ("Explain in a short paragraph why measuring a system is better than "
          "guessing about it, and give one concrete example from computer "
          "performance work.")


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


def free_port(start=18450):
    for p in range(start, start + 50):
        s = socket.socket()
        try:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
        finally:
            s.close()
    return start


def build_argv(args, port):
    a = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(MODEL),
         "--jinja", "-fa", "on", "-np", "1", "-t", str(args.threads),
         "-c", str(args.ctx),
         "--batch-size", str(args.batch), "--ubatch-size", str(args.ub),
         "--cache-type-k", args.kv_k, "--cache-type-v", args.kv_v,
         "--keep", "-1", "--load-mode", args.load_mode,
         "--host", "127.0.0.1", "--port", str(port), "--api-key", API_KEY,
         "--no-warmup"]
    if args.ngl is not None:
        a += ["-ngl", str(args.ngl)]
    else:
        a += ["--fit", "on", "-fitc", str(args.ctx)]
    if args.n_cpu_ffn:
        a += ["--n-cpu-ffn", str(args.n_cpu_ffn)]
    if args.mmproj:
        a += ["--mmproj", str(MMPROJ), "--image-min-tokens", "1024"]
    if args.mtp:
        a += ["--spec-type", "draft-mtp",
              "--spec-draft-n-max", str(args.spec_n_max),
              "--spec-draft-p-min", str(args.spec_p_min)]
    return a


def wait_health(port, proc, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc=%s" % proc.returncode
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:{}/health".format(port), timeout=3) as r:
                if r.status == 200:
                    return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(1.5)
    return False, time.time() - t0, "health timeout"


def chat(port, prompt, max_tokens, timeout_s=1200, temperature=0, no_think=False):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": False, "temperature": temperature,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:{}/v1/chat/completions".format(port), data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data, time.time() - t0


def parse_log(log):
    """Pull the numbers only the server log carries."""
    d = {}
    m = re.search(r"draft acceptance[^\d]*([\d.]+)", log)
    if m:
        d["acceptance"] = float(m.group(1))
    m = re.search(r"mean len[^\d]*([\d.]+)", log)
    if m:
        d["mean_len"] = float(m.group(1))
    m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", log)
    if m:
        d["offloaded_layers"], d["total_layers"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"n_ctx_slot\s*=\s*(\d+)", log)
    if m:
        d["n_ctx_slot"] = int(m.group(1))
    m = re.search(r"n_ctx\s*=\s*(\d+)", log)
    if m:
        d["n_ctx"] = int(m.group(1))
    m = re.search(r"max\s+blk\.(\d+)", log)
    d["max_blk_seen"] = int(m.group(1)) if m else None
    m = re.search(r"graphs reused\s*=\s*(\d+)", log)
    if m:
        d["graphs_reused"] = int(m.group(1))
    d["assert_hits"] = len(re.findall(r"GGML_ASSERT|assert.*failed", log))
    d["cuda_errors"] = len(re.findall(r"CUDA error|cuBLAS error|out of memory", log, re.I))
    d["reprefill"] = len(re.findall(r"forcing full prompt re-processing|"
                                    r"full prompt re-process", log, re.I))
    return d


def run(args):
    if not MODEL.exists():
        print("[X] model not found: %s" % MODEL)
        if CHAT_ROOT is None:
            print("    set CHAT_DIR (models root), or LLAMA_DIR so the launcher "
                  "registry's chat_dir can be read")
        return None
    used0, total = vram()
    print("VRAM before: %s / %s MiB" % (used0, total))
    if used0 is not None and used0 > 1500:
        print("[!] GPU already busy (%d MiB) - stop the other server first" % used0)
        return None

    port = free_port()
    argv = build_argv(args, port)
    if args.dry:
        print("argv:")
        for x in argv:
            print("   %s" % x)
        return None

    LOGS.mkdir(exist_ok=True)
    logfile = LOGS / ("q6-%s.log" % args.tag)
    print("launching (tag=%s port=%d ctx=%d ngl=%s n-cpu-ffn=%s kv=%s/%s batch=%d/%d "
          "mmproj=%s mtp=%s)" % (args.tag, port, args.ctx,
                                 args.ngl if args.ngl is not None else "fit",
                                 args.n_cpu_ffn, args.kv_k, args.kv_v,
                                 args.batch, args.ub, args.mmproj, args.mtp))
    t0 = time.time()
    p = subprocess.Popen(argv, cwd=str(LLAMA_DIR), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    log_lines = []
    import threading

    def drain():
        try:
            for ln in p.stdout:
                log_lines.append(ln.rstrip())
        except Exception:
            pass

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    rec = {"tag": args.tag, "ctx": args.ctx, "ngl": args.ngl,
           "n_cpu_ffn": args.n_cpu_ffn, "kv_k": args.kv_k, "kv_v": args.kv_v,
           "batch": args.batch, "ub": args.ub, "mmproj": bool(args.mmproj),
           "mtp": bool(args.mtp), "n_predict": args.n_predict,
           "load_mode": args.load_mode}
    try:
        ok, secs, why = wait_health(port, p, args.load_timeout)
        rec["load_s"] = round(secs, 1)
        if not ok:
            log = "\n".join(log_lines)
            logfile.write_text(log, encoding="utf-8")
            print("[X] not healthy: %s" % why)
            print("    last lines:\n%s" % "\n".join(log_lines[-25:]))
            rec["error"] = why
            return rec

        # warmup (thrown away), then the measured request
        try:
            chat(port, "hi", 1, timeout_s=300)
        except Exception as e:  # noqa: BLE001
            print("[!] warmup request failed: %s" % e)
        prompt = args.prompt
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
            if args.long_extra > 0:
                prompt = prompt + "\n\n" + prompt[:args.long_extra]
            print("   long prompt: ~%d chars (~%d tokens)" %
                  (len(prompt), len(prompt) // 4))
        data, wall = chat(port, prompt, args.n_predict, no_think=args.no_think)
        t = data.get("timings") or {}
        used1, _ = vram()
        log = "\n".join(log_lines)
        logfile.write_text(log, encoding="utf-8")
        _ch = (data.get("choices") or [{}])[0] or {}
        _ans = (_ch.get("message") or {}).get("content") or ""
        rec.update({
            "answer": _ans[:600],
            "finish": _ch.get("finish_reason"),
            "prompt_n": t.get("prompt_n"),
            "prompt_tps": t.get("prompt_per_second"),
            "n_gen": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "wall_s": round(wall, 1),
            "vram_used_mib": used1,
            "headroom_mib": (total - used1) if used1 else None,
            "no_think": bool(args.no_think),
        })
        rec.update(parse_log(log))
        print("    load={}s prompt={} tok @ {} t/s  decode={} t/s  gen={} tok  "
              "VRAM={} MiB headroom={} MiB  offl={}/{}  acc={}".format(
                  rec.get("load_s"),
                  rec.get("prompt_n"),
                  "{:.0f}".format(rec["prompt_tps"]) if rec.get("prompt_tps") else "?",
                  "{:.1f}".format(rec["decode_tps"]) if rec.get("decode_tps") else "?",
                  rec.get("n_gen"),
                  rec.get("vram_used_mib"),
                  rec.get("headroom_mib"),
                  rec.get("offloaded_layers"), rec.get("total_layers"),
                  rec.get("acceptance", "-")))
        if rec.get("reprefill"):
            print("    [!] 'forcing full prompt re-processing' seen %d time(s)" %
                  rec["reprefill"])
        return rec
    finally:
        try:
            p.terminate()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        # wait for VRAM to come back before the next arm runs
        for _ in range(20):
            u, _t = vram()
            if u is not None and u < 800:
                break
            time.sleep(1.5)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ctx", type=int, default=141312,
                    help="default 141312 = 138K tokens (the hard floor from the task)")
    ap.add_argument("--ngl", type=int, default=None,
                    help="manual GPU layer count; omit to use --fit on -fitc <ctx>")
    ap.add_argument("--n-cpu-ffn", type=int, default=0,
                    help="keep the dense FFN of the first N layers on CPU "
                         "(dense-model analog of n-cpu-moe; untried on this model)")
    ap.add_argument("--kv-k", default="q8_0")
    ap.add_argument("--kv-v", default="q8_0")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--mmproj", action="store_true")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--spec-n-max", type=int, default=3)
    ap.add_argument("--spec-p-min", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=300)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--load-mode", default="none", choices=["none", "mmap", "mmap+mlock"])
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--prompt-file", default="",
                    help="use a file as the prompt (long-context arms)")
    ap.add_argument("--long-extra", type=int, default=0,
                    help="extra chars appended (repeat of the file head) to grow it")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny ctx / tiny output smoke test (BSOD-class checks)")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs enable_thinking=false "
                         "(NIAH: keep the answer out of the thinking chain)")
    ap.add_argument("--dry", action="store_true", help="print argv only")
    args = ap.parse_args()
    if args.smoke:
        args.ctx = 4096
        args.n_predict = 16
        if not args.tag:
            args.tag = "smoke"
    if not args.tag:
        print("[X] --tag is required (names the log + JSON record)")
        return 2

    rec = run(args)
    if rec is None:
        return 2 if not args.dry else 0
    # append to the accumulating results file
    hist = []
    if RESULTS.exists():
        try:
            hist = json.loads(RESULTS.read_text(encoding="utf-8"))
        except Exception:
            hist = []
    hist.append(rec)
    RESULTS.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print("result appended to %s" % RESULTS.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
