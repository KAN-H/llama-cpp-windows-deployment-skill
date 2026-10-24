#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrency (n_parallel / --kv-unified) probe.

One llama-server launch per `--np` value. For that launch it measures:

  1. `n_ctx_slot` / `n_ctx` reported by the server  <- proves whether the KV
     pool is SHARED (unified: slot gets the whole `-c`) or SPLIT (non-unified:
     slot only gets `-c / np`). This is the single most important number here.
  2. concurrency ladder c = 1..np: c simultaneous requests with disjoint
     prompts, temp 0. Reports
        aggregate t/s = total generated tokens / wall clock
        per-stream t/s = mean(token_i / wall_i)
        slowdown      = single-stream t/s / per-stream t/s at that level
  3. a deterministic QUALITY check: the same greedy prompt is answered at c=1
     before and after the ladder; both replies are stored in the record so
     arms can be diffed for exact equality ("no accuracy loss").
  4. VRAM after the ladder, plus every CUDA error / GGML_ASSERT in the log.

Concurrency levels above `np` are impossible (there are only `np` slots), so
run one arm per np value and compare arms.

Usage (from <llama-cpp-dir>):
    .\\.venv\\Scripts\\python.exe plan\\_np_probe.py --np 1 --tag q38-np1
    .\\.venv\\Scripts\\python.exe plan\\_np_probe.py --np 2 --unified --tag q38-np2-u
    .\\.venv\\Scripts\\python.exe plan\\_np_probe.py --np 2 --no-unified --tag q38-np2-nu
    .\\.venv\\Scripts\\python.exe plan\\_np_probe.py --np 4 --unified --model-dir Qwen3.6-... --n-cpu-moe 27 --ctx 260000 --tag q36-np4-u

Environment: LLAMA_DIR / LAUNCHER_DIR / CHAT_DIR / LLAMA_API_KEY (same as
_q6_probe.py).
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or HERE.parent)
LAUNCHER_DIR = Path(os.environ.get("LAUNCHER_DIR") or (LLAMA_DIR / "launcher"))
LOGS = LLAMA_DIR / "logs"
API_KEY = os.environ.get("LLAMA_API_KEY") or "sk-local-001"


def _chat_root():
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


def free_port(start=18500):
    for p in range(start, start + 60):
        s = socket.socket()
        try:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
        finally:
            s.close()
    return start


# A synthetic but content-bearing prompt. `--prompt-tokens` roughly controls
# size (English text tokenizes near 4 chars/token).
FILLER = (
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. "
    "It extends from the Arctic Ocean in the north to the Southern Ocean in the "
    "south and is bounded by Asia and Australia in the west and the Americas in "
    "the east. Its mean depth is about four thousand metres. "
)


def build_prompt(tag, target_chars):
    head = "Context block {}: ".format(tag)
    reps = max(1, target_chars // len(FILLER))
    # The answer MUST be long enough to fill --n-predict; a "one sentence"
    # answer stops at ~14 tokens and makes the throughput numbers meaningless.
    return head + (FILLER * reps) + (
        "\n\nUsing the context above as background, write a detailed 8-paragraph "
        "technical explanation of ocean circulation and heat transport. "
        "Write at least 400 words and do not stop early.")


QUALITY_PROMPT = ("Answer with only the number: what is 17 * 23?")
QUALITY_PROMPT2 = ("List the first five prime numbers, comma separated, nothing else.")


def chat(port, prompt, max_tokens, timeout_s=900, temperature=0.0):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:{}/v1/chat/completions".format(port), data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data, time.time() - t0


def wait_health(port, proc, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, time.time() - t0, "process exited rc={}".format(proc.returncode)
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:{}/health".format(port), timeout=3) as r:
                if r.status == 200:
                    return True, time.time() - t0, "ok"
        except Exception:
            pass
        time.sleep(1.5)
    return False, time.time() - t0, "health timeout"


def parse_log(log):
    d = {}
    m = re.search(r"n_ctx_slot\s*=\s*(\d+)", log)
    if m:
        d["n_ctx_slot"] = int(m.group(1))
    m = re.search(r"n_ctx\s*=\s*(\d+)", log)
    if m:
        d["n_ctx"] = int(m.group(1))
    m = re.search(r"n_slots\s*=\s*(\d+)", log)
    if m:
        d["n_slots"] = int(m.group(1))
    m = re.search(r"kv_unified\s*=\s*'(\w+)'", log)
    if m:
        d["kv_unified"] = m.group(1)
    d["cuda_errors"] = len(re.findall(r"CUDA error|cuBLAS error|out of memory", log, re.I))
    d["assert_hits"] = len(re.findall(r"GGML_ASSERT|assert.*failed", log))
    d["reprefill"] = len(re.findall(r"forcing full prompt re-processing", log, re.I))
    accs = [float(x) for x in re.findall(r"draft acceptance\s*=\s*([\d.]+)", log)]
    if accs:
        d["acceptance"] = round(sum(accs) / len(accs), 4)
    return d


def run_level(port, level, prompt_chars, n_predict):
    """Fire `level` concurrent requests; return aggregate + per-stream stats."""
    results = [None] * level

    def worker(i):
        prompt = build_prompt("s{}".format(i), prompt_chars)
        try:
            data, wall = chat(port, prompt, n_predict)
            t = data.get("timings") or {}
            pn = t.get("prompt_n") or 0
            ptps = t.get("prompt_per_second") or 0.0
            prefill_s = (pn / ptps) if ptps else 0.0
            results[i] = {
                "wall": wall,
                "gen": t.get("predicted_n") or 0,
                "prompt_n": pn,
                "decode": t.get("predicted_per_second") or 0.0,
                "prefill": ptps,
                "decode_s": max(0.0, wall - prefill_s),
            }
        except Exception as e:  # noqa: BLE001
            results[i] = {"error": str(e)}

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    ok = [r for r in results if r and "error" not in r]
    total_gen = sum(r["gen"] for r in ok)
    decode_window = max((r["decode_s"] for r in ok), default=0.0)
    return {
        "level": level,
        "wall_s": round(wall, 2),
        "ok": len(ok),
        "errors": [r["error"] for r in results if r and "error" in r],
        "total_gen": total_gen,
        "aggregate_tps": round(total_gen / wall, 2) if wall > 0 else 0,
        "agg_decode_tps": round(total_gen / decode_window, 2) if decode_window > 0 else 0,
        "per_stream_tps": round(sum(r["decode"] for r in ok) / len(ok), 2) if ok else 0,
        "streams": [{"gen": r["gen"], "decode": round(r["decode"], 2),
                     "prefill": round(r["prefill"], 1), "wall": round(r["wall"], 2)} for r in ok],
    }


def build_argv(a, port):
    model = Path(a.model_dir)
    argv = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(model),
            "--jinja", "-fa", "on", "-np", str(a.np), "-t", str(a.threads),
            "-c", str(a.ctx),
            "--batch-size", str(a.batch), "--ubatch-size", str(a.ub),
            "--cache-type-k", a.kv_k, "--cache-type-v", a.kv_v,
            "--keep", "-1", "--load-mode", a.load_mode,
            "--host", "127.0.0.1", "--port", str(port), "--api-key", API_KEY,
            "--no-warmup", "--metrics"]
    if a.unified:
        argv.append("--kv-unified")
    else:
        argv.append("--no-kv-unified")
    if a.n_cpu_moe:
        argv += ["--n-cpu-moe", str(a.n_cpu_moe)]
    elif a.ngl is not None:
        argv += ["-ngl", str(a.ngl)]
    else:
        argv += ["--fit", "on", "-fitc", str(a.ctx)]
    if a.mtp:
        argv += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(a.spec_n_max)]
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="model FOLDER under chat dir (path relative to CHAT_DIR)")
    ap.add_argument("--ctx", type=int, default=141312)
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--min-headroom", type=int, default=1024,
                    help="abort when free VRAM after load is below this many MiB")
    ap.add_argument("--prior-headroom", type=int, default=None,
                    help="free VRAM measured at np=1 for the SAME model/ctx; np>1 is "
                         "refused when it is below --min-headroom")
    ap.add_argument("--unified", dest="unified", action="store_true", default=True)
    ap.add_argument("--no-unified", dest="unified", action="store_false")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--spec-n-max", type=int, default=3)
    ap.add_argument("--kv-k", default="q4_0")
    ap.add_argument("--kv-v", default="q4_0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ub", type=int, default=256)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-cpu-moe", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt-chars", type=int, default=6000)
    ap.add_argument("--load-mode", default="none")
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--tag", default="np")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    root = _chat_root()
    if root is None:
        print("[X] cannot resolve CHAT_DIR"); return 2
    folder = Path(a.model_dir)
    if not folder.is_absolute():
        folder = root / folder
    cands = [c for c in folder.glob("*.gguf")
             if "mmproj" not in c.name.lower()]
    if not cands:
        print("[X] no main .gguf under {}".format(folder)); return 2
    # largest file = the weights (draft/secondary ggufs are always smaller)
    a.model_dir = str(max(cands, key=lambda c: c.stat().st_size))
    print("model: {}".format(a.model_dir))

    port = free_port()
    argv = build_argv(a, port)
    if a.dry:
        for x in argv:
            print("   {}".format(x))
        return 0

    used0, total = vram()
    print("VRAM before: {} / {} MiB".format(used0, total))
    if used0 is not None and used0 > 1500:
        print("[!] GPU busy ({} MiB) - stop the other server first".format(used0))
        return 1

    # GATE 1 (pre-launch): the 2026-10-24 incident. np=1 at 141312 + MTP left
    # 313 MiB free; launching np=2 on top of that green-screened the machine
    # (BugCheck 0x00020001) during load, before a single request was sent.
    # A low-headroom np=1 measurement therefore forbids any np>1 launch.
    if a.np > 1 and a.prior_headroom is not None and a.prior_headroom < a.min_headroom:
        print("[X] refusing np={}: the np=1 run for this model/ctx left only "
              "{} MiB free (< {} MiB guard).".format(a.np, a.prior_headroom, a.min_headroom))
        print("    Free memory first: lower --ctx, use a smaller KV type, or drop --mtp.")
        return 4

    LOGS.mkdir(exist_ok=True)
    print("launching tag={} port={} ctx={} np={} unified={} mtp={} ncmoe={}".format(
        a.tag, port, a.ctx, a.np, a.unified, a.mtp, a.n_cpu_moe))
    p = subprocess.Popen(argv, cwd=str(LLAMA_DIR), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    threading.Thread(target=lambda: [lines.append(ln.rstrip()) for ln in p.stdout],
                     daemon=True).start()

    rec = {"tag": a.tag, "ctx": a.ctx, "np": a.np, "unified": a.unified,
           "mtp": a.mtp, "kv_k": a.kv_k, "kv_v": a.kv_v, "batch": a.batch,
           "ub": a.ub, "ngl": a.ngl, "n_cpu_moe": a.n_cpu_moe,
           "n_predict": a.n_predict, "prompt_chars": a.prompt_chars,
           "model": Path(a.model_dir).name}
    try:
        ok, secs, why = wait_health(port, p, a.load_timeout)
        rec["load_s"] = round(secs, 1)
        if not ok:
            log = "\n".join(lines)
            (LOGS / "np-{}.log".format(a.tag)).write_text(log, encoding="utf-8")
            print("[X] not healthy: {}".format(why))
            print("\n".join(lines[-20:]))
            rec["error"] = why
            return 1

        # GATE 2 (post-launch, pre-inference): never measure on a thin card.
        used1, total1 = vram()
        headroom = (total1 - used1) if used1 is not None else None
        rec["headroom_after_load_mib"] = headroom
        if headroom is not None and headroom < a.min_headroom:
            (LOGS / "np-{}.log".format(a.tag)).write_text("\n".join(lines), encoding="utf-8")
            print("[X] headroom after load = {} MiB < {} MiB guard - aborting BEFORE any "
                  "request".format(headroom, a.min_headroom))
            rec["error"] = "headroom guard ({} MiB)".format(headroom)
            return 3

        # quality baseline at c=1
        q1, _ = chat(port, QUALITY_PROMPT, 32)
        q2, _ = chat(port, QUALITY_PROMPT2, 48)
        rec["quality_before"] = [
            (q1.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            (q2.get("choices") or [{}])[0].get("message", {}).get("content", ""),
        ]

        ladder = []
        for level in range(1, a.np + 1):
            r = run_level(port, level, a.prompt_chars, a.n_predict)
            ladder.append(r)
            print("  c={}  wall={}s  gen={}  agg={} t/s  agg_decode={} t/s  per-stream={} t/s  ok={}/{}".format(
                level, r["wall_s"], r["total_gen"], r["aggregate_tps"],
                r["agg_decode_tps"], r["per_stream_tps"], r["ok"], level))
            if r["errors"]:
                print("       errors: {}".format(r["errors"][:2]))
        rec["ladder"] = ladder

        # quality after the ladder (same slot order may differ -> catches KV
        # corruption / slot leakage, which is the real np>1 risk)
        q3, _ = chat(port, QUALITY_PROMPT, 32)
        q4, _ = chat(port, QUALITY_PROMPT2, 48)
        rec["quality_after"] = [
            (q3.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            (q4.get("choices") or [{}])[0].get("message", {}).get("content", ""),
        ]
        rec["quality_stable"] = rec["quality_before"] == rec["quality_after"]

        log = "\n".join(lines)
        (LOGS / "np-{}.log".format(a.tag)).write_text(log, encoding="utf-8")
        rec.update(parse_log(log))
        used1, _ = vram()
        rec["vram_used_mib"] = used1
        rec["headroom_mib"] = (total - used1) if used1 else None
        print("  n_ctx={} n_ctx_slot={} n_slots={} kv_unified={}".format(
            rec.get("n_ctx"), rec.get("n_ctx_slot"), rec.get("n_slots"),
            rec.get("kv_unified")))
        print("  VRAM={} MiB headroom={} MiB  quality_stable={}  cuda_err={} assert={}".format(
            rec.get("vram_used_mib"), rec.get("headroom_mib"),
            rec.get("quality_stable"), rec.get("cuda_errors"), rec.get("assert_hits")))
        return 0
    finally:
        try:
            p.terminate(); p.wait(timeout=30)
        except Exception:
            p.kill()
        for _ in range(20):
            u, _t = vram()
            if u is not None and u < 800:
                break
            time.sleep(1.5)
        time.sleep(2)
        res_path = HERE / "_np_results.json"
        try:
            old = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else []
        except Exception:
            old = []
        old.append(rec)
        res_path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  -> appended to plan/_np_results.json")


if __name__ == "__main__":
    sys.exit(main())
