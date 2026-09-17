# -*- coding: utf-8 -*-
"""Measure the prefill effect of --ubatch-size on a real server (community finding #28363).

The claim being tested, from ggml-org/llama.cpp#28363 (same 16 GiB class of card,
same Qwen3.6-35B-A3B family, 22.7K context): raising the physical ubatch from a
small value to ~2048 took cold prefill from 184-1293 tok/s to 1460-2145 tok/s
(**7.9x / +66%**), with decode unchanged.

The mechanism is specific and worth stating because it explains both the gain and
its limit: CPU-resident expert tensors are copied to the GPU **once per layer per
ubatch**, regardless of how many tokens that ubatch holds. More tokens per ubatch
is therefore pure amortisation of the same transfer. It also predicts where the
gain stops: ubatch is capped by the compute buffer, and the compute buffer is
VRAM - on a card already at the "3 GB free" line, a large ubatch can cost the
margin we just spent the whole budget equation protecting.

So this probe measures three things per ubatch, not one:
  1. prefill tok/s at real depth   (the thing that should improve a lot)
  2. decode tok/s                  (the thing that should not move)
  3. VRAM used + headroom          (the thing that can silently break the rule)
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

BASE = Path(__file__).resolve().parent.parent
# Nothing machine-specific is baked into this file. Point CHAT_DIR at your
# models directory and LLAMA_API_KEY at whatever --api-key your server was
# started with; leave the key unset if your server runs without one.
CHAT = Path(os.environ.get("CHAT_DIR", r"<models-dir>"))
API_KEY = os.environ.get("LLAMA_API_KEY", "")
Q5 = CHAT / r"Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P"


def auth_args():
    """`--api-key <key>`, or nothing at all when no key is configured."""
    return ["--api-key", API_KEY] if API_KEY else []


def auth_headers():
    """Bearer header when a key is configured, plain JSON headers otherwise."""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return h


def vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = [int(x) for x in out.splitlines()[0].split(",")]
        return used, total
    except Exception:
        return None, None


def free_port(start=8091):
    for p in range(start, start + 40):
        s = socket.socket()
        try:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
        finally:
            s.close()
    return start


def wait_health(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:{}/health".format(port), timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def post(port, payload, timeout=900):
    req = urllib.request.Request(
        "http://127.0.0.1:{}/v1/chat/completions".format(port),
        data=json.dumps(payload).encode("utf-8"),
        headers=auth_headers())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def build_prompt(target_tokens):
    """Long, varied-enough text. Repetition is fine - we want prefill cost, not quality."""
    unit = ("The llama.cpp launcher ecosystem keeps every tunable in a single JSON "
            "file so that the router configuration and the batch launchers cannot "
            "drift apart. ")
    n = max(1, int(target_tokens * 4 / len(unit)))
    return unit * n


def run_one(ub, ctx, ncmoe, target_tokens, batch=None):
    port = free_port()
    model = next(Q5.glob("*.gguf")).__str__()
    mains = [f for f in Q5.glob("*.gguf") if "mmproj" not in f.name.lower()]
    mmproj = [f for f in Q5.glob("*.gguf") if "mmproj" in f.name.lower()]
    # ubatch is capped by batch (llama.cpp requires n_ubatch <= n_batch), and
    # silently clamps - which is exactly how the first run of this probe
    # produced identical VRAM at ub 512 and ub 2048: --batch-size 512 held the
    # real value at 512 in both arms. So they are varied together.
    batch = batch or ub
    args = [str(BASE / "llama-server.exe"),
            "-m", str(mains[0]),
            "-c", str(ctx), "--n-cpu-moe", str(ncmoe),
            "--batch-size", str(batch), "--ubatch-size", str(ub),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--load-mode", "none", "-fa", "on", "--no-warmup",
            "--host", "127.0.0.1", "--port", str(port)] + auth_args()
    if mmproj:
        args += ["--mmproj", str(mmproj[0])]
    print("    cmd: llama-server -c {} --n-cpu-moe {} --batch-size {} "
          "--ubatch-size {} ...".format(ctx, ncmoe, batch, ub))
    before, total = vram_mib()
    p = subprocess.Popen(args, cwd=str(BASE), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        if not wait_health(port):
            print("    [X] server did not become healthy")
            return None
        time.sleep(3)
        used, total = vram_mib()
        prompt = build_prompt(target_tokens)
        body, wall = post(port, {
            "model": "q5", "max_tokens": 32, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]})
        t = body.get("timings") or {}
        res = {
            "batch": batch,
            "ub": ub,
            "prompt_tokens": t.get("prompt_n"),
            "prefill_tps": t.get("prompt_per_second"),
            "decode_tps": t.get("predicted_per_second"),
            "vram_mib": used,
            "headroom_mib": (total - used) if (used and total) else None,
            "wall_s": round(wall, 1),
        }
        print("    prompt_n={}  prefill={} t/s  decode={} t/s  VRAM={} MiB "
              "(headroom {} MiB)".format(
                  res["prompt_tokens"],
                  "{:.1f}".format(res["prefill_tps"]) if res["prefill_tps"] else "?",
                  "{:.1f}".format(res["decode_tps"]) if res["decode_tps"] else "?",
                  res["vram_mib"], res["headroom_mib"]))
        return res
    finally:
        try:
            p.terminate()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ub", default="512,2048",
                    help="comma-separated ubatch sizes (default: %(default)s)")
    ap.add_argument("--batch", default=None,
                    help="comma-separated batch sizes paired with --ub "
                         "(default: same as --ub). They must move together - "
                         "ubatch is silently clamped to batch")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--ncmoe", type=int, default=27)
    ap.add_argument("--tokens", type=int, default=20000,
                    help="prompt size in tokens (default: %(default)s)")
    args = ap.parse_args()

    used, total = vram_mib()
    print("VRAM before: {} / {} MiB".format(used, total))
    if used and used > 1500:
        print("[!] something already holds {} MiB of VRAM - results will be wrong."
              .format(used))
        print("    stop the running llama-server first")
        return 2

    out = []
    ubs = [int(x) for x in args.ub.split(",")]
    bss = ([int(x) for x in args.batch.split(",")] if args.batch else list(ubs))
    for ub, bs in zip(ubs, bss):
        print("\n== batch {} / ubatch {} ==".format(bs, ub))
        r = run_one(ub, args.ctx, args.ncmoe, args.tokens, batch=bs)
        if r:
            out.append(r)

    print("\n" + "=" * 78)
    print("{:>7} {:>7} {:>12} {:>12} {:>12} {:>14}".format(
        "batch", "ubatch", "prompt tok", "prefill t/s", "decode t/s",
        "headroom MiB"))
    for r in out:
        print("{:>7} {:>7} {:>12} {:>12} {:>12} {:>14}".format(
            r["batch"], r["ub"], r["prompt_tokens"],
            "{:.1f}".format(r["prefill_tps"]) if r["prefill_tps"] else "?",
            "{:.1f}".format(r["decode_tps"]) if r["decode_tps"] else "?",
            r["headroom_mib"] if r["headroom_mib"] else "?"))
    Path(BASE / "plan" / "_ub_probe.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote plan/_ub_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
