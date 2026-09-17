# -*- coding: utf-8 -*-
"""Verify a grafted MTP head end to end: does it load, and does it actually speed decode up?

Two arms, same weights, same prompt, same everything else:

  A  no speculative decoding            <- the baseline
  B  --spec-type draft-mtp              <- the head is INSIDE the model file, so
     --spec-draft-n-max 2                  there is no separate draft GGUF to load

The community measurement this is checking (ggml-org/llama.cpp#28363, same
16 GiB class of card, same Qwen3.6-35B-A3B family at 128K): decode
62.2 -> 79.4-79.7 tok/s (+28%), draft acceptance 0.851-0.891.

What this script refuses to skip
--------------------------------
* **It verifies the graft loaded**, not just that the process stayed up: the
  startup log must report the extra block. A file whose block_count was rewritten
  but whose tensors were mis-offset can still load and produce subtly wrong
  output, so the acceptance counter is read too - acceptance near zero means the
  head is running but not predicting anything useful, which is the signature of a
  bad graft rather than a bad idea.
* **It compares against a baseline measured in the same session.** The earlier
  lesson from this project stands: an absolute number from a post is not
  comparable with a number measured here, only A/B is.
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

BASE = Path(__file__).resolve().parent.parent
# Nothing machine-specific is baked into this file. Point CHAT_DIR at your
# models directory and LLAMA_API_KEY at whatever --api-key your server was
# started with; leave the key unset if your server runs without one.
CHAT = Path(os.environ.get("CHAT_DIR", r"<models-dir>"))
API_KEY = os.environ.get("LLAMA_API_KEY", "")
DEFAULT_MODEL = CHAT / "_mtp_graft" / r"Qwen3.6-35B-A3B-UNC-Hau-MTPhead.gguf"
MMPROJ_DIR = CHAT / r"Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P"


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


def free_port(start=8101):
    for p in range(start, start + 40):
        s = socket.socket()
        try:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
        finally:
            s.close()
    return start


def wait_health(port, timeout=240):
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


def run_arm(model, mmproj, spec, ctx, ncmoe, batch, ub, n_predict):
    port = free_port()
    args = [str(BASE / "llama-server.exe"), "-m", str(model),
            "-c", str(ctx), "--n-cpu-moe", str(ncmoe),
            "--batch-size", str(batch), "--ubatch-size", str(ub),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--load-mode", "none", "-fa", "on", "--jinja", "--no-warmup",
            "--host", "127.0.0.1", "--port", str(port)] + auth_args()
    if mmproj:
        args += ["--mmproj", str(mmproj)]
    if spec:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    print("    arm: {}".format("MTP" if spec else "baseline"))
    before, total = vram_mib()
    p = subprocess.Popen(args, cwd=str(BASE), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    log_lines = []

    def drain():
        try:
            for ln in p.stdout:
                log_lines.append(ln.rstrip())
        except Exception:
            pass

    import threading
    th = threading.Thread(target=drain, daemon=True)
    th.start()
    try:
        if not wait_health(port):
            tail = "\n".join(log_lines[-25:])
            print("    [X] did not become healthy. last log lines:\n{}".format(tail))
            return None
        time.sleep(2)
        used, total = vram_mib()
        req = urllib.request.Request(
            "http://127.0.0.1:{}/v1/chat/completions".format(port),
            data=json.dumps({
                "model": "m", "max_tokens": n_predict, "temperature": 0,
                "messages": [{"role": "user",
                              "content": "Write a short paragraph about why "
                                         "measuring is better than guessing."}]},
            ).encode("utf-8"),
            headers=auth_headers())
        with urllib.request.urlopen(req, timeout=900) as r:
            body = json.loads(r.read().decode("utf-8"))
        t = body.get("timings") or {}
        log = "\n".join(log_lines)
        acc = re.search(r"draft acceptance[^\d]*([\d.]+)", log)
        mlen = re.search(r"mean len[^\d]*([\d.]+)", log)
        nblocks = re.search(r"n_layer\s*=\s*(\d+)", log)
        # the loader prints the block count it actually built
        built = max([int(x) for x in re.findall(r"blk\.(\d+)\.", log)] or [-1])
        res = {
            "arm": "mtp" if spec else "baseline",
            "n_prompt": t.get("prompt_n"),
            "n_gen": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "vram_mib": used,
            "headroom_mib": (total - used) if (used and total) else None,
            "acceptance": float(acc.group(1)) if acc else None,
            "mean_len": float(mlen.group(1)) if mlen else None,
            "max_blk_seen": built,
        }
        print("    decode={} t/s  gen={} tok  VRAM={} MiB  maxblk={}  acc={}".format(
            "{:.1f}".format(res["decode_tps"]) if res["decode_tps"] else "?",
            res["n_gen"], res["vram_mib"], built,
            res["acceptance"] if res["acceptance"] is not None else "-"))
        return res, log
    finally:
        try:
            p.terminate()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--ncmoe", type=int, default=27)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--n-predict", type=int, default=400,
                    help="generated tokens per arm (default: %(default)s). Needs "
                         "to be long enough for the acceptance counter to be "
                         "meaningful, short enough to not cost minutes")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print("[X] model not found: {}".format(model))
        return 2
    mm = None
    if MMPROJ_DIR.exists():
        c = [f for f in MMPROJ_DIR.glob("*.gguf") if "mmproj" in f.name.lower()]
        mm = c[0] if c else None
    used, total = vram_mib()
    print("model : {} ({:.2f} GiB)".format(model.name, model.stat().st_size / (1 << 30)))
    print("mmproj: {}".format(mm.name if mm else "(none)"))
    print("VRAM before: {} / {} MiB".format(used, total))
    if used and used > 1500:
        print("[!] VRAM already in use - stop the running server first")
        return 2

    results = []
    for spec in (False, True):
        print("\n== {} ==".format("with built-in MTP head" if spec else "baseline"))
        r = run_arm(model, mm, spec, args.ctx, args.ncmoe, args.batch, args.ub,
                    args.n_predict)
        if r:
            results.append(r[0])
            Path(BASE / "plan" / "_mtp_verify_{}.log".format(
                "mtp" if spec else "base")).write_text(r[1], encoding="utf-8")

    print("\n" + "=" * 78)
    print("{:>10} {:>10} {:>12} {:>12} {:>10} {:>12}".format(
        "arm", "gen tok", "decode t/s", "vs base", "accept", "headroom MiB"))
    base = next((r for r in results if r["arm"] == "baseline"), None)
    for r in results:
        gain = ""
        if base and base.get("decode_tps") and r.get("decode_tps"):
            gain = "{:+.1f}%".format(
                (r["decode_tps"] / base["decode_tps"] - 1) * 100)
        print("{:>10} {:>10} {:>12} {:>12} {:>10} {:>12}".format(
            r["arm"], r["n_gen"],
            "{:.1f}".format(r["decode_tps"]) if r["decode_tps"] else "?",
            gain,
            r["acceptance"] if r["acceptance"] is not None else "-",
            r["headroom_mib"] if r["headroom_mib"] else "?"))
    Path(BASE / "plan" / "_mtp_verify.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\nwrote plan/_mtp_verify.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
