#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp deployment environment diagnostic (cross-platform Python version).

Cross-platform counterpart of `scripts/detect.ps1` (PowerShell, Windows-only).
Detects the same 6 items: llama-server presence, CUDA arch, MTP parameter
fingerprint, NVIDIA driver version, model files, and VRAM state.

Works on Windows / Linux / macOS. Requires Python 3.7+.
`nvidia-smi` is optional - GPU checks are skipped with a warning if missing.

Usage:
    python detect.py
    python detect.py --llama-dir /path/to/llama.cpp --models-dir /path/to/models

Cross-platform notes:
    - On Windows the binary is `llama-server.exe`; on Linux/macOS it is `llama-server`.
    - The Blackwell driver minimum (610.47 / 610.62) check applies ONLY on Windows
      (Linux driver versions are 535/550/560 etc. and would wrongly trigger it),
      on Linux the driver version is simply reported.
    - Keep this file in sync with detect.ps1 when adding checks.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

BLACKWELL_MIN = (610, 47)
BLACKWELL_REC = (610, 62)


def is_windows():
    return os.name == "nt"


def server_name():
    return "llama-server.exe" if is_windows() else "llama-server"


def parse_version(v):
    """Return a comparable tuple from '610.47' / '535.104.05' style strings."""
    return tuple(int(x) for x in re.findall(r"\d+", str(v))) or (0,)


def run(cmd, timeout=90):
    """Run a command; return (exitcode, combined_output)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return -1, ""
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as exc:  # pragma: no cover
        return -1, str(exc)


def check_server(llama_dir):
    print("[1/6] Checking llama-server ... ", end="")
    svr = Path(llama_dir) / server_name()
    if svr.is_file():
        print("OK FOUND")
        print(f"      path: {svr}")
        return str(svr)
    print("MISSING")
    print(f"      expected: {svr}")
    print("      Please extract a llama.cpp prebuilt package to the correct directory.")
    return None


def check_cuda(binary):
    print("[2/6] Detecting CUDA arch ... ", end="")
    if not binary:
        print("SKIPPED (llama-server not found)")
        return None
    _, help_text = run([binary, "--help"])
    m = re.search(r"sm_\d+", help_text)
    if not m:
        print("UNKNOWN (CPU-only build?)")
        return None
    arch = m.group(0)
    print(f"OK {arch}")
    if arch == "sm_89":
        print("      WARNING sm_89 = CUDA 12.4 build; Blackwell (RTX 50) needs sm_120")
        print("      Recommend: download the CUDA 13.3 build (llama-bXXXX-bin-cuda-13.3-x64.zip)")
    elif arch == "sm_120":
        print("      OK Blackwell native support")
    else:
        print(f"      arch: {arch} (non-Blackwell)")
    return arch


def check_mtp(binary):
    print("[3/6] Detecting MTP parameter naming ... ", end="")
    if not binary:
        print("SKIPPED (llama-server not found)")
        return None
    _, help_text = run([binary, "--help"])
    if "spec-draft-n-max" in help_text:
        print("OK --spec-* naming (b10056 standard)")
        found = [p for p in ("spec-type", "spec-draft-n-max", "spec-draft-ngl") if p in help_text]
        print(f"      available: {', '.join(found)}")
        if re.search(r"spec-type.*draft-mtp", help_text):
            print("      OK draft-mtp type available")
        else:
            print("      WARNING draft-mtp type unavailable")
        if "spec-draft-buffer" in help_text:
            print("      WARNING spec-draft-buffer still present (expected removed in b10056)")
        if "draft-n" in help_text:
            print("      INFO old --draft-* params shown as removed")
        return "spec-* (b10056)"
    if re.search(r"draft-model|draft-mtp-n", help_text):
        print("WARNING --draft-* naming (newer build)")
        print("      MTP params need migration to --draft-* naming")
        print("      spec->draft mapping:")
        print("        --spec-type         -> --draft-type")
        print("        --spec-draft-n-max -> --draft-mtp-n")
        print("        --gpu-layers-draft -> --draft-mtp-ngl")
        print("        --model-draft      -> --draft-model")
        return "draft-* (newer build)"
    print("UNKNOWN MTP params")
    return "Unknown"


def check_driver():
    print("[4/6] Detecting NVIDIA driver ... ", end="")
    code, out = run(["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"])
    if code != 0 or not out.strip():
        print("WARNING nvidia-smi returned no data")
        return None
    parts = [p.strip() for p in out.strip().split(",")]
    driver = parts[0] if parts else ""
    total = parts[1] if len(parts) > 1 else ""
    print(f"OK driver {driver} | VRAM {total}")
    if is_windows():
        dv = parse_version(driver)
        if dv and dv < BLACKWELL_MIN:
            print(f"      WARNING Blackwell minimum is 610.47, current {driver}")
        elif dv and dv >= BLACKWELL_REC:
            print("      OK driver meets recommendation")
    else:
        print("      INFO driver threshold check is Windows-only; version reported as-is")
    return driver


def scan_models(models_dir):
    print("[5/6] Scanning model directory ... ", end="")
    md = Path(models_dir)
    if not md.is_dir():
        print("MISSING directory:", md)
        return None
    gguf = sorted(md.rglob("*.gguf"))
    print(f"OK found {len(gguf)} GGUF files")
    groups = {}
    for f in gguf:
        groups.setdefault(str(f.parent), []).append(f)
    print("      model dir distribution:")
    for d in sorted(groups):
        files = groups[d]
        rel = d.replace(str(md), "") or os.sep
        total_gb = sum(f.stat().st_size for f in files) / (1024 ** 3)
        print(f"        {rel} : {len(files)} files, {total_gb:.2f} GB")
    nonstd = [f for f in gguf if f.name != "model.gguf" and not re.search(r"mmproj|mtp", str(f.parent), re.IGNORECASE)]
    if nonstd:
        print(f"      INFO {len(nonstd)} non-standard named GGUF (not model.gguf)")
    return len(gguf)


def check_vram():
    print("[6/6] Detecting GPU VRAM ... ", end="")
    code, out = run(["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total",
                     "--format=csv,noheader,nounits", "-i", "0"])
    if code != 0 or not out.strip():
        print("WARNING unable to get VRAM info")
        return None
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) < 3:
        print("WARNING unexpected nvidia-smi output")
        return None
    free_mb, used_mb, total_mb = (int(p) for p in parts[:3])
    used_pct = round(used_mb / total_mb * 100, 1) if total_mb else 0.0
    print(f"OK free {free_mb}MB / used {used_mb}MB ({used_pct}%) / total {total_mb}MB")
    if free_mb < 2000:
        print("      WARNING free VRAM < 2GB, large model deployment may be limited")
    if total_mb < 8000:
        print("      WARNING total VRAM < 8GB, 12B+ models may fail to load")
    return free_mb


def main():
    default_llama = r"C:\llama.cpp" if is_windows() else "llama.cpp"
    default_models = r"C:\models" if is_windows() else "models"

    ap = argparse.ArgumentParser(description="llama.cpp deployment environment diagnostic")
    ap.add_argument("--llama-dir", default=default_llama, help="llama.cpp working directory")
    ap.add_argument("--models-dir", default=default_models, help="models directory")
    args = ap.parse_args()

    # Best-effort UTF-8 output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    results = {}
    print("=" * 44)
    print("  llama.cpp environment diagnostic")
    print("  platform: " + sys.platform)
    print("=" * 44)
    print()

    binary = check_server(args.llama_dir)
    results["llama-server"] = "Found" if binary else "Missing"

    arch = check_cuda(binary)
    if arch:
        results["CUDA_Arch"] = arch

    mtp = check_mtp(binary)
    if mtp:
        results["MTP_Fingerprint"] = mtp

    driver = check_driver()
    if driver:
        results["NVIDIA_Driver"] = driver

    count = scan_models(args.models_dir)
    if count is not None:
        results["GGUF_Count"] = count

    vram = check_vram()
    if vram is not None:
        results["VRAM_Free_MB"] = vram

    print()
    print("=" * 44)
    print("  Summary")
    print("=" * 44)
    for k in sorted(results):
        print(f"  {k} : {results[k]}")
    print("=" * 44)


if __name__ == "__main__":
    main()
