#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q6 Phase-3 apply: measured config -> launcher ecosystem (one script).

Registry-side changes (generator fixes FND-070/FND-071 are already applied in
launcher/update_launchers.py and need no data):

  launcher-models.json
    1. region: emit the missing AUTO_QWEN3_8_27B_GSQ_RCO__MM variable
       (FND-070 - the body referenced it but no definition existed, so the
       menu item died at :check_file with an empty path)
    2. entries: rewrite the Q6 plain-text Agent block with measured values
    3. entries: add the Q6 vision block right after it (same dirs, mmproj)

  preset-overrides.json
    4. Q6 params/tuning -> measured values (Router side + hub display)

Both JSON files are backed up into launcher/backup/ before writing.
--check prints what would change; --go writes.

The numbers come from the measured rows (_q6_results.json - resolved next to
this script, under <LLAMA_DIR>/plan/, or in the skill repo's
references/sessions/). Update PLAIN/VISION below only after the final ladder
is confirmed on hardware.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or HERE.parent)
L = Path(os.environ.get("LAUNCHER_DIR") or (LLAMA_DIR / "launcher"))
REG = L / "launcher-models.json"
OVR = L / "preset-overrides.json"
BAK = L / "backup"


def _find_results():
    """The measured rows, wherever this copy lives (plan/ or the skill repo)."""
    cands = (HERE / "_q6_results.json",
             LLAMA_DIR / "plan" / "_q6_results.json",
             HERE.parent.parent / "references" / "sessions" / "_q6_results.json")
    for c in cands:
        if c.is_file():
            return c
    return HERE / "_q6_results.json"


RESULTS = _find_results()

DIRNAME = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"
VMAIN = "AUTO_QWEN3_8_27B_GSQ_RCO_"
VMM = "AUTO_QWEN3_8_27B_GSQ_RCO__MM"
MMPROJ_FILE = "mmproj-Qwen3.8-27B-BF16.gguf"
GOTO_PLAIN = "RUN_QWEN3_8_27B_GSQ_RCO__MTP"
GOTO_VISION = "RUN_QWEN3_8_27B_GSQ_RCO__IMG"

# ---- final configuration (filled from _q6_results.json after the ladder) ----
PLAIN = dict(
    label="Qwen3.8-27B-GSQ-RCO 138K Agent [MTP 28t/s]",  # keep ASCII-ish for GBK bat
    ctx=141312, ngl=64, mtp=True, nmax=3, batch=256, ub=256, kv="q4_0",
    timeout=600,
)
VISION = dict(
    label="Qwen3.8-27B-GSQ-RCO 138K Vision [mmproj 20t/s]",
    ctx=141312, ngl=64, mtp=False, nmax=3, batch=256, ub=256, kv="q4_0",
    timeout=600,
)


def build_body(goto, cfg, mmproj=False):
    b = [":" + goto,
         'call :check_file "%{}%"'.format(VMAIN)]
    if mmproj:
        b.append('call :check_file "%{}%"'.format(VMM))
    b += ["echo ============================================",
          "echo " + cfg["label"],
          "echo ============================================",
          "echo [MANUAL] ctx={}".format(cfg["ctx"]),
          "llama-server.exe ^",
          '  -m "%{}%" ^'.format(VMAIN)]
    if mmproj:
        b.append('  --mmproj "%{}%" ^'.format(VMM))
    b.append("  --jinja ^")
    if mmproj:
        b.append("  --image-min-tokens 1024 ^")
    b.append("  -c {} ^".format(cfg["ctx"]))
    b.append("  -ngl {} ^".format(cfg["ngl"]))
    if cfg["mtp"]:
        b += ["  --spec-type draft-mtp ^",
              "  --spec-draft-n-max {} ^".format(cfg["nmax"])]
    b += ["  -fa on ^",
          "  -np 1 ^",
          "  -t 12 ^",
          "  --batch-size {} ^".format(cfg["batch"]),
          "  --ubatch-size {} ^".format(cfg["ub"]),
          "  --cache-type-k {} ^".format(cfg["kv"]),
          "  --cache-type-v {} ^".format(cfg["kv"]),
          "  --keep -1 ^",
          "  --load-mode none ^",
          "  --host 0.0.0.0 ^",
          "  --port %PORT% ^",
          "  --api-key %API_KEY% ^",
          "  --temp 1.0 ^",
          "  --top-p 0.95 ^",
          "  --top-k 20 ^",
          "  --reasoning-preserve ^",
          "  --reasoning-budget 8192 ^",
          "  --reasoning-format deepseek ^",
          "  --min-p 0.0 ^",
          "  --presence-penalty 0.0 ^",
          "  --repeat-penalty 1.0 ^",
          '  --chat-template-kwargs "{\\"reasoning_effort\\":\\"medium\\"}" ^',
          "  --metrics ^",
          "  %TOOLS_ARG% ^",
          "  --timeout {}".format(cfg["timeout"]),
          "pause & goto menu"]
    return "\r\n".join(b)


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def measured_points():
    """Pull the 45K-context rows for the report/note from _q6_results.json."""
    try:
        hist = load(RESULTS)
    except Exception:
        return {}
    out = {}
    for r in hist:
        if r.get("prompt_n", 0) > 40000 and not r.get("error"):
            out[r["tag"]] = {k: r.get(k) for k in
                             ("decode_tps", "prompt_tps", "acceptance",
                              "vram_used_mib", "headroom_mib", "ngl",
                              "mtp", "batch")
                             if k in r}
    return out


def apply_registry(check):
    reg = load(REG)
    qwen = reg["launchers"]["qwen"]
    changes = []

    # 1. region: add the missing _MM variables (FND-070). Q6 is the task
    # target; Q5 (35B UNC-Hau-MTP) is the same bug's original victim - fix both
    # in one pass, since both menu items died at :check_file with an empty path.
    mm_fixes = (
        (DIRNAME, VMM, MMPROJ_FILE),
        ("Qwen3.6-35B-A3B-UNC-Hau-MTP", "AUTO_QWEN3_6_35B_A3B_UNC__MM",
         "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
    )
    for dirname, vmm, mmfile in mm_fixes:
        found = False
        for grp in qwen["region"]:
            if grp.get("t") != "group":
                continue
            if any(dirname in v for v in grp.get("vars", [])):
                found = True
                if not any(v.startswith('set "{}='.format(vmm))
                           for v in grp["vars"]):
                    line = 'set "{}={}"'.format(
                        vmm, "\\".join(["%CHAT%", dirname, mmfile]))
                    grp["vars"].append(line)
                    changes.append("region: appended {}".format(line))
                break
        if not found:
            raise SystemExit("[X] region group not found for " + dirname)

    # 2/3. entries: rewrite plain, insert vision after it
    ents = qwen["entries"]
    idx = next((i for i, e in enumerate(ents) if e.get("goto") == GOTO_PLAIN), None)
    if idx is None:
        raise SystemExit("[X] Q6 plain entry not found")
    ents[idx]["label"] = PLAIN["label"]
    ents[idx]["body"] = build_body(GOTO_PLAIN, PLAIN, mmproj=False)
    ents[idx]["mmproj"] = False
    ents[idx].pop("drafts", None)
    changes.append("entries[{}]: rewrote plain body (ngl {} mtp {} kv {} ctx {})"
                   .format(idx, PLAIN["ngl"], PLAIN["mtp"], PLAIN["kv"], PLAIN["ctx"]))
    has_vision = any(e.get("goto") == GOTO_VISION for e in ents)
    if not has_vision:
        vision_entry = {
            "label": VISION["label"],
            "goto": GOTO_VISION,
            "dirs": [DIRNAME],
            "drafts": [],
            "banner": [],
            "body": build_body(GOTO_VISION, VISION, mmproj=True),
            "profile": "qwen38-27b",
            "mmproj": True,
        }
        ents.insert(idx + 1, vision_entry)
        changes.append("entries: inserted vision block at {} ({} -> {})"
                       .format(idx + 1, GOTO_VISION, VISION["label"]))

    if check:
        for c in changes:
            print("  [check] " + c)
        return False
    BAK.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(REG, BAK / "launcher-models.json.bak-{}".format(stamp))
    with REG.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for c in changes:
        print("  [go]    " + c)
    return True


def apply_overrides(check):
    ovr = load(OVR)
    models = ovr["models"]
    ent = models.get(DIRNAME)
    if ent is None:
        raise SystemExit("[X] Q6 override entry not found")
    pts = measured_points()
    # final ladder: g3 (ngl64+MTP+b256) is the winning row; fall back to d3
    d3 = (pts.get("g3-full-b256-mtp") or pts.get("d3-mtp-long56k-b512", {}))
    d1 = pts.get("d1-long56k-q4-b512", {})
    ent["id"] = "Qwen3.8-27B-138K-Agent"
    # Router (models-config.ini) section must stay text-first: loading the
    # vision tower costs +888 MiB and drops the 141K/full-GPU config into the
    # slow fallback path (measured 2.9 t/s).  The menu Q6m entry keeps mmproj.
    ent["no_mmproj"] = True
    aliases = ent.setdefault("alias", [])
    for a in ("Q6 Qwen3.8-27B-253K-IMG", "Q6 Qwen3.8-27B-64K-IMG"):
        if a not in aliases:
            aliases.append(a)
    ent["note"] = (
        "PURE-TEXT Agent | measured 2026-10-12 @45K prompt: decode {:.1f} t/s (MTP, "
        "acc {:.2f}) / prefill {:.0f} t/s | no-MTP {} t/s | ctx {} ngl {} KV {} "
        "| headroom {} MiB".format(
            d3.get("decode_tps", 0) or 0, d3.get("acceptance", 0) or 0,
            d3.get("prompt_tps", 0) or 0, round(d1.get("decode_tps", 0) or 0),
            PLAIN["ctx"], PLAIN["ngl"], PLAIN["kv"], d3.get("headroom_mib", "?")))
    ent["params"] = {
        "ctx-size": PLAIN["ctx"],
        "batch-size": PLAIN["batch"],
        "ubatch-size": PLAIN["ub"],
        "cache-type-k": PLAIN["kv"],
        "cache-type-v": PLAIN["kv"],
        "temp": 1.0, "top-p": 0.95, "top-k": 20, "min-p": 0.0,
        "presence-penalty": 0.0, "repeat-penalty": 1.0,
        "reasoning-budget": 8192, "reasoning-format": "deepseek",
        "spec-type": "draft-mtp", "spec-draft-n-max": PLAIN["nmax"],
    }
    tuning = {"source": "measured-2026-10-12 (plan/_q6_probe.py)",
              "measured_realctx": []}
    for tag, r in sorted(pts.items()):
        tuning["measured_realctx"].append({
            "tag": tag,
            "decode_tps": round(r.get("decode_tps") or 0, 2),
            "prompt_tps": round(r.get("prompt_tps") or 0, 1),
            "acceptance": r.get("acceptance"),
            "vram_mib": r.get("vram_used_mib"),
            "headroom_mib": r.get("headroom_mib"),
            "date": "2026-10-12",
        })
    ent["tuning"] = tuning
    if check:
        print("  [check] overrides: Q6 -> {}".format(ent["note"][:110] + "..."))
        return False
    BAK.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(OVR, BAK / "preset-overrides.json.bak-{}".format(stamp))
    with OVR.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(ovr, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("  [go]    overrides: Q6 params/tuning updated")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.check and not a.go:
        ap.error("choose --check or --go")
    print("== registry ==")
    apply_registry(check=a.check)
    print("== preset-overrides ==")
    apply_overrides(check=a.check)
    if a.go:
        print("now run: python launcher\\update_launchers.py --yes  (renders the .bat + ini;")
        print("         do NOT use update-launchers.bat - its trailing pause blocks automation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
