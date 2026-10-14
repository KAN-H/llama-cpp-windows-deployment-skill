#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_refactor2_regression.py - Phase 1 invariant checks for the launcher ecosystem
==============================================================================
Checks (all must pass before AND after any regeneration):

  A. no entry body contains an empty line inside a `^` continuation
     (FND-014 - cmd.exe silently drops every arg after such a blank line)
  B. every `goto` label is unique per launcher
  C. every body that launches llama-server.exe declares --port and --api-key
  D. --strict only: models-config.bat passes --models-max exactly once
     (CON-006; owned by Phase 3 TASK-020, advisory until then)
  E. MoE / --fit regressions - pinned because every one of them shipped, and
     the user found each of them before any test did:
       E1 FND-064  --n-cpu-moe in the body disagrees with preset-overrides.json
                   (or is missing altogether) -> the .bat silently runs a
                   different configuration than the tuned and validated one
       E2 FND-066  --fit sitting next to --n-cpu-moe -> fit gives up as soon as
                   tensor_buft_overrides is non-empty, so the flag advertises a
                   safety net that never runs
       E3 FND-033  an explicit -ngl does the same thing to --fit
       E4 FND-063  '-fittarget' is not a flag; llama-server refuses to start
       E5 TASK-080 --no-mmap is deprecated in favour of --load-mode

Exit code 0 = PASS, 1 = FAIL.

--self-test runs the E checks against deliberately broken bodies: a check that
passes on everything is worthless, and these bugs shipped precisely because
nothing failed when they appeared.
Launcher dir: LAUNCHER_DIR env > <LLAMA_DIR>/launcher (LLAMA_DIR env overrides
the default, this script's parent.parent - the script is expected to sit in
<llama-cpp-dir>/plan/)."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Resolved launcher dir: LAUNCHER_DIR env > <LLAMA_DIR>/launcher, where
# LLAMA_DIR defaults to this script's parent.parent (the script is expected
# to sit in <llama-cpp-dir>/plan/; set LLAMA_DIR to run it from elsewhere).
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or Path(__file__).resolve().parent.parent)
LAUNCHER_DIR = Path(os.environ.get("LAUNCHER_DIR") or (LLAMA_DIR / "launcher"))
REGISTRY = LAUNCHER_DIR / "launcher-models.json"
MODELS_CONFIG = LAUNCHER_DIR / "models-config.bat"

BLANK_CONT_RE = re.compile(r"\^[ \t]*\r?\n[ \t]*\r?\n")
REQUIRED_ARGS = ("--port", "--api-key")

# --- E group patterns. Deliberately written as "whitespace then the flag" so a
# `REM` line cannot be mistaken for a real argument; executable() strips those
# first, but a body may also carry the flag inside an `echo` banner.
MOE_RE = re.compile(r"(?<!\S)--n-cpu-moe\s+(\d+)")
# Mirrors strip_decorative_fit() exactly: only the two flag forms that carry a
# value. A bare `--fit` inside an `echo` banner ("echo --fit 自动分层") is
# prose, not an argument, and must not be reported as one - the first run of
# this check flagged two such banners and sent me chasing a non-bug.
FIT_RE = re.compile(r"(?<!\S)(--fit-ctx|-fitc)\s+\d+"
                    r"|(?<!\S)(--fit|-fit)\s+(on|off)(?=\s|$)")
FIT_ON_RE = re.compile(r"(?<!\S)(--fit|-fit)\s+on(?=\s|$)")
NGL_RE = re.compile(r"(?<!\S)-ngl\s+\S+")
BAD_FLAG_RE = re.compile(r"(?<!\S)-fittarget(?=\s|$)")
DEPRECATED_RE = re.compile(r"(?<!\S)--no-mmap(?=\s|$)")
FIT_OFF_RE = re.compile(r"(?<!\S)(--fit|-fit)\s+off(?=\s|$)")
REM_PREFIXES = ("REM", "@REM", "::", "#", ";")
OVERRIDES = LAUNCHER_DIR / "preset-overrides.json"


def executable(body):
    """Body text with remark lines removed - a comment may name a flag safely."""
    return "\n".join(ln for ln in body.splitlines()
                      if not ln.lstrip().upper().startswith(REM_PREFIXES))


API_KEY_RE = re.compile(r'set\s+"API_KEY=([^"\r\n]+)"')


def shared_api_key():
    """The API key every launcher shares, straight from the registry (FND-021).

    The per-launcher `head` text literally contains `set "API_KEY=..."`, so
    reading the value from there keeps this file free of any credential of
    its own. Returns None when the registry is unreadable or defines no key -
    callers then skip the comparison instead of guessing.
    """
    try:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return None
    keys = []
    for cfg in (reg.get("launchers") or {}).values():
        m = API_KEY_RE.search(str(cfg.get("head", "")))
        if m:
            keys.append(m.group(1))
    return max(set(keys), key=keys.count) if keys else None


def check_entry(script, e, want):
    """Return the E-group problems for one registry entry.

    `want` is the tuned --n-cpu-moe for this entry's model, or None when the
    model is not an offloaded MoE. Kept out of main() so --self-test can drive
    it with synthetic bodies.
    """
    out = []
    body = e.get("body", "") or ""
    if not body or "llama-server" not in body:
        return out        # goto-alias: the launch block is elsewhere and carries
                          # the arguments (e.g. RUN_26B_AGENT -> RUN_26B_MTP)
    exe = executable(body)
    where = "{}: entry '{}'".format(script, e.get("label", "?"))

    if BAD_FLAG_RE.search(exe):
        out.append("{}: '-fittarget' is not a flag - llama-server exits at "
                   "startup (FND-063)".format(where))
    if DEPRECATED_RE.search(exe):
        out.append("{}: deprecated '--no-mmap' - use '--load-mode none' "
                   "(TASK-080)".format(where))
    if FIT_ON_RE.search(exe) and NGL_RE.search(exe):
        out.append("{}: has '--fit on' and an explicit -ngl - -ngl lands in "
                   "tensor_buft_overrides, so --fit gives up and the flag "
                   "does nothing (FND-033)".format(where))
    if FIT_ON_RE.search(exe) and FIT_OFF_RE.search(exe):
        out.append("{}: has both '--fit on' and '-fit off' - the last flag "
                   "wins, so fit is silently disabled and a dense model "
                   "with no -ngl runs CPU-only (FND-071)".format(where))

    if want is None:
        return out
    m = MOE_RE.search(exe)
    if not m:
        out.append("{}: MoE model (tuned to --n-cpu-moe {}) launches without "
                   "it - an explicit -ngl or a plain --fit loads it the wrong "
                   "way (FND-033/FND-064)".format(where, want))
        return out
    if int(m.group(1)) != want:
        out.append("{}: --n-cpu-moe {} but preset-overrides.json says {} - the "
                   "entry is stale and runs a different configuration than "
                   "the tuned one (FND-064)".format(where, m.group(1), want))
    if FIT_RE.search(exe):
        out.append("{}: --fit next to --n-cpu-moe - fit bails whenever "
                   "tensor_buft_overrides is set, so the safety net never "
                   "runs (FND-066)".format(where))
    return out


# --- negative controls -------------------------------------------------
SELFTEST_BODY = ('llama-server.exe --model "%MODEL_DIR%\\m\\x.gguf" '
                 '--port %PORT% --api-key sk-local-001 --host 0.0.0.0 '
                 '--keep -1 ')


def self_test():
    """Run the E checks against broken bodies; return the number of failures."""
    cases = (
        ("E1 stale --n-cpu-moe", {"dirs": ["m"], "label": "t",
                                    "body": SELFTEST_BODY + "--n-cpu-moe 18 "},
         27, "FND-064"),
        ("E2 --fit beside --n-cpu-moe", {"dirs": ["m"], "label": "t",
                                          "body": SELFTEST_BODY +
                                          "--n-cpu-moe 27 --fit on "},
         27, "FND-066"),
        ("E2b MoE without the flag", {"dirs": ["m"], "label": "t",
                                       "body": SELFTEST_BODY + "-ngl 99 "},
         27, "FND-033/FND-064"),
        ("E3 --fit on + -ngl", {"label": "t",
                                 "body": SELFTEST_BODY + "--fit on -ngl 99 "},
         None, "FND-033"),
        ("E4 -fittarget", {"label": "t",
                            "body": SELFTEST_BODY + "-fittarget 1024 "},
         None, "FND-063"),
        ("E5 --no-mmap", {"label": "t", "body": SELFTEST_BODY + "--no-mmap "},
         None, "TASK-080"),
        ("E7 fit on + fit off", {"label": "t",
                                  "body": SELFTEST_BODY + "--fit on -fit off "},
         None, "FND-071"),
        ("clean MoE body", {"dirs": ["m"], "label": "t",
                             "body": SELFTEST_BODY +
                             "--n-cpu-moe 27 --load-mode none "},
         27, None),
        ("clean goto alias", {"label": "t", "body": "goto RUN_X\r\n"},
         None, None),
        ("remark mentioning --no-mmap",
         {"label": "t", "body": "REM --no-mmap was removed\r\n" + SELFTEST_BODY},
         None, None),
    )
    bad = 0
    for name, e, want, needle in cases:
        got = check_entry("selftest", e, want)
        expect_one = needle is not None
        ok = (len(got) == (1 if expect_one else 0)
              and (not expect_one or any(needle in p for p in got)))
        if not ok:
            bad += 1
        print("  {} {:<30} {} problem(s)".format(
            "[OK]" if ok else "[X]", name, len(got)))
        if not ok:
            for p in got:
                print("        {}".format(p))
    return bad


# --------------------------------------------- F: pure-function assertions
# llama_hub.py had ZERO automated coverage. Checks A-E read the generated
# .bat / ini / models-config.bat, so they never touch it - yet it is the file the
# user actually operates (the menu, --guide, --audit, the parameter editor), and
# it is where this session's moe_cell() / retag_ctx() / vram_verdict() landed.
#
# The helpers below are pure - no I/O, no global state, deterministic - which
# makes them the cheapest possible place to close the widest coverage gap.
#
# Every expected value is REAL measured data, not a number picked to make the
# test pass: the two VRAM points are the ones from the user's own OOM
# investigation (n=20 OOM'd on an image request, n=27 fixed it).
Q5_VRAM_CALIB = {
    "layers": 40, "kv_kib_per_token": 13.24, "kv_type": "q8_0",
    "expert_layer_mib": 531.0, "fixed_mib": 3779, "vram_total_mib": 16311,
}


def check_pure(problems):
    """Assert the pure helpers in llama_hub / update_launchers. Returns count."""
    if str(LAUNCHER_DIR) not in sys.path:
        sys.path.insert(0, str(LAUNCHER_DIR))
    try:
        import llama_hub as H
    except Exception as exc:
        problems.append("llama_hub is not importable: {}: {}".format(
            type(exc).__name__, exc))
        return 0
    n = 0

    # --- retag_ctx: rewrite the context token, and ONLY it -----------------
    for mid, ctx, want in (
            ("Q5 Qwen3.6-35B-A3B-64K-MoE-IMG-UNC-Hau", 262144,
             "Q5 Qwen3.6-35B-A3B-256K-MoE-IMG-UNC-Hau"),
            ("Q3 QwenPaw-9B-128K-MTP-UNC", 65536,
             "Q3 QwenPaw-9B-64K-MTP-UNC"),
            # 0.6B and F16 are model facts, not context - they must survive
            ("Q4 Qwen3-0.6B-F16-32K", 16384,
             "Q4 Qwen3-0.6B-F16-16K"),
    ):
        n += 1
        got = H.retag_ctx(mid, ctx)
        if got != want:
            problems.append("retag_ctx({!r}, {}) -> {!r}, expected {!r}"
                            .format(mid, ctx, got, want))
    for mid, ctx in (("NoContextToken", 65536), ("X 32K", None), ("", 65536)):
        n += 1
        got = H.retag_ctx(mid, ctx)
        if got is not None:
            problems.append(
                "retag_ctx({!r}, {!r}) -> {!r}; with nothing to rewrite it "
                "must return None, not a mangled id".format(mid, ctx, got))

    # --- moe_cell: the value AND its provenance ----------------------------
    # The four tags exist because the same number means very different things
    # depending on where it came from; a bench-ctx value in particular can only
    # rank and will OOM if trusted. If this column ever collapses to one label,
    # FND-067 is back.
    for entry, want in (
            ({"params": {"n-cpu-moe": 28},
              "tuning": {"source": "inherited-from:qwen36-35b-a3b"}},
             "28 继承"),
            ({"params": {"n-cpu-moe": 28},
              "tuning": {"source": "user-measured-2026-09-12"}},
             "28 实测"),
            ({"params": {"n-cpu-moe": 28},
              "tuning": {"measured_benchctx": [{"n-cpu-moe": 28, "tps": 70}]}},
             "28 实测小ctx"),
            ({"params": {"n-cpu-moe": 28}, "tuning": {}}, "28 预算"),
            ({"params": {}}, "-"),
    ):
        n += 1
        got = H.moe_cell(entry)
        if got != want:
            problems.append("moe_cell({}) -> {!r}, expected {!r}".format(
                entry.get("tuning"), got, want))

    # --- vram_verdict: the budget equation against real measurements -------
    q5 = {"params": {"n-cpu-moe": 27, "ctx-size": 65536},
          "tuning": {"vram": Q5_VRAM_CALIB}}
    for ncm, ctx, want_est, want_free, want_ok in (
            # the fix: measured 11,529 MiB used / 4,782 free -> inside the rule
            (27, 65536, 11529, 4782, True),
            # the OOM: measured 15,246 / 1,065 -> below the 3 GB rule
            (20, 65536, 15246, 1065, False),
            # The two above only prove the function computes SOMETHING. Neither
            # pins the rule itself - 4,782 clears any sane threshold and 1,065
            # clears none. These two sit either side of 3,072, so changing
            # TUNE_MIN_HEADROOM_MIB now fails the suite instead of sliding by.
            (24, 65536, 13122, 3189, True),
            (23, 65536, 13653, 2658, False),
    ):
        n += 1
        est, head, ok = H.vram_verdict(q5, q5["params"], ctx=ctx, n=ncm)
        if (round(est), round(head), bool(ok)) != (want_est, want_free, want_ok):
            problems.append(
                "vram_verdict(n={}) -> ({}, {}, {}), expected ({}, {}, {})"
                .format(ncm, round(est), round(head), ok,
                        want_est, want_free, want_ok))
    # An uncalibrated model must NOT come back looking safe.
    n += 1
    if H.vram_verdict({"params": {}, "tuning": {}}, {}) != (None, None, None):
        problems.append("vram_verdict() with no calibration must return "
                        "(None, None, None) - an unknown model cannot be "
                        "reported as safe")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="also fail on the CON-006 --models-max check "
                         "(enabled once Phase 3 TASK-020 has run)")
    ap.add_argument("--self-test", action="store_true",
                    help="negative controls: run only the E-group checks "
                         "against deliberately broken bodies and report "
                         "whether each one fires")
    args = ap.parse_args()

    if args.self_test:
        bad = self_test()
        if bad:
            print("FAIL - {} negative control(s) did not fire".format(bad))
            return 1
        print("PASS - all E-group negative controls fired")
        return 0

    problems = []
    warnings = []

    if not REGISTRY.exists():
        print("[X] registry not found: {}".format(REGISTRY))
        return 1

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))

    for lname, cfg in reg["launchers"].items():
        script = cfg.get("script", lname)
        entries = cfg.get("entries", [])

        # --- B. unique goto labels
        gotos = [e.get("goto") for e in entries]
        dup = sorted({g for g in gotos if gotos.count(g) > 1})
        if dup:
            problems.append("{}: duplicate goto label(s) {}".format(script, dup))

        # --- E6. every %..._MM% reference must have a definition (FND-070):
        # the body called :check_file with %AUTO_x__MM% while no `set` line
        # ever minted it - the empty path killed the menu item before launch.
        defs = set()
        def _collect(text):
            defs.update(re.findall(r'set "([A-Z0-9_]+)=', text or ""))
        _collect(cfg.get("head", ""))
        for item in cfg.get("region", []):
            if item.get("t") == "static":
                _collect(item.get("s"))
            for v in item.get("vars") or []:
                _collect(v)
        refs = set()
        for e in entries:
            refs.update(re.findall(r"%([A-Z0-9_]+_MM)%", e.get("body", "") or ""))
        missing = sorted(r for r in refs if r not in defs)
        if missing:
            problems.append(
                "{}: body references mmproj variable(s) {} that no `set` "
                "line defines - :check_file receives an empty path and the "
                "menu item dies before launch (FND-070)".format(script, missing))

        for e in entries:
            label = e.get("label", "?")
            body = e.get("body", "")

            # --- A. blank line inside a caret continuation
            m = BLANK_CONT_RE.search(body)
            if m:
                line_no = body[:m.start()].count("\r\n") + 1
                problems.append(
                    "{}: entry '{}' has a blank line inside a caret "
                    "continuation (body line {})".format(script, label, line_no))

            # --- C. launcher must carry --port / --api-key
            if "llama-server.exe" in body:
                for arg in REQUIRED_ARGS:
                    if arg not in body:
                        problems.append(
                            "{}: entry '{}' launches llama-server.exe but "
                            "has no {}".format(script, label, arg))

    # --- D. models-config.bat: router launcher hygiene
    if MODELS_CONFIG.exists():
        raw = MODELS_CONFIG.read_text(encoding="utf-8", errors="replace")
        # only executable lines count - REM/#/; remarks may mention a flag
        exe = "\n".join(
            ln for ln in raw.splitlines()
            if not ln.strip().upper().startswith(("REM", "#", ";")))
        for flag, want, msg in (
            ("--models-max", 1,
             "--models-max appears {} time(s), expected 1 (CON-006)"),
            ("--models-dir", 0,
             "--models-dir still used {} time(s) - it duplicates preset "
             "entries and exposes gemma4_mtp (TASK-019)"),
            ("--mlock", 0,
             "--mlock appears {} time(s) - it silently overrides "
             "--load-mode (FND-013)"),
        ):
            n = exe.count(flag)
            if n != want:
                text = msg.format(n)
                (problems if args.strict else warnings).append(text)
        want_key = shared_api_key()
        if want_key is None:
            warnings.append("models-config.bat: could not read the shared "
                            "API_KEY from the registry - key check skipped "
                            "(FND-021)")
        elif want_key not in exe:
            problems.append("models-config.bat: --api-key is not the shared "
                            "key (FND-021)")
    else:
        problems.append("models-config.bat not found: {}".format(MODELS_CONFIG))

    # --- E. MoE / --fit regressions (TASK-082 / 089 / 091 / 092)
    # The override is the source of truth for "this model offloads experts", so
    # the tuned set is derived from it rather than from a GGUF metadata parse -
    # that keeps this file fast and dependency-free, and it matches exactly the
    # set find_moe_gaps() acts on.
    tuned = {}
    if OVERRIDES.exists():
        ov = json.loads(OVERRIDES.read_text(encoding="utf-8"))
        for rel, ent in (ov.get("models") or {}).items():
            n = (ent.get("params") or {}).get("n-cpu-moe")
            if n is not None:
                tuned[rel] = int(n)
    else:
        problems.append("preset-overrides.json not found: {}".format(OVERRIDES))

    for lname, cfg in reg["launchers"].items():
        script = cfg.get("script", lname)
        for e in cfg.get("entries", []):
            rel = (e.get("dirs") or [None])[0]
            problems.extend(check_entry(script, e, tuned.get(rel)))

    # --- F. pure helpers (llama_hub had no automated coverage at all)
    pure_count = check_pure(problems)

    # FND-063 again, but on the generated artefact: the user reads and copies
    # from the .bat, so a bad flag there is just as harmful as in the registry.
    for path in sorted(LAUNCHER_DIR.glob("*.bat")):
        txt = path.read_text(encoding="utf-8", errors="replace")
        if BAD_FLAG_RE.search(txt):
            problems.append("{}: contains '-fittarget' (FND-063)"
                            .format(path.name))

    if problems:
        print("FAIL ({} problem(s))".format(len(problems)))
        for p in problems:
            print("  [X] {}".format(p))
        for w in warnings:
            print("  [!] {}".format(w))
        return 1

    total = sum(len(c.get("entries", []))
                for c in reg["launchers"].values())
    print("PASS - {} launchers / {} entries checked".format(
        len(reg["launchers"]), total))
    print("  [OK] no blank-line caret continuation")
    print("  [OK] goto labels unique")
    print("  [OK] --port / --api-key present in every launch block")
    print("  [OK] no -fittarget, no deprecated --no-mmap")
    print("  [OK] every MoE block agrees with preset-overrides.json")
    print("  [OK] no --fit beside --n-cpu-moe, no -ngl beside --fit on")
    print("  [OK] pure helpers: retag_ctx / moe_cell / vram_verdict ({} case(s) "
          "against real measured data)".format(pure_count))
    if warnings:
        for w in warnings:
            print("  [!] {}".format(w))
    else:
        print("  [OK] models-config.bat --models-max count == 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
