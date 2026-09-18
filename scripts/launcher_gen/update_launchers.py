#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_launchers.py - llama.cpp launcher auto-updater
======================================================
Scans the configured model directory and keeps these artifacts in sync with
the models actually present on disk:

  1. start-CPU-Toolcall-Launcher.bat   (pure ASCII)
  2. start-Gemma4-Launcher.bat         (GBK / cp936, no BOM)
  3. start-Qwen-Launcher.bat           (GBK / cp936, no BOM)
  4. models-config.ini                 (Router Mode --models-preset)

Behavior:
  - Model dir deleted  -> its menu items / variables / ini sections are removed
    and the remaining menu is renumbered automatically.
  - New model dir      -> a default entry is generated from a per-family
    template (label marked [NEW]), added to the launcher and the registry.
  - Every run backs up changed files to backup\\ before writing.

Modes:
  --extract     Re-extract launcher-models.json from the current 3 launchers.
                (bootstrap / rebuild the registry, one-time operation)
  --check       Dry run: report changes, write preview copies under
                backup\\preview\\, write nothing real.
  --yes         Apply changes without the interactive confirmation.
  --no-scan     Render the registry verbatim (no disk scan). Used for
                byte-exact regression testing against the original scripts.
  --chat DIR    Override the model directory (used for testing).

Encoding rules (hard-won lessons, see repo memory):
  - The Gemma4/Qwen launchers MUST remain GBK(936), no BOM, no chcp line.
    We decode/encode strictly as gbk and never re-save them as UTF-8.
  - The CPU launcher and models-config.ini are pure ASCII.

Registry: launcher-models.json lives next to this script.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Where the launcher DATA lives: launcher-models.json, preset-overrides.json,
# model-profiles.json, the key cache, and backup/.
#
# Defaults to this file's own directory, so running the tool from inside a
# launcher/ folder keeps working unchanged. Set LAUNCHER_DIR when the tool lives
# somewhere else - as it does once installed in the skill, where the data belongs
# to the user rather than to the repository.
BASE = Path(os.environ.get("LAUNCHER_DIR") or Path(__file__).resolve().parent).resolve()

# Where the models live. There is deliberately NO baked-in default: the models
# path is a property of the machine, not of this tool. Set CHAT_DIR or pass
# --chat, and the tool says so plainly instead of guessing at one layout.
_ENV_CHAT = os.environ.get("CHAT_DIR")
DEFAULT_CHAT = Path(_ENV_CHAT) if _ENV_CHAT else None

MTP_DIR_NAME = "gemma4_mtp"
REGISTRY = BASE / "launcher-models.json"
BACKUP = BASE / "backup"
PREVIEW_DIR = BACKUP / "preview"

SPECS = {
    "gemma4": ("start-Gemma4-Launcher.bat", "gbk"),
    "qwen": ("start-Qwen-Launcher.bat", "gbk"),
    "cpu": ("start-CPU-Toolcall-Launcher.bat", "ascii"),
}

# ---------------------------------------------------------------- patterns
ITEM_LEAD_RE = re.compile(r"^echo\s+\d+\)\s")
EXIT_ITEM_RE = re.compile(r"^echo\s+\d+\)\s*(退出|Exit)\s*$")
ITEM_PARSE_RE = re.compile(r"^(echo\s+)(\d+)\)\s+(.*)$")
DISP_PARSE_RE = re.compile(r'^(if "%c%"==")(\d+)(" goto )(\S+)$')
EXIT_DISP_RE = re.compile(r'^(if "%c%"==")(\d+)(" exit)\s*$')
PROMPT_PARSE_RE = re.compile(r'^(set /p c=")(.*\[1-)(\d+)(\].*)$')
VAR_CHAT_RE = re.compile(r'^set "([A-Za-z0-9_]+)=%CHAT%\\(.+)\\([^\\"]+)\.gguf"$')
VAR_MTP_RE = re.compile(r'^set "([A-Za-z0-9_]+)=%MTP%\\([^\\"]+)\.gguf"$')
LABEL_RE = re.compile(r"^:RUN_[A-Za-z0-9_]+$")
PAUSE_END_RE = re.compile(r"^\s*pause & goto menu\s*$")
GOTO_END_RE = re.compile(r"^\s*goto RUN_\S+\s*$")
TOKEN_RE = re.compile(r"^\{(ITEM|DISP)(\d+)\}$")

MTP_PREFIXES = ["gemma-4-12b", "gemma-4-26B-A4B", "gemma-4-E4B"]
QUANT_RE = re.compile(
    r"(IQ\d_XS|IQ\d_S|Q\d_K_XL|Q\d_K_M|Q\d_K_S|Q\d_K_L|Q\d_K|Q\d_0|Q\d_1|BF16|F16|F32|Q8_0|Q4_0)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"(\d+)\s*[bB]")


def log(msg=""):
    print(msg, flush=True)


def read_text(path, enc):
    raw = Path(path).read_bytes()
    return raw.decode(enc), raw.endswith(b"\r\n")


def write_text(path, text, enc):
    Path(path).write_bytes(text.encode(enc, errors="strict"))


def ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def backup_file(path):
    BACKUP.mkdir(exist_ok=True)
    dst = BACKUP / f"{Path(path).name}.bak-{ts()}"
    shutil.copy2(path, dst)
    return dst


# ==================================================== curated profiles
PROFILES_PATH = BASE / "model-profiles.json"

# Hand-editable, machine-checked parameter source (Phase 2 / REQ-012).
# BASE-relative on purpose: Phase 5 moves this script into launcher\ and the
# data file follows automatically.
OVERRIDES_PATH = BASE / "preset-overrides.json"
KEYS_CACHE = BASE / ".llama-server-keys.json"

# Keys the generator adds on its own behalf (not llama-server options).
EXTRA_PRESET_KEYS = {"model", "mmproj", "alias"}

# Params that must never be emitted with an empty value.
PARAM_ORDER = [
    "ctx-size", "n-cpu-moe", "cpu-moe", "batch-size", "ubatch-size",
    "n-gpu-layers", "cache-type-k", "cache-type-v", "temp", "top-p",
    "top-k", "min-p", "presence-penalty", "repeat-penalty",
    "reasoning-budget", "metrics",
]


def sorted_params(params):
    """Deterministic key order so regeneration is byte-stable."""
    def rank(k):
        return (PARAM_ORDER.index(k), k) if k in PARAM_ORDER else (999, k)
    return sorted(params.items(), key=lambda kv: rank(kv[0]))


def find_server_exe():
    """llama-server.exe lives in the repo root; the updater may live in a
    sub-folder after Phase 5, so probe both locations."""
    for cand in (BASE / "llama-server.exe", BASE.parent / "llama-server.exe"):
        if cand.exists():
            return cand
    return None


def server_option_keys(force=False):
    """Long option names accepted by this llama-server build.

    A preset section may only contain keys llama-server understands - anything
    else makes the router refuse to start (FND-029). Returns a set, or None if
    the binary could not be queried (caller degrades to a warning).
    """
    exe = find_server_exe()
    if exe is None:
        return None
    build = ""
    try:
        import subprocess
        ver = subprocess.run([str(exe), "--version"], capture_output=True,
                             text=True, errors="replace", timeout=30)
        m = re.search(r"build\s+(\d+)", ver.stdout + ver.stderr)
        build = m.group(1) if m else ""
    except Exception:
        pass
    if not force and KEYS_CACHE.exists():
        try:
            cached = json.loads(KEYS_CACHE.read_text(encoding="utf-8"))
            if build and cached.get("build") == build and cached.get("keys"):
                return set(cached["keys"])
        except Exception:
            pass
    try:
        import subprocess
        out = subprocess.run([str(exe), "--help"], capture_output=True,
                             text=True, errors="replace", timeout=60)
        text = out.stdout + out.stderr
    except Exception as exc:
        log("[!] could not query {}: {}".format(exe.name, exc))
        return None
    keys = sorted(set(re.findall(r"--([a-z][a-z0-9]*(?:-[a-z0-9]+)*)", text)))
    if not keys:
        log("[!] no options parsed from --help - key whitelist disabled")
        return None
    try:
        KEYS_CACHE.write_text(json.dumps(
            {"build": build, "count": len(keys), "keys": keys},
            ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return set(keys)


def load_overrides():
    """Load preset-overrides.json, or return a skeleton if absent."""
    if not OVERRIDES_PATH.exists():
        return None
    with OVERRIDES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_overrides(ovr):
    if OVERRIDES_PATH.exists():
        backup_file(OVERRIDES_PATH)
    with OVERRIDES_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(ovr, f, ensure_ascii=False, indent=2)
        f.write("\n")


def measured_points(entry):
    """Measurement points for a tuning block, plus the origin of the numbers.

    FND-067: a `n-cpu-moe` optimum is only valid for the context it was measured
    at. `llama-bench` runs its own tiny default context (nearly no KV), so it
    never reaches the VRAM ceiling and therefore cannot locate the wall; the real
    Router runs 65536. Mixing the two produces nonsense - llama-bench's 85.5 t/s
    at n=16 would "beat" a real-context config measured at 46.4 - so they live in
    separate keys and are never merged.

    Returns (points, origin):
      'realctx'  - measured at the Router's own context: authoritative
      'measured' - legacy key, no context recorded: usable but unlabelled
      'benchctx' - llama-bench: ordering only, never pick a value from it
      None       - nothing measured yet
    """
    t = (entry or {}).get("tuning") or {}
    if t.get("measured_realctx"):
        return t["measured_realctx"], "realctx"
    if t.get("measured"):
        return t["measured"], "measured"
    if t.get("measured_benchctx"):
        return t["measured_benchctx"], "benchctx"
    return [], None


def override_param(overrides, reldir, key, default=None, cast=None):
    """`params.<key>` from preset-overrides.json, or `default` when absent.

    preset-overrides.json is the single source of truth for tunables (see
    TUNABLE_KEYS). make_auto() and gen_ini_v2() must both read it - when only
    one of them did, the Router and the .bat launchers silently disagreed about
    the same model (FND-064a).
    """
    try:
        ent = model_entry_for(overrides or {}, reldir) or {}
    except NameError:
        return default
    v = (ent.get("params") or {}).get(key)
    if v is None:
        return default
    if cast is not None:
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default
    return v


def override_ncpu_moe(overrides, reldir):
    """Explicit `params.n-cpu-moe` from preset-overrides.json, or None.

    FND-064a: make_auto() used to derive --n-cpu-moe purely from the profile's
    `moe.start_ratio` and never look at preset-overrides.json. Since the ini
    generator DOES read that file, the Router and the .bat launchers computed
    the value from two different sources and drifted apart permanently once a
    value was measured (30 in the launcher, 20 in the ini, for the same model).
    """
    return override_param(overrides, reldir, "n-cpu-moe", None, int)

def strip_decorative_fit(text):
    """Drop `--fit on` / `--fit-ctx N` from a command that uses `--n-cpu-moe`.

    FND-066: `--n-cpu-moe N` compiles down to -ot tensor overrides on
    blk.0..N-1 ffn_*_exps, and common_fit_params() bails out the moment
    tensor_buft_overrides is non-empty. The two never cooperate - the fit never
    runs, so those flags only advertise a safety net that does not exist.

    Verified by A/B on 26B-A4B at 131072 ctx (plan/_ab-g2-{fit,nofit}.json):
      with `--fit on -fitc 131072`  -> 10,398 MiB / 39.7 t/s
      without                       -> 10,398 MiB / 40.3 t/s
    Same VRAM to the MiB, speed delta inside run-to-run noise.

    Deleting a token that sat alone on a continuation line leaves a bare `^`
    (or `  ^`); that line is removed too, otherwise cmd.exe keeps a pointless
    caret and the block can drift into an FND-014 blank-line continuation.
    """
    if not text:
        return text
    t = text
    # the specific spellings first, then the bare --fit/-fit
    t = re.sub(r"(?<!\S)(--fit-ctx|-fitc)\s+\S+\s?", "", t)
    t = re.sub(r"(?<!\S)(--fit|-fit)\s+(on|off)\s?", "", t)
    # a continuation line left holding nothing but a caret
    t = re.sub(r"(?m)^[ \t]*\^[ \t]*(\r?\n|$)", "", t)
    return t


KV_BYTES_PER_ELEM = {
    "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32,
    "q8_0": 34 / 32, "q8_1": 40 / 32, "f16": 2.0, "bf16": 2.0, "f32": 4.0,
}

# ggml_type enum -> bytes per stored weight. Block-quantised types pack many
# weights into one fixed-size block, so the unit here is bytes-per-weight, not
# bytes-per-block (Q4_K = 144 bytes per 256 weights = 0.5625 B/w).
#
# Getting one of these wrong would silently shift every VRAM estimate, so the
# parser does not trust the table on its own: read_gguf_tensors() adds up the
# whole table and compares the result against the real file size, which is an
# independent check that fails loudly if a number here is wrong.
GGML_TYPE_BYTES = {
    0: 4.0,                    # F32
    1: 2.0,                    # F16
    2: 18 / 32,                # Q4_0
    3: 20 / 32,                # Q4_1
    6: 22 / 32,                # Q5_0
    7: 24 / 32,                # Q5_1
    8: 34 / 32,                # Q8_0
    9: 36 / 32,                # Q8_1
    10: 84 / 256,              # Q2_K
    11: 110 / 256,             # Q3_K
    12: 144 / 256,             # Q4_K
    13: 176 / 256,             # Q5_K
    14: 210 / 256,             # Q6_K
    15: 292 / 256,             # Q8_K
    16: 66 / 256,              # IQ2_XXS
    17: 74 / 256,              # IQ2_XS
    18: 98 / 256,              # IQ3_XXS
    19: 50 / 256,              # IQ1_S
    20: 18 / 32,               # IQ4_NL
    21: 110 / 256,             # IQ3_S
    22: 82 / 256,              # IQ2_S
    23: 136 / 256,             # IQ4_XS
    24: 1.0,                   # I8
    25: 2.0,                   # I16
    26: 4.0,                   # I32
    27: 8.0,                   # I64
    28: 8.0,                   # F64
    29: 56 / 256,              # IQ1_M
    30: 2.0,                   # BF16
    34: 54 / 256,              # TQ1_0
    35: 66 / 256,              # TQ2_0
    39: 17 / 32,               # MXFP4
}


def load_profiles():
    if not PROFILES_PATH.exists():
        return []
    with PROFILES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("profiles", [])


def _norm_name(s):
    """Fold away the separators that community re-uploads love to move around."""
    return re.sub(r"[-_.\s]+", "", str(s).lower())


def _size_token(s):
    """The parameter-size token of a model name ('35' from Qwen3.6-35B-A3B).

    Run on the RAW name, not the normalized one: normalisation strips the
    separators, and `qwen3635ba3b` no longer has a boundary the regex can use.
    The lookarounds are what keep `A3B` from contributing a bogus `3b`.
    """
    m = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])",
                  str(s).lower())
    return m.group(1) if m else None


def _family(s):
    """Leading alphabetic family of a model name ('gemma', 'qwen', 'lfm')."""
    m = re.match(r"([a-z]+)", _norm_name(s))
    return m.group(1) if m else ""


def profile_match(dirname, profiles):
    """Best profile for a model directory, plus HOW it was matched.

    Returns `(profile_or_None, how)` where `how` is one of:

      'exact'      - pass 1, a literal substring match (historical behaviour)
      'normalized' - pass 2, separators stripped, so `gemma-4-26b` also matches
                     the `Gemma4-26B-A4B-QAT-Uncensored-...` community builds
      'neighbour'  - pass 3, TASK-078: same FAMILY and same PARAMETER SIZE as
                     an existing profile, e.g. an uncensored re-upload of a
                     model whose size is already profiled
      None         - nothing close enough; the caller must fall back to the
                     generic defaults and say so

    Pass 3 is deliberately conservative. It requires both the family and the
    size to agree, because those are the two things that make a profile's
    `start_ratio` transferable - a ratio measured on a 26B is meaningless on a
    12B, however similar the names look. It is also strictly additive: it runs
    only when passes 1-2 found nothing, so it can never change which profile an
    existing model resolves to.

    The `how` is returned rather than swallowed because the caller records it in
    `tuning.source` (`inherited-from:<id>`), and "the tool computed this" and
    "this was inherited from a lookalike" are very different levels of trust.
    """
    low = dirname.split("/")[-1].lower()
    for p in profiles:
        if all(s.lower() in low for s in p.get("match", [])):
            return p, "exact"
    nlow = _norm_name(low)
    if not nlow:
        return None, None
    for p in profiles:
        toks = [_norm_name(s) for s in p.get("match", [])]
        if toks and all(t and t in nlow for t in toks):
            return p, "normalized"
    fam = _family(low)
    size = _size_token(low)
    if not fam or not size:
        return None, None
    for p in profiles:
        if _family(p.get("id") or (p.get("match") or [""])[0]) != fam:
            continue
        if _size_token(p.get("id") or (p.get("match") or [""])[0]) != size:
            continue
        return p, "neighbour"
    return None, None


def profile_for_dirname(dirname, profiles):
    """Backwards-compatible wrapper - see profile_match() for the real logic."""
    return profile_match(dirname, profiles)[0]


# ==================================================== GGUF metadata
def read_gguf_meta(path, limit=1 << 20):
    """Parse GGUF header key-values (pure stdlib, reads only the head)."""
    import struct
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except OSError:
        return {}
    if len(data) < 16 or data[:4] != b"GGUF":
        return {}
    off = 8
    n_tensors, n_kv = struct.unpack_from("<QQ", data, off)
    off += 16
    out = {}
    scalar_fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                  6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    scalar_size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}
    for _ in range(n_kv):
        if off + 8 > len(data):
            break
        (klen,) = struct.unpack_from("<Q", data, off)
        off += 8
        key = data[off:off + klen].decode("utf-8", "replace")
        off += klen
        if off + 4 > len(data):
            break
        (vtype,) = struct.unpack_from("<I", data, off)
        off += 4
        if vtype == 8:                       # string
            if off + 8 > len(data):
                break
            (slen,) = struct.unpack_from("<Q", data, off)
            off += 8
            out[key] = data[off:off + slen].decode("utf-8", "replace")
            off += slen
        elif vtype == 9:                     # array
            if off + 12 > len(data):
                break
            atype, alen = struct.unpack_from("<IQ", data, off)
            off += 12
            if atype == 8:                   # array of strings
                vals = []
                for _ in range(alen):
                    if off + 8 > len(data):
                        break
                    (slen,) = struct.unpack_from("<Q", data, off)
                    off += 8
                    vals.append(data[off:off + slen].decode("utf-8", "replace"))
                    off += slen
                out[key] = vals
            elif atype in scalar_fmt:        # numeric array
                esz = scalar_size[atype]
                out[key] = [
                    struct.unpack_from("<" + scalar_fmt[atype], data, off + i * esz)[0]
                    for i in range(min(alen, 1024))]
                off += esz * alen
            else:
                off += scalar_size.get(atype, 0) * alen
        elif vtype in scalar_fmt:
            esz = scalar_size[vtype]
            if off + esz > len(data):
                break
            out[key] = struct.unpack_from("<" + scalar_fmt[vtype], data, off)[0]
            off += esz
        else:
            break
    arch = out.get("general.architecture", "")
    def g(k, default=None):
        return out.get(k, out.get("general." + k.split(".", 1)[-1], default))
    return {
        "arch": arch,
        "name": out.get("general.name", ""),
        "ctx_train": g(arch + ".context_length"),
        "layers": g(arch + ".block_count"),
        "kv_heads": g(arch + ".attention.head_count_kv"),
        "key_len": g(arch + ".attention.key_length"),
        "value_len": g(arch + ".attention.value_length"),
        "swa": g(arch + ".attention.sliding_window"),
        "swa_pattern": g(arch + ".attention.sliding_window_pattern"),
        "interval": g(arch + ".full_attention_interval"),
        "experts": g(arch + ".expert_count"),
    }


# Expert FFN tensors are what `--n-cpu-moe` moves, and their names come in two
# shapes that do NOT overlap:
#
#   Qwen   ffn_gate_exps / ffn_up_exps / ffn_down_exps   (split)
#   Gemma  ffn_gate_up_exps / ffn_down_exps             (FUSED)
#
# The first version of this pattern only listed the split names. It still
# produced a perfect file-size cross-check, because the total is unaffected -
# only the expert/non-expert SPLIT was wrong, and it was wrong in the direction
# that matters: Gemma came out at 30% expert instead of ~88%. The cross-check
# cannot catch a classification error, so read_gguf_tensors() also returns the
# matched name patterns and the probe prints them. If a model ever ships a third
# naming shape, that listing is where it shows up.
#
# Deliberately NOT expert tensors, despite the similar names:
#   ffn_gate_inp_shexp   the router that picks experts
#   ffn_*_shexp          the always-on shared experts
# Neither is touched by `--n-cpu-moe`, so counting them would overstate E_layer.
EXPERT_TENSOR_RE = re.compile(r"\.ffn_(?:gate_up|gate|up|down)_exps\.")
BLOCK_RE = re.compile(r"^blk\.(\d+)\.")
EXPERT_SHAPE_RE = re.compile(r"^blk\.\d+\.(.+)$")


def read_gguf_tensors(path):
    """Parse the GGUF tensor table; split expert from non-expert bytes.

    read_gguf_meta() reads a fixed 1 MiB from the head. That is enough for the
    architecture keys, which come first, but it can never reach the tensor table:
    the table sits after the ENTIRE key-value block, and the tokenizer arrays
    alone run to megabytes on a 256K-vocab model. So this walks the file and
    SKIPS each KV value by its declared size instead of reading it.

    Why it matters (TASK-074): the budget equation was guessing that 88% of a
    MoE checkpoint is expert weights (MOE_EXPERT_SHARE). That guess is the only
    thing standing between "an n-cpu-moe that loads" and one that OOMs, and it
    is checkable for free - the tensor table says exactly how many bytes are
    expert bytes, and the answer is per-model, not a constant.

    Returns {} when the file is not a readable GGUF v2/v3, else:

        {n_tensors, total_gb, expert_gb, non_expert_gb, e_layer_gb,
         layers, per_layer, uniform_pct, size_ok, delta_pct}

    `size_ok` is the self-check: the summed table must equal the file size minus
    the tensor-data offset. A mismatch means GGML_TYPE_BYTES has a wrong entry,
    and every number in the result is then suspect - callers must treat
    size_ok=False as "no data", not as a slightly-off estimate.
    """
    import struct
    try:
        f = open(path, "rb")
    except OSError:
        return {}
    try:
        head = f.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            return {}
        ver, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
        if ver < 2:
            return {}

        def rd(fmt, size):
            b = f.read(size)
            if len(b) != size:
                raise EOFError
            return struct.unpack("<" + fmt, b)[0]

        scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                  10: 8, 11: 8, 12: 8}
        for _ in range(n_kv):
            f.seek(rd("Q", 8), 1)               # key name - not needed here
            vtype = rd("I", 4)
            if vtype == 8:                       # string
                f.seek(rd("Q", 8), 1)
            elif vtype == 9:                     # array
                atype = rd("I", 4)
                alen = rd("Q", 8)
                if atype == 8:                   # array of strings
                    for _ in range(alen):
                        f.seek(rd("Q", 8), 1)
                elif atype in scalar:
                    f.seek(scalar[atype] * alen, 1)
                else:
                    return {}
            elif vtype in scalar:
                f.seek(scalar[vtype], 1)
            else:
                return {}

        total = 0
        expert = 0
        per_layer = {}
        shapes = {}
        layer_ids = set()
        for _ in range(n_tensors):
            nlen = rd("Q", 8)
            name = f.read(nlen).decode("utf-8", "replace")
            nd = rd("I", 4)
            n_elem = 1
            for _ in range(nd):
                n_elem *= rd("Q", 8)
            ttype = rd("I", 4)
            f.seek(8, 1)                         # data offset - not needed
            bpe = GGML_TYPE_BYTES.get(ttype)
            if bpe is None:
                return {}                        # unknown quant - refuse to guess
            nbytes = int(n_elem * bpe)
            total += nbytes
            bm = BLOCK_RE.match(name)
            if bm:
                layer_ids.add(int(bm.group(1)))
            if EXPERT_TENSOR_RE.search(name):
                expert += nbytes
                if bm:
                    lay = int(bm.group(1))
                    per_layer[lay] = per_layer.get(lay, 0) + nbytes
                sm = EXPERT_SHAPE_RE.match(name)
                if sm:
                    shapes[sm.group(1)] = shapes.get(sm.group(1), 0) + 1

        data_off = f.tell()
        data_off = (data_off + 31) // 32 * 32      # aligned to 32 bytes
        try:
            size = os.path.getsize(path)
        except OSError:
            return {}
        real = size - data_off
        delta = abs(total - real) / real * 100 if real else 100.0

        vals = sorted(per_layer.values())
        uniform = 0.0
        if vals and vals[-1]:
            # spread of the per-layer expert budget. A model that loads the
            # same tensors in every layer lands at 0%; anything appreciable
            # means a flat "n layers off" is the wrong model of reality.
            uniform = (vals[-1] - vals[0]) / vals[-1] * 100
        gb = 1 << 30
        return {
            "n_tensors": n_tensors,
            "total_gb": total / gb,
            "expert_gb": expert / gb,
            "non_expert_gb": (total - expert) / gb,
            "e_layer_gb": (expert / len(per_layer) / gb) if per_layer else 0.0,
            "layers": len(per_layer),
            "per_layer": {k: v / gb for k, v in sorted(per_layer.items())},
            "uniform_pct": uniform,
            "size_ok": delta < 1.0,
            "delta_pct": delta,
            "expert_shapes": shapes,
            "layer_ids": sorted(layer_ids),
        }
    except (EOFError, struct.error, OSError):
        return {}
    finally:
        f.close()


def kv_estimate_gb(meta, ctx, kv_type):
    """KV cache size estimate in decimal GB (K+V), SWA-aware."""
    layers = meta.get("layers")
    kl = meta.get("key_len")
    vl = meta.get("value_len")
    if not layers or not ctx or kl is None or vl is None:
        return None
    bpe = KV_BYTES_PER_ELEM.get(kv_type, 2.0)
    kv_heads = meta.get("kv_heads")
    if isinstance(kv_heads, list):
        elems = kl + vl          # key/value_length are total KV dims here
    else:
        elems = (kv_heads or 1) * (kl + vl)   # per-head dims
    swa = meta.get("swa")
    pattern = meta.get("swa_pattern")
    interval = meta.get("interval")
    if swa and isinstance(pattern, list) and len(pattern) == layers:
        n_global = sum(1 for p in pattern if not p)
        n_swa = layers - n_global
        return (n_global * elems * ctx + n_swa * elems * min(ctx, swa)) * bpe / 1e9
    if interval:
        n_full = (layers + interval - 1) // interval
        n_swa = layers - n_full
        w = swa or 4096
        return (n_full * elems * ctx + n_swa * elems * min(ctx, w)) * bpe / 1e9
    return layers * elems * ctx * bpe / 1e9


def build_varmap(cfg):
    """Collect `set "VAR=value"` from head + region; resolve %VAR% chains."""
    texts = [cfg["head"]]
    for it in cfg["region"]:
        if it.get("t") == "static":
            texts.append(it["s"])
        else:
            texts.extend(it.get("comments") or [])
            texts.extend(it.get("vars") or [])
    raw = {}
    for txt in texts:
        for m in re.finditer(r'^set "(\w+)=([^"]*)"\s*$', txt, re.M):
            raw[m.group(1)] = m.group(2)
    def resolve(v):
        for _ in range(5):
            v2 = re.sub(r"%(\w+)%", lambda m: raw.get(m.group(1), m.group(0)), v)
            if v2 == v:
                break
            v = v2
        return v
    return {k: resolve(v) for k, v in raw.items()}


def block_args(body, varmap):
    """Extract llama-server args from a launch block (resolves %VAR%)."""
    toks = body.replace("^", " ").split()
    args = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key, _, val = t[2:].partition("=")
            if val:
                args[key] = val
            elif i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                args[key] = toks[i + 1]
                i += 1
        elif re.match(r"^-[a-zA-Z]{1,3}$", t) and i + 1 < len(toks) \
                and not toks[i + 1].startswith("-"):
            args[t[1:]] = toks[i + 1]
            i += 1
        i += 1
    return {k: re.sub(r"%(\w+)%",
                      lambda m: varmap.get(m.group(1), m.group(0)), v)
            for k, v in args.items()}


def audit_registry(reg, chat, profiles):
    """Compare every registry entry with curated profiles + VRAM estimate."""
    dirs, _ = scan_chat(chat)
    log("== parameter audit (registry vs curated profiles) ==")
    log("VRAM assumed: 16 GB. Estimate = weights(ngl share) + KV(ngl share)"
        " + draft + mmproj")
    log("")
    for lname, cfg in reg["launchers"].items():
        varmap = build_varmap(cfg)
        log("[{}]".format(cfg["script"]))
        for e in cfg["entries"]:
            label = e["label"][:44]
            body = e.get("body", "")
            a = block_args(body, varmap)
            d = e.get("dirs", [""])[0]
            files = dirs.get(d) or {}
            mains = files.get("mains") or []
            gguf = str(Path(chat) / d / mains[0]) if mains else None
            meta = read_gguf_meta(gguf) if gguf else {}
            profile = profile_for_dirname(d or "", profiles)
            # --- size / VRAM estimate
            wgb = os.path.getsize(gguf) / 1e9 if gguf and os.path.exists(gguf) else None
            draft = e.get("drafts") or []
            dgb = sum(os.path.getsize(str(Path(chat) / MTP_DIR_NAME / f)) / 1e9
                      for f in draft
                      if os.path.exists(str(Path(chat) / MTP_DIR_NAME / f)))
            mmgb = 0.0
            mml = files.get("mmproj") or []
            mm_warn = ""
            if len(mml) > 1:
                mm_warn = "mmproj x{} (using {})".format(len(mml), mml[0])
            if mml:
                mp = str(Path(chat) / d / mml[0])
                if os.path.exists(mp):
                    mmgb = os.path.getsize(mp) / 1e9
            ctx = a.get("c")
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = None
            kvg = kv_estimate_gb(meta, ctx, a.get("cache-type-k", "f16"))
            ngl = a.get("ngl")
            try:
                ngl = int(ngl)
            except (TypeError, ValueError):
                ngl = 99
            layers = meta.get("layers") or 0
            frac = min(1.0, ngl / layers) if layers else 0.0
            gpu_est = ((wgb or 0) + (kvg or 0)) * frac + dgb + mmgb
            if gpu_est <= 13.5:
                verdict = "OK"
            elif gpu_est <= 16.0:
                verdict = "tight"
            else:
                verdict = "OFFLOAD(layers/KV to CPU)"
            arch = meta.get("arch") or "?"
            prof_src = (profile.get("sources", [{}])[0].get("verified", "?")
                        if profile else "-")
            # --- sampling vs profile
            warns = []
            if mm_warn:
                warns.append(mm_warn)
            if profile:
                samp = profile.get("sampling") or {}
                for key, arg in (("temp", "temp"), ("top_p", "top-p"),
                                 ("top_k", "top-k")):
                    if key in samp:
                        cur = a.get(arg)
                        try:
                            if abs(float(cur) - float(samp[key])) > 0.005:
                                warns.append("{}={} (official {})".format(
                                    arg, cur, samp[key]))
                        except (TypeError, ValueError):
                            pass
                if ctx and profile.get("ctx_max") and ctx > profile["ctx_max"]:
                    warns.append("c={} > ctx_max {}".format(
                        ctx, profile["ctx_max"]))
            log("  {:<44} {} L{} KV@{:<5} kv={:<6} W={:<6} estGPU={:<6} {}"
                .format(label, arch, layers, ctx or "?",
                        "{:.2f}G".format(kvg) if kvg else "?",
                        "{:.2f}G".format(wgb) if wgb else "?",
                        "{:.2f}G".format(gpu_est), verdict))
            if warns:
                log("      [!] {}".format("; ".join(warns)))
            if profile:
                log("      profile: {} [{}] ctx_max={}".format(
                    profile["id"], prof_src, profile.get("ctx_max")))
            else:
                log("      profile: none (family template only)")
        log("")





# ================================================================ extraction
def extract_launcher(name):
    """Parse one launcher .bat into registry pieces (see schema above)."""
    script, enc = SPECS[name]
    path = BASE / script
    if not path.is_file():
        # --extract rebuilds launcher-models.json FROM existing launcher files,
        # so a machine that has none cannot be bootstrapped this way - which is
        # the situation on every fresh install. Say that plainly instead of
        # dying with a FileNotFoundError from three frames down.
        raise SystemExit(
            "[X] {} not found in {}\n"
            "    --extract rebuilds launcher-models.json FROM existing launcher\n"
            "    files, so it needs at least one to work from, and a fresh\n"
            "    install has none. See scripts/launcher_gen/README.md for the\n"
            "    cold-start path and the registry schema."
            .format(script, BASE))
    text, trailing = read_text(path, enc)
    lines = text.split("\r\n")
    if trailing and lines and lines[-1] == "":
        lines.pop()

    var_start = next(
        (i for i, ln in enumerate(lines) if VAR_CHAT_RE.match(ln)), None
    )
    if var_start is None:
        raise SystemExit(f"[X] {script}: no model path variables found")
    head = "\r\n".join(lines[:var_start])

    region_end = next(
        (i for i in range(var_start, len(lines))
         if lines[i].strip().lower() == "goto :menu"), None
    )
    if region_end is None:
        raise SystemExit(f"[X] {script}: no 'goto :menu' found")

    region = []
    pending_comments = []
    i = var_start
    while i < region_end:
        ln = lines[i]
        if VAR_CHAT_RE.match(ln):
            grp = {"comments": pending_comments, "vars": []}
            pending_comments = []
            while i < region_end and VAR_CHAT_RE.match(lines[i]):
                grp["vars"].append(lines[i])
                i += 1
            region.append({"t": "group", **grp})
            continue
        if ln.lstrip().startswith("REM"):
            pending_comments.append(ln)
            i += 1
            continue
        if pending_comments:
            region.extend({"t": "static", "s": c} for c in pending_comments)
            pending_comments = []
        region.append({"t": "static", "s": ln})
        i += 1
    if pending_comments:
        region.extend({"t": "static", "s": c} for c in pending_comments)

    # --- menu region
    menu_start = next(
        (j for j in range(region_end + 1, len(lines))
         if ITEM_LEAD_RE.match(lines[j])), None
    )
    if menu_start is None:
        raise SystemExit(f"[X] {script}: no menu item lines found")
    menu_open = "\r\n".join(lines[region_end:menu_start])

    entries = []            # {"label", "goto", ...}
    menu_lines = []         # tokens + verbatim static lines
    item_tpl = dispatch_tpl = prompt_tpl = exit_line_tpl = exit_dispatch_tpl = None
    disp_pairs = []         # (num, target)
    item_nums = []
    j = menu_start
    exit_disp_idx = None
    while j < len(lines):
        ln = lines[j]
        if EXIT_ITEM_RE.match(ln):
            exit_line_tpl = re.sub(r"\d+\)", "{EXIT})", ln, count=1)
            menu_lines.append("{EXITLINE}")
            j += 1
            continue
        m = ITEM_PARSE_RE.match(ln)
        if m:
            num, label = int(m.group(2)), m.group(3)
            item_nums.append(num)
            entries.append({"label": label})
            menu_lines.append("{{ITEM{}}}".format(len(entries) - 1))
            # per-item echo prefix differs between items in some scripts
            # (e.g. two spaces for 1-9, one space for 10+) - keep exact.
            entries[-1]["item_tpl"] = "{}{{N}}) {{LABEL}}".format(m.group(1))
            if item_tpl is None:
                item_tpl = entries[-1]["item_tpl"]
            j += 1
            continue
        m = PROMPT_PARSE_RE.match(ln)
        if m:
            prompt_tpl = "{}{}{{MAX}}{}".format(m.group(1), m.group(2), m.group(4))
            menu_lines.append("{PROMPT}")
            j += 1
            continue
        m = DISP_PARSE_RE.match(ln)
        if m:
            num, target = int(m.group(2)), m.group(4)
            if target == "exit":
                exit_dispatch_tpl = "{}{{EXIT}}{}exit".format(
                    m.group(1), m.group(3))
                menu_lines.append("{EXITDISP}")
                exit_disp_idx = j
                break
            disp_pairs.append((num, target))
            menu_lines.append("{{DISP{}}}".format(len(disp_pairs) - 1))
            j += 1
            continue
        m = EXIT_DISP_RE.match(ln)
        if m:
            exit_dispatch_tpl = "{}{{EXIT}}{}".format(m.group(1), m.group(3))
            menu_lines.append("{EXITDISP}")
            exit_disp_idx = j
            break
        menu_lines.append(ln)      # group header / blank -> verbatim
        j += 1

    if exit_disp_idx is None:
        raise SystemExit(f"[X] {script}: no exit dispatch line found")
    # sanity: menu items / dispatches are strictly sequential 1..N
    n = len(entries)
    if item_nums != list(range(1, n + 1)):
        raise SystemExit(f"[X] {script}: non-sequential menu numbers {item_nums}")
    if [p[0] for p in disp_pairs] != list(range(1, n + 1)):
        raise SystemExit(f"[X] {script}: non-sequential dispatch numbers")
    for k, (_, target) in enumerate(disp_pairs):
        entries[k]["goto"] = target

    # --- blocks region
    post_start = exit_disp_idx + 1
    first_label = next(
        (j for j in range(post_start, len(lines)) if LABEL_RE.match(lines[j])),
        None)
    if first_label is None:
        raise SystemExit(f"[X] {script}: no :RUN_ labels found")
    post_dispatch = "\r\n".join(lines[post_start:first_label])

    label_idxs = [j for j in range(first_label, len(lines))
                  if LABEL_RE.match(lines[j])]
    if len(label_idxs) != n:
        raise SystemExit(
            f"[X] {script}: {len(label_idxs)} launch blocks vs {n} menu items")
    for k, li in enumerate(label_idxs):
        end = label_idxs[k + 1] if k + 1 < len(label_idxs) else len(lines)
        chunk = lines[li:end]
        end_cands = []
        for ci, cln in enumerate(chunk):
            if PAUSE_END_RE.match(cln) or GOTO_END_RE.match(cln):
                end_cands.append(ci)
        if not end_cands:
            raise SystemExit(
                f"[X] {script}: block {chunk[0]} has no pause/goto terminator")
        body_end = max(end_cands)
        body = "\r\n".join(chunk[:body_end + 1])
        banner = []
        for bln in chunk[body_end + 1:]:
            if re.search(r"\d+\)", bln):
                bln = re.sub(r"\d+\)", "{N})", bln, count=1)
            banner.append(bln)
        entries[k]["body"] = body
        # banner between this block and the next label belongs to the
        # FOLLOWING entry (it carries that entry's menu number)
        if k + 1 < len(entries):
            entries[k + 1]["banner"] = banner
        elif "banner" not in entries[k]:
            entries[k]["banner"] = []

    # --- resolve variable references inside blocks
    chat_map = {}
    draft_map = {}
    for item in region:
        if item["t"] == "group":
            for vln in item["vars"]:
                m = VAR_CHAT_RE.match(vln)
                chat_map[m.group(1)] = m.group(2)
        else:
            m = VAR_MTP_RE.match(item["s"])
            if m:
                draft_map[m.group(1)] = m.group(2) + ".gguf"
    for e in entries:
        refs = set(re.findall(r"%([A-Za-z0-9_]+)%", e["body"]))
        e["dirs"] = sorted({chat_map[r] for r in refs if r in chat_map})
        e["drafts"] = sorted({draft_map[r] for r in refs if r in draft_map})
        e["auto"] = False
    # agent-variant entries only `goto RUN_X` another block - inherit the
    # target entry's model dirs so they are removed together with the model
    for e in entries:
        if not e["dirs"]:
            m = re.search(r"goto (RUN_\S+)", e["body"])
            if m:
                target = m.group(1)
                for t in entries:
                    if t.get("goto") == target:
                        e["dirs"] = list(t["dirs"])
                        e["drafts"] = list(t.get("drafts", []))
                        break

    return {
        "script": script,
        "encoding": enc,
        "head": head,
        "region": region,
        "menu_open": menu_open,
        "menu_lines": menu_lines,
        "item_tpl": item_tpl,
        "dispatch_tpl": "if \"%c%\"==\"{N}\" goto {GOTO}",
        "prompt_tpl": prompt_tpl,
        "exit_line_tpl": exit_line_tpl,
        "exit_dispatch_tpl": exit_dispatch_tpl,
        "post_dispatch": post_dispatch,
        "trailing_newline": trailing,
        "entries": entries,
    }


# ================================================================= rendering
def var_dir_of(vline):
    """Relative model dir a `set "X=%CHAT%\\dir\\file.gguf"` line points to."""
    m = VAR_CHAT_RE.match(vline)
    return m.group(2) if m else None


def render_launcher(cfg, entries, extra_groups, dirs=None):
    """Render one launcher .bat from registry pieces + filtered entries."""
    gotos = [e.get("goto") for e in entries]
    dup = sorted({g for g in gotos if gotos.count(g) > 1})
    if dup:
        raise SystemExit(
            "[X] {}: duplicate goto label(s) {} - fix launcher-models.json "
            "before generating (would create unreachable menu blocks)"
            .format(cfg["script"], dup))
    # FND-014 guard: an empty physical line inside a `^` continuation makes
    # cmd.exe silently drop every argument after it. Never ship such a body.
    blank_cont = re.compile(r"\^[ \t]*\r?\n[ \t]*\r?\n")
    for e in entries:
        if blank_cont.search(e.get("body", "")):
            raise SystemExit(
                "[X] {}: entry '{}' has a blank line inside a caret "
                "continuation - cmd.exe would silently drop all args after "
                "it (FND-014)".format(cfg["script"], e.get("label", "?")))
    out = [cfg["head"]]

    # ---- variable region
    for item in cfg["region"]:
        if item.get("t") == "static":
            out.append(item["s"])
            continue
        emitted = list(item.get("comments") or [])
        for vln in item["vars"]:
            d = var_dir_of(vln)
            keep = (dirs is None) or (d in dirs)
            if keep:
                emitted.append(vln)
        if len(emitted) > len(item["comments"]):
            out.extend(emitted)
        # else: whole group dropped (comments included)
    for grp in extra_groups:
        out.extend(grp["comments"])
        out.extend(grp["vars"])
    out.append(cfg["menu_open"])

    # ---- menu region
    maxn = len(entries)
    n_orig = sum(1 for t in cfg["menu_lines"]
                 if re.match(r"\{ITEM\d+\}$", t))
    for tok in cfg["menu_lines"]:
        if tok == "{PROMPT}":
            out.append(cfg["prompt_tpl"].format(MAX=maxn))
        elif tok == "{EXITLINE}":
            if maxn > n_orig:  # extra entries beyond the original token slots
                if entries[n_orig].get("auto"):
                    out.append("echo --- [NEW] AUTO ---")
                for k in range(n_orig, maxn):
                    tpl = entries[k].get("item_tpl") or cfg["item_tpl"]
                    out.append(tpl.format(N=k + 1, LABEL=entries[k]["label"]))
            out.append(cfg["exit_line_tpl"].format(EXIT=maxn + 1))
        elif tok == "{EXITDISP}":
            if maxn > n_orig:
                for k in range(n_orig, maxn):
                    out.append(cfg["dispatch_tpl"].format(
                        N=k + 1, GOTO=entries[k]["goto"]))
            out.append(cfg["exit_dispatch_tpl"].format(EXIT=maxn + 1))
        else:
            m = TOKEN_RE.match(tok)
            if not m:
                out.append(tok)
                continue
            kind, idx = m.group(1), int(m.group(2))
            if idx >= maxn:
                continue
            if kind == "ITEM":
                tpl = entries[idx].get("item_tpl") or cfg["item_tpl"]
                out.append(tpl.format(N=idx + 1, LABEL=entries[idx]["label"]))
            else:
                out.append(cfg["dispatch_tpl"].format(
                    N=idx + 1, GOTO=entries[idx]["goto"]))

    # ---- launch blocks
    out.append(cfg["post_dispatch"])
    blocks = []
    for i, e in enumerate(entries):
        banner = e.get("banner") or []
        if banner:
            s = "\r\n".join(banner).replace("{N}", str(i + 1))
            blocks.append(s + "\r\n" + e["body"])
        else:
            blocks.append(e["body"])
    out.append("\r\n".join(blocks))

    text = "\r\n".join(out)
    if cfg.get("trailing_newline"):
        text += "\r\n"
    return text


# ==================================================================== scan
def scan_chat(chat):
    """Return ({rel_dir: {"mains": [...], "mmproj": [...]}}, mtp_root_files)."""
    dirs = {}
    mtp_files = []
    for root, dnames, fnames in os.walk(chat):
        rel = os.path.relpath(root, chat)
        base = os.path.basename(root)
        if rel == ".":
            dnames[:] = [d for d in dnames if not d.startswith("_")]
            continue
        if base.lower() == MTP_DIR_NAME.lower():
            mtp_files = sorted(f for f in fnames if f.lower().endswith(".gguf"))
            dnames[:] = []
            continue
        if base.startswith("_"):
            dnames[:] = []
            continue
        ggufs = sorted(f for f in fnames if f.lower().endswith(".gguf"))
        mains = [f for f in ggufs if "mmproj" not in f.lower()]
        mmproj = [f for f in ggufs if "mmproj" in f.lower()]
        if len(mmproj) > 1:
            mmproj = sorted(
                mmproj,
                key=lambda f: Path(root).joinpath(f).stat().st_mtime,
                reverse=True)
        if mains:
            dirs[rel] = {"mains": mains, "mmproj": mmproj}
        dnames[:] = [d for d in dnames if not d.startswith("_")]
    return dirs, mtp_files


def family_of(reldir):
    """Return the family NAME (registry key of reg['family'])."""
    low = reldir.split("/")[-1].lower()
    for key in ("gemma", "qwen", "lfm", "phi", "glm", "gpt"):
        if key in low:
            return key
    return "default"


def find_quant(stem):
    m = QUANT_RE.search(stem)
    return m.group(1) if m else "?"


def find_size(stem):
    m = SIZE_RE.search(stem)
    return m.group(1) + "B" if m else ""


# ------------------------------------------------- auto entry generation
def make_auto(launcher, reldir, files, mtp_files, taken_names, profiles=None,
              chat=None, overrides=None):
    stem = reldir.split("/")[-1]
    low = stem.lower()
    quant = find_quant(stem)
    size = find_size(stem)
    main = files["mains"][0]
    mmproj_list = files.get("mmproj") or []
    mm = mmproj_list[0] if mmproj_list else None
    if len(mmproj_list) > 1:
        log("[!] {}: multiple mmproj files {} - using '{}' (newest)".format(
            reldir, [os.path.basename(x) for x in mmproj_list], mm))
    profile = profile_for_dirname(reldir, profiles or [])
    prof_samp = (profile or {}).get("sampling") or {}

    # ---- MoE detection (REQ-020/021) --------------------------------------
    # Read the GGUF header so the generated block can use --n-cpu-moe and
    # --fit instead of the old hard-coded `-ngl 99`, which put a 22GB MoE
    # fully on a 16GB card AND disabled --fit at the same time (FND-033).
    meta = {}
    weight_gb = 0.0
    if chat:
        gguf = Path(chat) / reldir.replace("/", os.sep) / main
        try:
            meta = read_gguf_meta(str(gguf))
            if gguf.exists():
                weight_gb = gguf.stat().st_size / 1e9
        except Exception:
            meta = {}
    # DEC-003: a brand new model gets a conservative context unless its profile
    # recommends something else.
    est_ctx = int((profile or {}).get("ctx_recommend") or MOE_DEFAULT_CTX)
    moe_hint = (profile or {}).get("moe") or {}
    # FND-068: --batch-size used to be a per-branch constant (1024 for a generic
    # qwen body, 1024 inline in the gemma4 body, 256 for qwen3.8) and
    # derive_overrides() never wrote one. The SAME model therefore ran with a
    # different batch size depending on whether it was launched from a .bat or
    # through Router mode - and batch size moves the compute buffer, i.e. VRAM.
    # preset-overrides.json is the source of truth, so read it.
    ov_batch = override_param(overrides, reldir, "batch-size", None, int)
    # ubatch is capped by batch - llama.cpp CLAMPS n_ubatch to n_batch, it does
    # not warn - so the two have to be read and written together. Raising ubatch
    # on its own is a silent no-op, which the first measurement run proved the
    # hard way: ub 512 and ub 2048 produced identical VRAM to the MiB while
    # batch stayed pinned at 512.
    ov_ub = override_param(overrides, reldir, "ubatch-size", None, int)
    ub_seg = " --ubatch-size {}".format(ov_ub) if ov_ub else ""
    moe_est = None
    if weight_gb >= MOE_OFFLOAD_MIN_GB:
        _ov_ent = model_entry_for(overrides, reldir) or {}
        moe_est = estimate_ncpu_moe(
            meta, weight_gb, est_ctx, "q8_0",
            start_ratio=moe_hint.get("start_ratio", MOE_START_OFFLOAD_RATIO),
            tensors=read_gguf_tensors(str(gguf)) if weight_gb else None,
            batch=ov_batch,
            vram_calib=(_ov_ent.get("tuning") or {}).get("vram"))
    if moe_est:
        # FND-064a: preset-overrides.json is the source of truth for tunables;
        # the profile's start_ratio is only the fallback for an unmeasured model.
        ov = override_ncpu_moe(overrides, reldir)
        if ov is not None and ov != moe_est["n_init"]:
            log("[moe] {}: n-cpu-moe {} from preset-overrides.json "
                "overrides the start_ratio estimate {}"
                .format(stem[:44], ov, moe_est["n_init"]))
            moe_est["n_init"] = ov
        log("[moe] {}: {} layers / {} experts -> --n-cpu-moe {} "
            "(range {}..{} | {} free, {} to spare)".format(
                stem[:44], moe_est["layers"], moe_est["experts"],
                moe_est["n_init"], moe_est["n_fast"], moe_est["n_safe"],
                moe_est["spare_opt_mib"], moe_est["spare_safe_mib"]))
        log("       source={}  W_non={} GB  E/layer={} GB  KV={} GB  "
            "compute={} GB".format(
                moe_est["source"], moe_est["w_non_gb"], moe_est["e_layer_gb"],
                moe_est["kv_gb"], moe_est["compute_gb"]))

    sanit = re.sub(r"[^A-Za-z0-9]", "_", stem).upper()[:20]
    base = "AUTO_" + sanit
    i = 2
    while base in taken_names:
        base = "AUTO_{}_{}".format(sanit, i)
        i += 1
    vmain, vmm, vdraft = base, base + "_MM", base + "_D"
    taken_names |= {vmain, vmm, vdraft}

    rel_os = reldir.replace("/", os.sep)
    groups = [{
        "t": "group",
        "comments": ["REM --- [NEW] {} ---".format(stem)],
        "vars": ['set "{}={}\\{}"'.format(vmain, "%CHAT%", rel_os + os.sep + main)],
    }]
    entries = []
    goto = "RUN_" + sanit + str(i) if False else "RUN_" + sanit
    # ensure unique goto label (we only track taken_names for vars; keep simple)
    if "mtp" in low:
        goto += "_MTP"

    def common_checks(draft=False):
        c = ['call :check_file "%{}%"'.format(vmain)]
        if mm:
            c.append('call :check_file "%{}%"'.format(vmm))
        if draft:
            c.append('call :check_file "%{}%"'.format(vdraft))
        return c

    if launcher == "gemma4":
        if "26b" in low:
            ctxv, ctxk, ngl, gld = "%CTX_MAX_26B%", "64K", "%NGL_26B%", "24"
        elif "e4b" in low:
            ctxv, ctxk, ngl, gld = "%CTX_E4B%", "260K", "99", "99"
        else:
            ctxv, ctxk, ngl, gld = "%CTX_MAX_12B%", "260K", "99", "40"
        prefix = next((p for p in MTP_PREFIXES if p.lower() in main.lower()), None)
        drafts = [d for d in mtp_files if prefix and prefix.lower() in d.lower()]
        draft = drafts[0] if drafts else None
        if draft:
            groups[0]["vars"].append(
                'set "{}={}\\{}"'.format(vdraft, "%MTP%", draft))
        mmseg = ' --mmproj "%{}%"'.format(vmm) if mm else ""
        temp = prof_samp.get("temp", 1.0)
        top_p = prof_samp.get("top_p", 0.95)
        top_k = prof_samp.get("top_k", 64)
        # MoE must never use -ngl: an explicit -ngl makes --fit give up
        # entirely (FND-033). Pin the first N layers' expert weights to CPU
        # instead - and do NOT also write --fit/--fit-ctx, which cannot work
        # alongside --n-cpu-moe (FND-066, A/B verified).
        if moe_est:
            ngl_seg = "--n-cpu-moe {} ".format(moe_est["n_init"])
            moe_tag = " MoE ncmoe={}".format(moe_est["n_init"])
        else:
            ngl_seg = "-ngl {} ".format(ngl)
            moe_tag = ""
        label = "{} {} {}{} [NEW]".format(stem, quant, ctxk, moe_tag)
        body = "\r\n".join([
            ":{}".format(goto),
            *common_checks(False),
            "echo ============================================",
            "echo {}".format(label),
            "echo ============================================",
            "echo [MANUAL] ctx={}".format(ctxv),
            'llama-server.exe -m "%{}%"{} -c {} {}-fa on -np 2 -t 10 '
            "--batch-size {}{} --cache-type-k q8_0 --cache-type-v q8_0 "
            "--keep -1 --load-mode none --host 0.0.0.0 --port %PORT% "
            "--api-key %API_KEY% --temp {} --top-p {} --top-k {} "
            "--timeout 120 %TOOLS_ARG% %REASONING_ARG% %REPEAT_ARG% --jinja"
            .format(vmain, mmseg, ctxv, ngl_seg, ov_batch or 1024, ub_seg,
                    temp, top_p, top_k),
            "pause & goto menu",
        ])
        entries.append({"label": label, "goto": goto,
                        "dirs": [reldir], "drafts": [], "banner": [], "body": body,
                        "profile": (profile or {}).get("id")})
        if draft:
            body_mtp = "\r\n".join([
                ":{}_MTP".format(goto),
                *common_checks(True),
                "echo ============================================",
                "echo {} + MTP {}{} [NEW]".format(stem, ctxk, moe_tag),
                "echo ============================================",
                "echo [MANUAL] ctx={}".format(ctxv),
                'llama-server.exe -m "%{}%"{} --model-draft "%{}%" '
                "--spec-type draft-mtp --spec-draft-n-max %MTP_N_MAX% "
                "-c {} {}-gpu-layers-draft {} -fa on -np 2 -t 10 "
                "--batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 "
                "--keep -1 --load-mode none --host 0.0.0.0 --port %PORT% "
                "--api-key %API_KEY% --temp {} --top-p {} --top-k {} "
                "--timeout 120 %TOOLS_ARG% %REASONING_ARG% %REPEAT_ARG% --jinja"
                .format(vmain, mmseg, vdraft, ctxv, ngl_seg, gld, temp, top_p,
                        top_k),
                "pause & goto menu",
            ])
            entries.append({"label": "{} + MTP {} [NEW]".format(stem, ctxk),
                            "goto": goto + "_MTP", "dirs": [reldir],
                            "drafts": [draft], "banner": [], "body": body_mtp,
                            "profile": (profile or {}).get("id")})
        return groups, entries

    if launcher == "qwen":
        mmseg = ' --mmproj "%{}%"'.format(vmm) if mm else ""
        imgseg = "  --image-min-tokens 1024 ^" if mm else ""
        is_mtp = "mtp" in low
        is_qwen38 = "qwen3.8" in low
        if "27b" in low:
            ctx, ctxk, t, batch = "65536", "64K", "12", "256"
            if is_qwen38:
                ngl = None                      # --fit auto-layers, no hard -ngl
                temp, top_p, top_k, timeout = (
                    prof_samp.get("temp", 1.0), prof_samp.get("top_p", 0.95),
                    prof_samp.get("top_k", 20), "300")
                extra = ("  --reasoning-preserve ^\n"
                         "  --reasoning-budget 8192 ^\n"
                         "  --reasoning-format deepseek ^\n"
                         "  --min-p 0.0 ^\n"
                         "  --presence-penalty 0.0 ^\n"
                         "  --repeat-penalty 1.0 ^\n"
                         "  --ubatch-size 512 ^\n"
                         '  --chat-template-kwargs "{\\"reasoning_effort\\":\\"medium\\"}" ^\n'
                         "  --metrics ^")
                mmap_seg = "  --load-mode mmap ^"
            else:
                ngl = "48"
                temp, top_p, top_k, timeout = (
                    prof_samp.get("temp", 0.6), prof_samp.get("top_p", 0.95),
                    prof_samp.get("top_k", 20), "300")
                extra = "  --reasoning-preserve ^"
                mmap_seg = "  --load-mode none ^"
            mtp_seg = ("  -fit off ^\n  --spec-type draft-mtp ^\n"
                       "  --spec-draft-n-max %MTP_N_MAX% ^\n"
                       "  --spec-draft-p-min 0.75 ^") if is_mtp else ""
        else:
            # DEC-003: conservative context for a brand new model; the profile
            # may recommend more and wins when present.
            ctx = str(est_ctx)
            ctxk = "64K" if est_ctx >= 65536 else "32K"
            ngl = "99"
            t, batch = "10", str(ov_batch or 1024)
            temp, top_p, top_k, timeout = (
                prof_samp.get("temp", 0.7), prof_samp.get("top_p", 0.95),
                prof_samp.get("top_k", 20), "300")
            extra = ""
            # X3 fix: a model whose name carries MTP has built-in heads - turn
            # them on (no --model-draft needed). This used to be skipped in the
            # generic branch, silently giving up ~+74% throughput.
            mtp_seg = ("  --spec-type draft-mtp ^\n"
                       "  --spec-draft-n-max %MTP_N_MAX% ^") if is_mtp else ""
            mmap_seg = "  --load-mode none ^"
        if moe_est:
            # MoE: push the first N layers' expert weights to CPU. Never emit
            # -ngl (FND-033), and never emit --fit/--fit-ctx either: --n-cpu-moe
            # compiles to -ot tensor overrides and common_fit_params() bails out
            # as soon as tensor_buft_overrides is set, so the fit never runs
            # (FND-066). A/B on 26B-A4B at 131072 ctx: identical VRAM (10,398
            # MiB) and speed (39.7 vs 40.3 t/s) with and without the flags.
            layer_seg = "  --n-cpu-moe {} ^".format(moe_est["n_init"])
        elif is_qwen38:
            # Dense - here --fit genuinely runs and is the right control.
            layer_seg = "  --fit on ^\n  --fit-ctx {} ^".format(ctx)
        else:
            layer_seg = "  -ngl {} ^".format(ngl)
        moe_tag = " MoE ncmoe={}".format(moe_est["n_init"]) if moe_est else ""
        label = "{} {} {}{} [NEW]".format(stem, quant, ctxk, moe_tag)
        parts = [
            ":{}".format(goto),
            *common_checks(False),
            "echo ============================================",
            "echo {}".format(label),
            "echo ============================================",
            "echo [MANUAL] ctx={}".format(ctx),
            "llama-server.exe ^",
            '  -m "%{}%" ^'.format(vmain),
            '  --mmproj "%{}%" ^'.format(vmm) if mm else "",
            "  --jinja ^",
            imgseg,
            "  -c {} ^".format(ctx),
            layer_seg,
            mtp_seg,
            "  -fa on ^",
            "  -np 1 ^",
            "  -t {} ^".format(t),
            "  --batch-size {} ^".format(batch),
            "  --ubatch-size {} ^".format(ov_ub) if ov_ub else "",
            "  --cache-type-k q8_0 ^",
            "  --cache-type-v q8_0 ^",
            "  --keep -1 ^",
            mmap_seg,
            "  --host 0.0.0.0 ^",
            "  --port %PORT% ^",
            "  --api-key %API_KEY% ^",
            "  --temp {} ^".format(temp),
            "  --top-p {} ^".format(top_p),
            "  --top-k {} ^".format(top_k),
            extra,
            "  %TOOLS_ARG% ^",
            "  --timeout {}".format(timeout),
            "pause & goto menu",
        ]
        # FIX 2026-09-12 (FND-018 -> FND-014): drop empty fragments. An empty
        # physical line inside a `^` continuation silently truncates the
        # command - cmd.exe discards every argument after it, so the entry
        # would launch `llama-server.exe -m <model>` with default port,
        # no api-key and default ctx.
        flat = []
        for s in parts:
            flat.extend(x for x in s.split("\n") if x.strip())
        body = "\r\n".join(flat)
        # guardrail: fail loud if required qwen3.8 args missing (silent-fail prevention)
        if is_qwen38:
            required = ["--reasoning-budget 8192", "--reasoning-format deepseek",
                        "--chat-template-kwargs", "--reasoning-preserve",
                        "--min-p 0.0"]
            missing = [r for r in required if r not in body]
            if missing:
                raise SystemExit(
                    "[X] make_auto(qwen38) missing required arg(s) in body: "
                    "{} - fix template before writing".format(missing))
        entries.append({"label": label, "goto": goto, "dirs": [reldir],
                        "drafts": [], "banner": [], "body": body,
                        "profile": (profile or {}).get("id")})
        return groups, entries

    # cpu
    mmseg = '  --mmproj "%{}%" ^'.format(vmm) if mm else ""
    imgseg = "  --image-min-tokens 1024 ^" if mm else ""
    temp = prof_samp.get("temp", 0.7)
    top_p = prof_samp.get("top_p", 0.95)
    top_k = prof_samp.get("top_k", 20)
    rep_line = ""
    if prof_samp.get("repeat_penalty"):
        rep_line = "  --repeat-penalty {} ^".format(prof_samp["repeat_penalty"])
    label = "{} {} 128K [NEW]".format(stem, quant)
    if mm:
        label += " (multimodal)"
    # FIX 2026-09-12 (FND-014): filter empty fragments - an empty line inside a
    # `^` continuation silently truncates the command (see qwen branch note).
    body = "\r\n".join(x for x in [
        ":{}".format(goto),
        *common_checks(False),
        "echo ============================================",
        "echo {}".format(label),
        "echo ============================================",
        "llama-server.exe ^",
        '  -m "%{}%" ^'.format(vmain),
        mmseg,
        "  --jinja ^",
        imgseg,
        "  -c %CTX_CPU% ^",
        "  -ngl 0 ^",
        "  -np 2 ^",
        "  -t %THREADS% ^",
        "  --threads-batch %THREADS% ^",
        "  --batch-size 256 ^",
        "  --cache-type-k q8_0 ^",
        "  --cache-type-v q8_0 ^",
        "  --keep -1 ^",
        "  --load-mode mmap ^",
        "  --host 0.0.0.0 ^",
        "  --port %PORT% ^",
        "  --api-key %API_KEY% ^",
        "  --temp {} ^".format(temp),
        "  --top-p {} ^".format(top_p),
        "  --top-k {} ^".format(top_k),
        rep_line,
        "  %TOOLS_ARG% %REASONING_ARG% ^",
        "  --timeout 300",
        "pause & goto menu",
    ] if x.strip())
    entries.append({"label": label, "goto": goto, "dirs": [reldir],
                    "drafts": [], "banner": [], "body": body,
                    "profile": (profile or {}).get("id")})
    return groups, entries


# ============================================== mmproj drift (dtype change)
def mmproj_drift_report(reg, dirs, chat):
    """Warn when a registry/static var or entry body references a mmproj file
    that is no longer on disk (e.g. F16 -> F32 dtype change)."""
    issues = []
    for lname, cfg in reg["launchers"].items():
        texts = []
        for item in cfg["region"]:
            texts.append(item.get("s", ""))
            texts.extend(item.get("comments") or [])
            texts.extend(item.get("vars") or [])
        for e in cfg["entries"]:
            texts.append(e.get("body", ""))
        for txt in texts:
            for m in re.finditer(
                    r'set "([A-Za-z0-9_]*MM[A-Za-z0-9_]*?)"?=([^"\r\n]*mmproj'
                    r'[^"\r\n]*\.gguf)"?', txt, re.I):
                var, path = m.group(1), m.group(2)
                norm = path.replace("%CHAT%", str(chat)).replace("/", os.sep)
                p = Path(norm)
                if p.exists() or not p.parent.exists():
                    continue
                disk = [x for x in p.parent.iterdir()
                        if x.is_file() and "mmproj" in x.name.lower()
                        and x.suffix.lower() == ".gguf"]
                if not disk:
                    continue
                newest = max(disk, key=lambda x: x.stat().st_mtime)
                if newest.name != p.name:
                    issues.append((lname, var, p.name, newest.name))
    for lname, var, old, new in issues:
        log("[!] {}: {} references '{}' but disk has '{}' (mmproj dtype/"
            "file changed)".format(lname, var, old, new))
    return issues


def fix_mmproj_drift(reg, dirs, chat):
    """Update static mmproj var/body references to the newest on-disk mmproj
    file. Returns number of references updated (registry mutated in place)."""
    fixed = 0
    for lname, cfg in reg["launchers"].items():
        for item in cfg["region"]:
            vars_ = list(item.get("vars") or [])
            if not vars_:
                continue
            new_vars = []
            changed = False
            for vln in vars_:
                m = re.match(
                    r'^(set "[A-Za-z0-9_]*MM[A-Za-z0-9_]*?"?=)([^"\r\n]*mmproj'
                    r'[^"\r\n]*\.gguf)("?\s*)$', vln, re.I)
                if not m:
                    new_vars.append(vln)
                    continue
                path = m.group(2)
                norm = path.replace("%CHAT%", str(chat)).replace("/", os.sep)
                p = Path(norm)
                if p.exists():
                    new_vars.append(vln)
                    continue
                disk = [x for x in p.parent.iterdir()
                        if x.is_file() and "mmproj" in x.name.lower()
                        and x.suffix.lower() == ".gguf"] \
                    if p.parent.exists() else []
                if not disk:
                    new_vars.append(vln)
                    continue
                newest = max(disk, key=lambda x: x.stat().st_mtime)
                newpath = str(newest).replace(str(chat), "%CHAT%")
                newpath = newpath.replace(os.sep, "\\")
                new_vars.append(m.group(1) + newpath + m.group(3))
                log("[fix-mmproj] {} {}: {} -> {}".format(
                    lname, m.group(1).split("=")[0][4:], p.name, newest.name))
                fixed += 1
                changed = True
            if changed:
                item["vars"] = new_vars
        for e in cfg["entries"]:
            body = e.get("body", "")
            if "mmproj" not in body.lower():
                continue
            new_body = body
            for m in re.finditer(r'--mmproj "([^"]*mmproj[^"]*\.gguf)"',
                                 body, re.I):
                path = m.group(1)
                norm = path.replace("%CHAT%", str(chat)).replace("/", os.sep)
                p = Path(norm)
                if p.exists() or not p.parent.exists():
                    continue
                disk = [x for x in p.parent.iterdir()
                        if x.is_file() and "mmproj" in x.name.lower()
                        and x.suffix.lower() == ".gguf"]
                if not disk:
                    continue
                newest = max(disk, key=lambda x: x.stat().st_mtime)
                newpath = str(newest).replace(str(chat), "%CHAT%")
                newpath = newpath.replace(os.sep, "\\")
                new_body = new_body.replace(
                    m.group(0), '--mmproj "{}"'.format(newpath))
                log("[fix-mmproj] {} body: {} -> {}".format(
                    lname, p.name, newest.name))
                fixed += 1
            if new_body != body:
                e["body"] = new_body
    return fixed


# ============================================== MTP draft health / blacklist
BLACKLIST_PATH = BASE / "draft-blacklist.json"
HEALTH_PATH = BASE / "draft-health.json"

# A gemma4 MTP drafter must advertise this arch with a single MTP block.
DRAFT_ARCH = "gemma4-assistant"
DRAFT_LAYERS = 4

# A drafter pairs by FAMILY, not by quantization: the official README says the
# 12B drafter "pairs with any quant of the 12B". Matching is done on the
# separator-stripped name so `Gemma4-26B-...` and `gemma-4-26B-...` both hit.
DRAFT_FAMILIES = (
    ("12b", ("gemma-4-12b",)),
    ("26b-a4b", ("gemma-4-26b-a4b",)),
    ("e4b", ("gemma-4-e4b",)),
)


def draft_family(name):
    """'12b' / '26b-a4b' / 'e4b' / None for a drafter or main-model filename."""
    n = _norm_name(Path(name).name)
    for fam, pats in DRAFT_FAMILIES:
        if any(_norm_name(p) in n for p in pats):
            return fam
    return None

_blacklist = {}


def set_blacklist(bl):
    global _blacklist
    _blacklist = bl or {}


def load_blacklist():
    """{draft filename: reason} - drafts that must never be offered."""
    if not BLACKLIST_PATH.exists():
        return {}
    try:
        data = json.loads(BLACKLIST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log("[!] {} unreadable ({}), ignoring".format(
            BLACKLIST_PATH.name, exc))
        return {}
    return {d["file"]: d.get("reason", "unknown")
            for d in data.get("drafts", []) if d.get("file")}


def save_blacklist(entries):
    with BLACKLIST_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump({"drafts": entries}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def draft_meta_check(path):
    """Cheap pre-flight: right architecture and block count?"""
    meta = read_gguf_meta(str(path))
    arch = meta.get("arch")
    layers = meta.get("layers")
    if arch != DRAFT_ARCH:
        return False, "arch={} (expected {})".format(arch, DRAFT_ARCH), meta
    if layers != DRAFT_LAYERS:
        return False, "block_count={} (expected {})".format(layers,
                                                            DRAFT_LAYERS), meta
    return True, "ok", meta


def audit_drafts(reg, dirs, mtp_files, chat, overrides):
    """Static report answering 'which draft files can I safely delete?'.

    No model is loaded. Builds the full drafter <-> main-model family matrix,
    flags orphans (a drafter whose family has no main model left on disk),
    counts how many launch entries reference each file, and calls out the
    historical `D26_Q4 -> Q8_0` naming trap.
    """
    log("=" * 72)
    log("== MTP drafter audit (static - nothing is loaded) ==")
    log("=" * 72)

    # --- which launcher variables reference each drafter
    refs = {}
    for name, cfg in reg["launchers"].items():
        text, _ = read_text(BASE / cfg["script"], cfg["encoding"])
        if not text:
            continue
        for m in re.finditer(r'set "(\w+)=%MTP%\\([^"]+)"', text):
            var, fname = m.group(1), m.group(2)
            uses = len(re.findall(r"%{}%".format(re.escape(var)), text))
            refs.setdefault(fname, []).append((cfg["script"], var, uses))

    # --- which main models can accept a drafter
    mains = {}                      # family -> [(rel, gguf, size_gb)]
    for rel, info in sorted(dirs.items(), key=lambda kv: kv[0].lower()):
        gguf = info["mains"][0]
        fam = draft_family(gguf)
        if not fam:
            continue
        p = Path(chat) / rel.replace("/", os.sep) / gguf
        mains.setdefault(fam, []).append(
            (rel, gguf, (p.stat().st_size / (1 << 30)) if p.exists() else 0.0))

    wid = min(max([len(f) for f in mtp_files] + [28]), 46)
    log("")
    log("  {:<{w}} {:>7} {:<9} {:<9} {:<10} {}".format(
        "草稿文件", "大小", "家族", "元数据", "被引用", "命中的主模型",
        w=wid))
    log("  " + "-" * 72)
    orphans, used, unused = [], [], []
    for f in sorted(mtp_files):
        p = Path(chat) / MTP_DIR_NAME / f
        size = p.stat().st_size / (1 << 20) if p.exists() else 0
        fam = draft_family(f)
        ok, why, _meta = draft_meta_check(p)
        r = refs.get(f, [])
        used_vars = sum(1 for _s, _v, u in r if u)
        hits = mains.get(fam, [])
        log("  {:<{w}} {:>6.0f}M {:<9} {:<9} {:<10} {}".format(
            f[:wid], size, fam or "?", "OK" if ok else "BAD",
            "{}处/{}var".format(sum(u for _s, _v, u in r), len(r)),
            ", ".join(h[0][:26] for h in hits) or "*** 无 ***", w=wid))
        if not hits:
            orphans.append(f)
        if used_vars:
            used.append(f)
        else:
            unused.append(f)
        if not ok:
            log("       [X] 元数据不合格: {}".format(why))

    log("")
    if orphans:
        log("  [X] {} 个孤儿草稿（家族在主模型里已不存在，可删除）:".format(
            len(orphans)))
        for f in orphans:
            log("      {}".format(f))
    else:
        log("  [OK] 无孤儿：每个草稿家族都有存活的主模型")
    if unused:
        log("  [!] {} 个草稿没有任何启动条目引用（含对照/备用档则属正常）:"
            .format(len(unused)))
        for f in unused:
            log("      {}".format(f))

    # --- the historical naming trap
    for f, r in sorted(refs.items()):
        fam = draft_family(f)
        for script, var, uses in r:
            vfam = draft_family(var)
            if vfam and fam and vfam != fam:
                log("  [!] 命名陷阱: {} 里 {} 指向 {}（家族不符）".format(
                    script, var, f))
    # --- one file behind several variable names: the classic "I can't tell
    #     which drafts I have" confusion (D26_Q4 / D26_Q8 / D26_UNCENS are all
    #     the same Q8_0 file, a leftover from retiring the 26B third-party Q4_0)
    for f, r in sorted(refs.items()):
        if len(r) < 2:
            continue
        log("  [!] 一名多指: {} 被 {} 个变量引用 -> {}".format(
            f, len(r), ", ".join("{}:{}".format(s, v) for s, v, _u in r)))
        log("      这些变量指向的是【同一个文件】，不是多个草稿。")

    # --- family coverage summary
    log("")
    log("  家族 → 主模型 覆盖:")
    for fam, _pats in DRAFT_FAMILIES:
        got = mains.get(fam, [])
        log("    {:<9} {} 个主模型 {}".format(
            fam, len(got), "，".join(h[0][:34] for h in got) or "(无)"))
    log("")
    log("  结论：可安全删除的 = 上面标 *** 无 *** 的草稿；其余都仍可用。")
    log("        「未被任何条目引用」≠ 可删——它们是对照/备用档。")
    log("")
    return {"orphans": orphans, "unused": unused, "refs": refs, "mains": mains}


def _free_port(start):
    import socket
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def plan_draft_pairs(dirs, mtp_files, overrides, configured_only=False):
    """Decide which (main-model, drafter) pairs the load test should cover.

    `configured_only=True` reproduces the historical behaviour: only the pairs
    named by `mtp.draft` in preset-overrides.json.

    Otherwise every family-consistent combination is tested, plus ONE control
    pair per drafter (a main model from a *different* family). The controls
    matter: if a mismatched pair also looks healthy, the family-pairing
    assumption is wrong and every "OK" becomes meaningless.

    Returns [(rel, draft, expect_reject)].
    """
    models = overrides.get("models") or {}
    if configured_only:
        out = []
        for rel, e in sorted(models.items()):
            draft = (e.get("mtp") or {}).get("draft")
            if draft and draft in mtp_files:
                out.append((rel, draft, False))
        return out

    by_family = {}
    for rel, info in sorted(dirs.items(), key=lambda kv: kv[0].lower()):
        fam = draft_family(info["mains"][0])
        if fam:
            by_family.setdefault(fam, []).append(rel)

    out = []
    for draft in sorted(mtp_files):
        fam = draft_family(draft)
        for rel in by_family.get(fam, []):
            out.append((rel, draft, False))
        # one control: the first main of any other family
        other = next((r for f, rs in sorted(by_family.items()) if f != fam
                      for r in rs), None)
        if other:
            out.append((other, draft, True))
    return out


def validate_drafts(dirs, mtp_files, chat, overrides, deep=True,
                    timeout_s=300, configured_only=False):
    """Verify every configured (main model, draft) pair the way the router
    actually loads it - through llama-server.

    llama-cli succeeding is NOT proof: the 26B third-party Q4_0 drafter used to
    load fine in llama-cli while llama-server crashed with `invalid vector
    subscript`. Returns (health, failed) where failed maps draft -> reason.
    """
    import subprocess
    import time
    import urllib.request

    exe = find_server_exe()
    if exe is None:
        raise SystemExit("[X] llama-server.exe not found")

    health = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llama_server": str(exe),
        "drafts": [],
        "pairs": [],
    }
    failed = {}

    # ---- 1. cheap metadata pass over every draft on disk
    for name in sorted(mtp_files):
        p = Path(chat) / MTP_DIR_NAME / name
        ok, why, meta = draft_meta_check(p)
        health["drafts"].append({
            "file": name,
            "size": p.stat().st_size if p.exists() else None,
            "arch": meta.get("arch"),
            "layers": meta.get("layers"),
            "meta_ok": ok,
            "meta_note": why,
        })
        if not ok:
            failed[name] = "metadata: " + why

    if not deep:
        return health, failed

    # ---- 2. real load test per pair (exhaustive by family + controls)
    pairs = plan_draft_pairs(dirs, mtp_files, overrides,
                             configured_only=configured_only)
    log("[plan] {} pair(s) to test{}".format(
        len(pairs), "" if configured_only else
        " (family-consistent combinations + 1 control per drafter)"))
    idx = 0
    for rel, draft, expect_reject in pairs:
        main_name = (dirs.get(rel) or {}).get("mains") or []
        if not main_name:
            log("[skip] {}: no main gguf on disk".format(rel))
            continue
        main_path = Path(chat) / rel.replace("/", os.sep) / main_name[0]
        draft_path = Path(chat) / MTP_DIR_NAME / draft
        if not draft_path.exists():
            if not expect_reject:
                failed[draft] = "missing on disk"
            health["pairs"].append({"model": rel, "draft": draft,
                                    "expected": "rejected" if expect_reject
                                                else "healthy",
                                    "ok": bool(expect_reject),
                                    "reason": "missing on disk"})
            continue
        if (not expect_reject
                and failed.get(draft, "").startswith("metadata:")):
            health["pairs"].append({"model": rel, "draft": draft,
                                    "expected": "healthy", "ok": False,
                                    "reason": failed[draft]})
            continue

        port = _free_port(18200 + idx * 3)
        idx += 1
        logf = Path(tempfile.gettempdir()) / "draftcheck-{}.log".format(port)
        # No -ngl: an explicit -ngl makes common_fit_params() bail out (FND-033)
        # and a 17GB model would then land entirely on a 16GB card.
        cmd = [str(exe), "-m", str(main_path), "--model-draft",
               str(draft_path), "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "2", "--fit", "on", "-fitc", "4096",
               "-c", "4096", "-np", "1", "--host", "127.0.0.1",
               "--port", str(port), "--no-warmup", "-t", "8"]
        log("[check] {}{} + {}".format(
            "" if not expect_reject else "[CONTROL] ", rel, draft))
        started = time.time()
        with logf.open("w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=str(Path(exe).parent))
        entry = {"model": rel, "draft": draft, "ok": False, "port": port,
                 "log": str(logf)}
        try:
            healthy = False
            while time.time() - started < timeout_s:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                            "http://127.0.0.1:{}/health".format(port),
                            timeout=3) as resp:
                        if resp.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(2)
            if not healthy:
                entry["reason"] = ("server did not become healthy "
                                   "(rc={})".format(proc.poll()))
            else:
                body = json.dumps({
                    "messages": [{"role": "user", "content":
                                  "Write one short sentence about the sea."}],
                    "max_tokens": 48, "stream": False}).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:{}/v1/chat/completions".format(port),
                    data=body, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=timeout_s) as r:
                        json.loads(r.read().decode("utf-8", "replace"))
                    entry["ok"] = True
                except Exception as exc:
                    entry["reason"] = "request failed: {}".format(exc)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        text = logf.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"draft acceptance\s*=\s*([\d.]+).*?"
                      r"mean len\s*=\s*([\d.]+)", text)
        if m:
            entry["acceptance"] = float(m.group(1))
            entry["mean_len"] = float(m.group(2))
            if entry["acceptance"] < 0.3:
                entry["ok"] = False
                entry["reason"] = "acceptance {:.3f} < 0.30 - draft is not " \
                                  "aligned with the model".format(
                                      entry["acceptance"])
        elif entry.get("ok"):
            entry["ok"] = False
            entry["reason"] = "no 'draft acceptance' in log - speculative " \
                              "decoding never ran"
        healthy = bool(entry.get("ok"))
        entry["expected"] = "rejected" if expect_reject else "healthy"
        entry["healthy"] = healthy
        if expect_reject:
            # Pass condition is inverted: this pair MUST NOT look healthy.
            entry["control"] = True
            if healthy:
                entry["ok"] = False
                entry["reason"] = (
                    "CONTROL LEAK: a mismatched family reported acceptance "
                    "{:.3f} - the family-pairing rule is not real, every "
                    "other OK is suspect".format(entry.get("acceptance") or 0.0))
            else:
                entry["ok"] = True
                entry["reason"] = "control: rejected as expected"
        else:
            entry["ok"] = healthy
            if not healthy:
                for pat in (r"invalid vector subscript", r"error", r"failed",
                            r"abort"):
                    mm = re.search(r".*{}.*".format(pat), text, re.I)
                    if mm:
                        entry.setdefault("crash", mm.group(0).strip()[:200])
                        break
                failed[draft] = entry.get("reason", "load test failed")
        entry["seconds"] = round(time.time() - started, 1)
        health["pairs"].append(entry)
        log("   -> {}{}  {}{}".format(
            "OK" if entry["ok"] else "FAIL",
            " [CONTROL]" if expect_reject else "",
            "acceptance={:.3f} mean_len={}".format(
                entry["acceptance"], entry["mean_len"])
            if "acceptance" in entry else "",
            "" if entry["ok"] else "  [{}]".format(entry.get("reason"))))

    return health, failed
# ============================================== blank-line body repair
BLANK_CONT_RE = re.compile(r"(\^[ \t]*\r?\n)(?:[ \t]*\r?\n)+")


# FND-066: `--fit`/`--fit-ctx` next to `--n-cpu-moe` is dead weight. The fit
# machinery aborts as soon as tensor_buft_overrides is non-empty, and setting
# `--n-cpu-moe N` is exactly that. Detection pattern for find_moe_gaps().
#
# Only the two forms that actually take a value, mirroring strip_decorative_fit.
# A bare `--fit` in prose is not a flag: two menu banners say
# `echo --fit 自动分层, KV=q8_0`, and the wider pattern this replaced matched
# that text. The result was an advisory that fired on every run, named a
# repair, and could never be satisfied - strip_decorative_fit removes only
# real arguments, so the body never changed and the warning never cleared.
DECORATIVE_FIT_RE = re.compile(
    r"(?<!\S)(--fit-ctx|-fitc)\s+\d+"
    r"|(?<!\S)(--fit|-fit)\s+(on|off)(?=\s|$)")


def find_moe_gaps(reg, dirs, chat, profiles, overrides):
    """Registry entries whose MoE `--n-cpu-moe` disagrees with the override.
    Three shapes are reported, worst first:

      'ngl'   - the body carries an explicit `-ngl N`, which makes
                common_fit_params() give up entirely (FND-033). A MoE larger than
                VRAM then loads fully on the device and OOMs. Guaranteed broken.
      'moe'   - the body uses `--fit on` but never sets `--n-cpu-moe`. It will
                load, because --fit can still drop whole layers, but that moves
                attention together with the experts and is far slower than just
                moving the expert tensors.
      'stale' - the body sets `--n-cpu-moe N` with a value that no longer
                matches preset-overrides.json (FND-064b). This is what happens
                after you measure a better value: the override is updated, the
                Router ini follows it, and the .bat launcher silently keeps the
                old number forever. The old code treated "the flag is present"
                as "the flag is correct", so it never noticed.
      'fit'   - the body has the right `--n-cpu-moe N` but still carries
                `--fit on` / `--fit-ctx N` next to it. Those flags never run
                (FND-066 - common_fit_params() gives up as soon as
                tensor_buft_overrides is set, which is exactly what --n-cpu-moe
                does), so they only advertise a safety net that does not exist.

    Returns a list of (script, label, entry, rel, ncpu, ctx, kind).
    """
    found = []
    for name, cfg in reg["launchers"].items():
        for e in cfg.get("entries", []):
            rel = (e.get("dirs") or [None])[0]
            if not rel or rel not in dirs:
                continue
            gguf = Path(chat) / rel.replace("/", os.sep) / dirs[rel]["mains"][0]
            if not read_gguf_meta(str(gguf)).get("experts"):
                continue                     # dense model - -ngl is correct
            ent = model_entry_for(overrides, rel)
            ncpu = (ent.get("params") or {}).get("n-cpu-moe")
            if ncpu is None:
                continue                     # no override yet - the gate handles it
            body = e.get("body") or ""
            if "llama-server" not in body:
                continue        # goto-alias entry: the real block is elsewhere and
                                # carries the launch arguments (e.g. RUN_26B_AGENT
                                # sets env vars then `goto RUN_26B_MTP`)
            ctx = ((ent.get("params") or {}).get("ctx-size")
                   or (overrides.get("global") or {}).get("ctx-size") or 65536)
            m = re.search(r"(?<!\S)--n-cpu-moe\s+(\S+)", body)
            if m:
                try:
                    have = int(m.group(1))
                except ValueError:
                    continue
                if have != int(ncpu):
                    found.append((cfg["script"], e.get("label", "?"), e, rel, ncpu,
                                  ctx, "stale"))
                    continue
                if DECORATIVE_FIT_RE.search(body):
                    # right value, but the dead --fit/--fit-ctx is still there
                    found.append((cfg["script"], e.get("label", "?"), e, rel, ncpu,
                                  ctx, "fit"))
                    continue
                # FND-068: --batch-size sizes the compute buffer, so for an
                # offloaded model it is a VRAM knob, not a preference. Leaving
                # it unset is fine (llama.cpp picks a default); CONTRADICTING
                # the override is not - that is how one model ended up running
                # with 1024 from a .bat and the larger default through Router.
                #
                # ubatch is checked alongside it because llama.cpp CLAMPS
                # n_ubatch to n_batch: a body whose ubatch exceeds its batch is
                # quietly running at the lower value, so the two must agree or
                # the pair is meaningless.
                bad = False
                for key, flag in (("batch-size", "--batch-size"),
                                  ("ubatch-size", "--ubatch-size")):
                    want_v = override_param(overrides, rel, key, None, int)
                    m2 = re.search(r"(?<!\S)" + re.escape(flag) + r"\s+(\S+)",
                                   body)
                    if want_v is None or not m2:
                        continue
                    try:
                        have_v = int(m2.group(1))
                    except ValueError:
                        have_v = None
                    if have_v is not None and have_v != want_v:
                        bad = True
                if bad:
                    found.append((cfg["script"], e.get("label", "?"), e, rel,
                                  ncpu, ctx, "batch"))
                continue                 # present AND correct
            kind = "ngl" if re.search(r"(?<!\S)-ngl\s+\S+", body) else "moe"
            found.append((cfg["script"], e.get("label", "?"), e, rel, ncpu, ctx,
                          kind))
    return found


def fix_moe_entries(reg, dirs, chat, profiles, overrides):
    """Repair the entries found by find_moe_gaps(). Returns a count."""
    n = 0
    for script, label, e, rel, ncpu, ctx, kind in find_moe_gaps(
            reg, dirs, chat, profiles, overrides):
        body = e["body"]
        new = body
        if kind == "stale":
            # FND-064b: the flag is there but holds an outdated number. Rewrite
            # the number (and the `MoE ncmoe=` marker in the banner) so a freshly
            # measured value actually reaches the launcher.
            old_m = re.search(r"(?<!\S)--n-cpu-moe\s+(\S+)", new)
            new = re.sub(r"(?<!\S)--n-cpu-moe\s+\S+",
                         "--n-cpu-moe {}".format(ncpu), new, count=1)
            if old_m:
                new = new.replace("ncmoe={}".format(old_m.group(1)),
                                  "ncmoe={}".format(ncpu))
        elif kind == "ngl":
            # Replace the -ngl that was disabling --fit with the control that
            # actually does the work. Do NOT also emit --fit/--fit-ctx: it
            # cannot run next to --n-cpu-moe (FND-066).
            new = re.sub(r"(?<!\S)-ngl\s+\S+",
                         "--n-cpu-moe {}".format(ncpu),
                         new, count=1)
            new = re.sub(r"(?<!\S)-c\s+\S+", "-c {}".format(ctx), new, count=1)
            new = re.sub(r"(?m)^(echo \[MANUAL\] ctx=).*$",
                         r"\g<1>{}".format(ctx), new, count=1)
            old_label = e.get("label") or ""
            ctx_k = ("{}K".format(int(int(ctx) / 1024))
                     if str(ctx).isdigit() else str(ctx))
            new_label = re.sub(r"\b\d+K\b", ctx_k, old_label, count=1)
            if new_label != old_label and old_label:
                new = new.replace("echo {}".format(old_label),
                                  "echo {}".format(new_label), 1)
                e["label"] = new_label
        elif kind == "batch":
            # FND-068: the override is the source of truth for tunables, so a
            # body that contradicts it must be rewritten, not left alone. Both
            # values move together - see the clamp note in find_moe_gaps().
            for key, flag in (("batch-size", "--batch-size"),
                              ("ubatch-size", "--ubatch-size")):
                want_v = override_param(overrides, rel, key, None, int)
                if want_v is None:
                    continue
                if re.search(r"(?<!\S)" + re.escape(flag) + r"\s+\S+", new):
                    new = re.sub(r"(?<!\S)" + re.escape(flag) + r"\s+\S+",
                                 "{} {}".format(flag, want_v), new, count=1)
                else:
                    # Insert it directly after --batch-size so the pair stays
                    # adjacent and readable in the generated .bat.
                    new = re.sub(r"((?<!\S)--batch-size\s+\S+)",
                                 r"\1 {} {}".format(flag, want_v), new, count=1)
        elif kind == "moe":
            # the flag is missing entirely: insert it where the loader will see
            # it. Preserve the body's own shape (inline vs `^` continuation) and
            # only add a token - whatever --fit/--fit-ctx was sitting there is
            # removed by the blanket strip below (FND-066).
            pat = r"(?<!\S)(--fit-ctx\s+\S+)"
            if re.search(pat, new):
                new = re.sub(pat, r"\1 --n-cpu-moe {}".format(ncpu), new, count=1)
            else:
                new = re.sub(r"(?<!\S)(--fit on)", r"\1 --n-cpu-moe {}".format(ncpu),
                             new, count=1)
        # kind == "fit": nothing to insert - the blanket strip below is the fix
        # FND-066: any body that ends up carrying --n-cpu-moe must not keep a
        # decorative --fit/--fit-ctx beside it - the fit never runs there, so
        # the flags only promise a safety net that does not exist.
        if re.search(r"(?<!\S)--n-cpu-moe\s", new):
            new = strip_decorative_fit(new)
        if re.search(r"(?<!\S)--n-cpu-moe\s", new) and new != body:
            e["body"] = new
            n += 1
            log("[fix-moe] {}: '{}' ({}) -> --n-cpu-moe {}".format(
                script, label, kind, ncpu))
    return n


def fix_blank_continuations(reg):
    """Repair stored entry bodies containing an empty line inside a `^`
    continuation (FND-014). Returns the number of bodies repaired.

    An empty physical line inside a caret continuation makes cmd.exe discard
    every argument after it, so such a body silently launches llama-server
    with default port/ctx and no api-key. make_auto() no longer produces them
    (FND-018 fix); this repairs data written before that fix.

    Also normalizes a bare LF to CRLF. make_auto() has joined its body
    fragments with "\\r\\n" since the FND-018 fix, but bodies written before it
    kept the bare `\\n` that the old builder left behind. The launcher files are
    a CRLF format and a mixed ending inside a `^` continuation is the same
    family of hazard as FND-014, so it is repaired here rather than left to be
    discovered by a shell that happens to be stricter than cmd.exe.
    """
    fixed = 0
    for lname, cfg in reg["launchers"].items():
        for e in cfg["entries"]:
            body = e.get("body", "")
            if not body:
                continue
            new = body
            while True:
                nxt = BLANK_CONT_RE.sub(lambda m: m.group(1), new)
                if nxt == new:
                    break
                new = nxt
            if new != body:
                dropped = body.count("\r\n") - new.count("\r\n")
                log("[fix-bodies] {}: '{}' - removed {} blank line(s) inside "
                    "a caret continuation".format(
                        cfg["script"], e.get("label", "?"), dropped))
            bare = len(re.findall(r"(?<!\r)\n", new))
            if bare:
                new = re.sub(r"(?<!\r)\n", "\r\n", new)
                log("[fix-bodies] {}: '{}' - normalized {} bare LF line "
                    "ending(s) to CRLF".format(
                        cfg["script"], e.get("label", "?"), bare))
            # TASK-080 (X5): `--no-mmap` / `--mmap` are DEPRECATED in this build
            # in favour of `--load-mode`. `--load-mode none` is the documented
            # equivalent of `--no-mmap` ("no special loading mode"), so the
            # swap is behaviour-preserving and drops a deprecation warning.
            if re.search(r"(?<!\S)(--no-mmap|--mmap)\b", new):
                hit = re.findall(r"(?<!\S)(--no-mmap|--mmap)\b", new)
                new = re.sub(r"(?<!\S)--no-mmap\b", "--load-mode none", new)
                new = re.sub(r"(?<!\S)--mmap\b", "--load-mode mmap", new)
                log("[fix-bodies] {}: '{}' - {} deprecated flag(s) -> "
                    "--load-mode".format(cfg["script"], e.get("label", "?"),
                                         len(hit)))
            if new != body:
                e["body"] = new
                fixed += 1
    return fixed


# ============================================================ ini generator
# Keys the generator owns. Everything else found in an existing section is
# carried over verbatim, so hand-tuned keys are never lost (FND-015).
INI_MANAGED_KEYS = {"model", "mmproj", "alias", "spec-type",
                    "spec-draft-model", "spec-draft-n-max"}


def read_ini_sections(path):
    """Parse an existing preset INI into {section: [raw 'key = value' lines]}."""
    if not Path(path).exists():
        return {}
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("ascii", errors="replace")
    sections, cur = {}, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1]
            sections[cur] = []
        elif cur is not None and s and not s.startswith(("#", ";")):
            sections[cur].append(ln.rstrip())
    return sections


def ini_preserved_keys(existing, alias, exclude=None):
    """Keys of `alias` that the generator does not own (hand-tuned values).

    `exclude` suppresses keys the generator is already emitting for this
    section, so a value that moved from the file into preset-overrides.json
    does not get written twice.
    """
    exclude = exclude or set()
    kept, seen = [], set()
    for ln in (existing or {}).get(alias, []):
        key = ln.split("=", 1)[0].strip().lower()
        if not key or key in INI_MANAGED_KEYS or key in exclude or key in seen:
            continue
        kept.append(ln)
        seen.add(key)
    return kept


def model_entry_for(ovr, rel):
    """preset-overrides.json entry for a model dir ({} when not configured)."""
    return ((ovr or {}).get("models") or {}).get(rel) or {}


def default_section_id(main):
    """Legacy section id: gguf filename with dots replaced by underscores."""
    return main[:-5].replace(".", "_")


def pick_mtp_draft(main, mtp_files, mtp=None):
    """The single draft to expose for this model.

    Preference: explicit `mtp.draft` > official Q8_0 > shortest name. Only one
    MTP variant is exposed per model so the router list stays readable
    (FND-025: four 12B aliases used to expose three distinct files).
    Blacklisted drafts (draft-blacklist.json) are never selected.
    """
    if _blacklist:
        mtp_files = [d for d in mtp_files if d not in _blacklist]
    want = (mtp or {}).get("draft")
    if want:
        if want in _blacklist:
            return None
        return want if want in mtp_files else None
    prefix = next((p for p in MTP_PREFIXES if p.lower() in main.lower()), None)
    if not prefix:
        return None
    cands = [d for d in mtp_files if prefix.lower() in d.lower()]
    if not cands:
        return None
    cands.sort(key=lambda d: (0 if "q8_0" in d.lower() else 1, len(d)))
    return cands[0]


def gen_ini_v2(dirs, mtp_files, chat, overrides=None, existing=None):
    """Render models-config.ini from disk + preset-overrides.json.

    Section order follows the model `id` so the router list is grouped.
    `global` supplies the [*] block; per-model `params` win over nothing - the
    router merges them over [*]. Hand-tuned keys already in the file that the
    generator does not own are carried over verbatim (FND-015 guard).

    Returns (text, preserved) where preserved is the list of
    (section, "key = value") lines that were actually carried over from the
    existing file - i.e. NOT supplied by preset-overrides.json.
    """
    ovr = overrides or {}
    preserved = []
    out = [
        "# llama.cpp Router preset - GENERATED FILE, do not edit by hand.",
        "# Source of truth: preset-overrides.json  (regenerate: llama-hub.bat)",
        "# Section name = model id shown in the Web UI.",
        "[*]",
    ]
    for k, v in sorted_params(ovr.get("global") or {}):
        out.append("{} = {}".format(k, v))
    out.append("")

    rows = []
    for rel in dirs:
        for main in dirs[rel]["mains"]:
            e = model_entry_for(ovr, rel)
            rows.append((e.get("id") or default_section_id(main), rel, main))
    rows.sort(key=lambda r: r[0].lower())

    for sid, rel, main in rows:
        e = model_entry_for(ovr, rel)
        note = e.get("note")
        if note:
            out.append("# {}".format(note))
        out.append("[{}]".format(sid))
        out.append("model = {}".format(str(Path(chat) / rel / main)))
        mml = dirs[rel].get("mmproj") or []
        if mml:
            out.append("mmproj = {}".format(str(Path(chat) / rel / mml[0])))
        # NOTE: llama-server keeps only ONE `alias` line per section - several
        # `alias = x` lines overwrite each other. Always emit a single
        # comma-separated line (FND-028).
        aliases = [a for a in (e.get("alias") or []) if a and a != sid]
        if aliases:
            out.append("alias = {}".format(", ".join(aliases)))
        params = e.get("params") or {}
        for k, v in sorted_params(params):
            out.append("{} = {}".format(k, v))
        carried = ini_preserved_keys(existing, sid,
                                     exclude=set(params) | {"alias"})
        preserved.extend((sid, ln) for ln in carried)
        out.extend(carried)
        out.append("")

        mtp = e.get("mtp") or {}
        draft = pick_mtp_draft(main, mtp_files, mtp)
        if draft:
            mid = (mtp.get("id")
                   or (sid + (mtp.get("id_suffix") or "-MTP")))
            mparams = dict(params)
            mparams.update(mtp.get("params") or {})
            if mtp.get("note"):
                out.append("# {}".format(mtp["note"]))
            out.append("# {} + MTP draft ({})".format(
                sid, os.path.basename(draft)))
            out.append("[{}]".format(mid))
            out.append("model = {}".format(str(Path(chat) / rel / main)))
            if mml:
                out.append("mmproj = {}".format(
                    str(Path(chat) / rel / mml[0])))
            out.append("spec-type = draft-mtp")
            out.append("spec-draft-model = {}".format(
                str(Path(chat) / MTP_DIR_NAME / draft)))
            out.append("spec-draft-n-max = {}".format(
                mparams.pop("spec-draft-n-max", 2)))
            for k, v in sorted_params(mparams):
                out.append("{} = {}".format(k, v))
            malign = [a for a in (mtp.get("alias") or []) if a and a != mid]
            if malign:
                out.append("alias = {}".format(", ".join(malign)))
            carried = ini_preserved_keys(existing, mid, exclude=set(mparams))
            preserved.extend((mid, ln) for ln in carried)
            out.extend(carried)
            out.append("")

    return "\r\n".join(out) + "\r\n", preserved


def diff_ini_sections(old_text, new_text):
    """Section-level and key-level diff between two rendered presets."""
    def parse(t):
        secs, cur = {}, None
        for ln in (t or "").splitlines():
            s = ln.strip()
            if s.startswith("[") and s.endswith("]"):
                cur = s[1:-1]
                secs[cur] = {}
            elif cur and s and not s.startswith(("#", ";")):
                k, _, v = s.partition("=")
                secs[cur][k.strip()] = v.strip()
        return secs
    o, n = parse(old_text), parse(new_text)
    added = [s for s in n if s not in o]
    dropped = [s for s in o if s not in n]
    lines = []
    for s in [x for x in n if x in o]:
        for k in n[s]:
            if k not in o[s]:
                lines.append(("+", s, k, n[s][k]))
            elif o[s][k] != n[s][k]:
                lines.append(("~", s, k, "{} -> {}".format(o[s][k], n[s][k])))
        for k in o[s]:
            if k not in n[s]:
                lines.append(("-", s, k, o[s][k]))
    for s in n:
        if s not in o:
            for k, v in n[s].items():
                lines.append(("+", s, k, v))
    return added, dropped, lines


def validate_preset_keys(overrides, allowed):
    """Assert every generated preset key is understood by this build (FND-029).

    Unknown keys make the router refuse to start entirely, so fail loudly
    before writing anything.
    """
    if not allowed:
        log("[!] key whitelist unavailable - skipping preset key validation")
        return []
    bad = []
    glob = (overrides or {}).get("global") or {}
    for k in sorted(glob):
        if k not in allowed:
            bad.append(("[*]", k))
    for rel, e in sorted(((overrides or {}).get("models") or {}).items()):
        for k in sorted(e.get("params") or {}):
            if k not in allowed:
                bad.append((rel, k))
        for k in sorted((e.get("mtp") or {}).get("params") or {}):
            if k not in allowed:
                bad.append((rel + " (mtp)", k))
    return bad


def tune_sweep(overrides, chat, target=None, reps=3, ctx=None):
    """Measure t/s for each `tuning.sweep` value of `n-cpu-moe` (REQ-013).

    Appends results to the measurement log - existing points are never
    overwritten, so repeated sweeps build history.

    `ctx` (TASK-090) is what makes the sweep mean something. llama-bench has
    **no** --ctx-size flag: it sizes the context as n_prompt + n_gen, so `-p N`
    is the only knob that forces a real allocation. Without it a sweep runs at
    p=512, never comes close to filling the card, and therefore reports "lower
    is faster" all the way down - a ladder that, followed literally, walks
    straight past the VRAM wall into OOM (FND-067). So:

      ctx set    -> `-p ctx`, results tagged with ctx, stored in
                    tuning.measured_realctx, eligible to decide the value
      ctx unset  -> p=512, results stored in tuning.measured_benchctx and
                    explicitly labelled ranking-only

    Be aware of the cost: -p 65536 makes every repetition take minutes on a
    35B-A3B, because the prompt itself has to be processed first.
    """
    import subprocess
    exe = BASE / "llama-bench.exe"
    if not exe.exists() and (BASE.parent / "llama-bench.exe").exists():
        exe = BASE.parent / "llama-bench.exe"
    if not exe.exists():
        log("[X] llama-bench.exe not found - cannot sweep")
        return 0
    ran = 0
    for rel, e in sorted((overrides.get("models") or {}).items()):
        if target and target not in ("all", rel):
            continue
        sweep = (e.get("tuning") or {}).get("sweep") or []
        if not sweep:
            continue
        files = None
        for d in sorted(Path(chat).iterdir()) if Path(chat).exists() else []:
            if d.is_dir() and d.name == rel:
                files = sorted(f for f in d.glob("*.gguf")
                               if "mmproj" not in f.name.lower())
                break
        if not files:
            log("[!] {}: no main gguf on disk - skipped".format(rel))
            continue
        log("[sweep] {}  model={}".format(rel, files[0].name))
        if ctx:
            log("        @ real ctx {} (passed as -p {}) - results go to "
                "measured_realctx and may decide the value".format(ctx, ctx))
        else:
            log("        @ bench ctx p=512 - CANNOT reach the VRAM wall, so "
                "this ladder can only rank; results go to measured_benchctx")
        bucket = "measured_realctx" if ctx else "measured_benchctx"
        measured = (e.setdefault("tuning", {})).setdefault(bucket, [])
        for v in sweep:
            cmd = [str(exe), "-m", str(files[0]), "-ncmoe", str(v),
                   "-p", str(ctx or 512), "-n", "128", "-r", str(reps),
                   "-o", "csv"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     errors="replace", timeout=1800)
            except Exception as exc:
                log("  [X] n-cpu-moe={}: {}".format(v, exc))
                continue
            tps = None
            # `llama-bench -o csv` emits a HEADER row followed by data rows.
            # There is NO `test` column: a row is a generation test iff
            # n_gen > 0 (pp512 has n_gen=0). Resolve column positions from the
            # header instead of guessing at offsets - the first column is
            # build_commit, not the test name.
            hdr = None
            for line in res.stdout.splitlines():
                if not line.strip():
                    continue
                parts = [p.strip().strip('"') for p in line.split(",")]
                if hdr is None:
                    if "avg_ts" in parts:
                        hdr = {name: i for i, name in enumerate(parts)}
                    continue
                try:
                    ts = float(parts[hdr["avg_ts"]])
                except (KeyError, IndexError, ValueError):
                    continue
                try:
                    is_gen = int(float(parts[hdr["n_gen"]])) > 0
                except (KeyError, IndexError, ValueError):
                    is_gen = False
                if is_gen:
                    tps = ts
            if tps is None:
                detail = "no generation row with an avg_ts column"
                if "error" in res.stdout.lower():
                    detail += " (llama-bench reported an error)"
                log("  [!] n-cpu-moe={}: could not parse tg128 - {} "
                    "(rc={})".format(v, detail, res.returncode))
                first = res.stdout.splitlines()[:2]
                for ln in first:
                    log("      | {}".format(ln[:150]))
                continue
            entry = {"n-cpu-moe": v, "tps": round(tps, 1), "ctx": ctx,
                     "date": datetime.now().strftime("%Y-%m-%d")}
            measured.append(entry)
            log("  n-cpu-moe={:<3} -> {:.1f} t/s  @ {}".format(
                v, tps, "{}K".format(ctx // 1024) if ctx else "bench(512)"))
            ran += 1
        # Sorted ladder for this model, so the optimum and the wall are visible
        # in one glance (lower n-cpu-moe = fewer experts on CPU = faster, until
        # the card runs out of VRAM and the value simply stops working).
        last = [m for m in measured if m.get("n-cpu-moe") is not None]
        tag = "{}K".format(ctx // 1024) if ctx else "bench"
        if last:
            best = max(last, key=lambda m: m.get("tps") or 0)
            log("  [ladder] {}  (@ {})".format(
                "  ".join("{}->{}".format(m["n-cpu-moe"], m.get("tps"))
                          for m in sorted(last, key=lambda m: m["n-cpu-moe"])),
                tag))
            log("  [best]   n-cpu-moe={} -> {} t/s  (@ {})".format(
                best["n-cpu-moe"], best.get("tps"), tag))
            if best.get("n-cpu-moe") == min(m["n-cpu-moe"] for m in last):
                log("  [!] the winner is the LOWEST value probed - the optimum "
                    "may be lower still; extend tuning.sweep downwards")
            if not ctx:
                log("  [!] this is a bench-ctx ladder - do NOT set a value from "
                    "it. Re-run with --ctx <real ctx> to reach the VRAM wall "
                    "(FND-067)")
    return ran


def describe_model(rel, files, prof, meta, mtp_draft):
    """Short human tag for a model (Chinese where it helps readability).

    This lands in the INI as a `#` comment line, so it is visible when the file
    is opened but never appears in the router's model list (FND-030).
    """
    tags = []
    if files.get("mmproj"):
        tags.append("图文多模态")
    if meta.get("experts"):
        tags.append("MoE 高速")
    if mtp_draft:
        tags.append("可挂 MTP 草稿")
    if prof and prof.get("tool_calling_sampling") is None:
        tags.append("不支持工具调用")
    if prof and prof.get("ctx_max"):
        tags.append("ctx<={}".format(prof["ctx_max"]))
    if meta.get("layers"):
        tags.append("{}L".format(meta["layers"]))
    if prof and prof.get("weights_gb"):
        tags.append("权重≈{}GB".format(prof["weights_gb"]))
    kv = (prof or {}).get("kv") or {}
    if kv.get("k"):
        tags.append("KV {}".format(kv["k"]))
    return " · ".join(tags) if tags else rel


MOE_OFFLOAD_MIN_GB = 12.0        # below this a MoE checkpoint fits VRAM on its own
MOE_START_OFFLOAD_RATIO = 0.75   # evidence-based start point (see estimate_ncpu_moe)
MOE_VRAM_GB = 15.9               # RTX 5060 Ti 16GB, usable after driver reservation
MOE_VRAM_MARGIN_GB = 1.5         # keep this much free (CUDA graph capture + desktop)
MOE_COMPUTE_BUF_GB = 2.0         # rough allowance for the compute graph
MOE_EXPERT_SHARE = 0.88          # expert-FFN share of a MoE checkpoint's bytes
MOE_DEFAULT_CTX = 65536
MOE_DEFAULT_BATCH = 512          # see FND-068: keeps the compute buffer small

# ---------------------------------------------------------------- VRAM model
# Guard rails decided by the user on 2026-09-13:
#   * stable decode below 35 t/s is not worth running locally, and
#   * at least 3 GB of VRAM must stay free.
# The 3 GB is not a round number picked for comfort. A single vision request was
# measured adding ~500 MiB of CUDA-graph cache that is never released, the
# vision path itself adds a similar amount, a browser or the desktop can claim
# the rest without warning - and WebUI/agent clients inject tools, so their
# compute buffers are larger than a plain API call's. Measured outcome: 1,065
# MiB free -> image request OOM; 4,782 MiB free -> fine.
TUNE_MIN_TPS = 35.0
TUNE_MIN_HEADROOM_MIB = 3072

# --- VRAM accounting basis (TASK-075) --------------------------------------
# The equation is written against the card as the driver reports it, because
# that is the number the user reads off nvidia-smi and the number stored in
# tuning.vram.vram_total_mib. Free space is what matters, and it has three
# separate thresholds instead of one arbitrary margin:
CARD_VRAM_MIB = 16311            # RTX 5060 Ti 16GB
SPARE_FITS_MIB = 1024            # llama.cpp's own --fit-target default floor
SPARE_MIN_MIB = TUNE_MIN_HEADROOM_MIB        # 3072 - the "3 GB free" rule
SPARE_SAFE_MIB = 2 * TUNE_MIN_HEADROOM_MIB   # 6144 - the extra-safe end

# The tensor table reports GiB (bytes / 2**30) while weight_gb and the VRAM
# budget are decimal. Mixing the two silently shifts every estimate by 7%, so
# the conversion is spelled out instead of being applied implicitly.
GIB_TO_GB = (1 << 30) / 1e9
GB_TO_MIB = 1e9 / (1 << 20)

# KV quantisation scales relative to q8_0 (bytes per element / 34/32).
KV_SCALE = {"q8_0": 1.0, "q4_0": 18.0 / 34.0,
            "f16": 2.0 / (34.0 / 32.0), "bf16": 2.0 / (34.0 / 32.0)}


def vram_estimate(entry, n_cpu_moe, ctx, kv_k="q8_0"):
    """Estimate MiB of VRAM for a configuration; None when not calibrated.

    Model:  fixed + KV(ctx, kv_type) + (layers - n) * per-layer expert bytes.
    The coefficients are measured per model and stored under `tuning.vram`,
    because they differ wildly between architectures - a 26B-A4B and a 35B-A3B
    share nothing here. Populate it with plan/_apply_q5_vram_calib.py.
    """
    c = (entry.get("tuning") or {}).get("vram") or {}
    if not c.get("kv_kib_per_token") or c.get("layers") is None:
        return None
    scale = KV_SCALE.get(kv_k, 1.0) / KV_SCALE.get(c.get("kv_type", "q8_0"), 1.0)
    kv = c["kv_kib_per_token"] * (ctx / 1024.0) * scale
    per_layer = c.get("expert_layer_mib") or 0
    return c.get("fixed_mib", 0) + kv + (c["layers"] - n_cpu_moe) * per_layer


def vram_verdict(entry, params, ctx=None, n=None, kv_k=None):
    """Return (estimate_mib, headroom_mib, ok) for a candidate configuration.

    `ok` means the configuration clears the 3 GB rule. None means the model has
    no calibration yet, which is NOT the same as "fine" - callers must say so
    rather than showing a green light.
    """
    p = dict(params)
    if ctx is not None:
        p["ctx-size"] = ctx
    if n is not None:
        p["n-cpu-moe"] = n
    if kv_k is not None:
        p["cache-type-k"] = kv_k
    c = (entry.get("tuning") or {}).get("vram") or {}
    total = c.get("vram_total_mib", 16311)
    est = vram_estimate(entry, p.get("n-cpu-moe") or 0,
                        int(p.get("ctx-size") or 65536),
                        p.get("cache-type-k") or "q8_0")
    if est is None:
        return None, None, None
    head = total - est
    return est, head, head >= TUNE_MIN_HEADROOM_MIB
# Only nag about a measured alternative when it is worth the churn - the same
# policy llama_hub.py --audit already applies. A +1 t/s "win" is inside
# run-to-run noise and is not worth re-verifying an entire model for.
MIN_GAIN_TPS = 3.0
MIN_GAIN_FRAC = 0.08          # DEC-003: conservative ctx for a brand new model


def extra_vram_gb(info, chat, rel, draft=None):
    """VRAM occupied by the mmproj projector and (optionally) an MTP draft."""
    total = 0.0
    for f in (info.get("mmproj") or [])[:1]:
        p = Path(chat) / rel.replace("/", os.sep) / f
        if p.exists():
            total += p.stat().st_size / 1e9
    if draft:
        p = Path(chat) / MTP_DIR_NAME / draft
        if p.exists():
            total += p.stat().st_size / 1e9
    return total


def compute_buf_gb(batch=None, vram_calib=None):
    """Allowance for the compute graph, in decimal GB (TASK-076).

    The compute buffer holds one activation set per micro-batch slot, so it grows
    with `--batch-size` - and batch-size is user-tunable from the parameter menu,
    which makes it a VRAM knob wearing the costume of a preference.

    The honest state of knowledge is that exactly ONE point is calibrated (Q5 at
    batch 512). A single point cannot yield a slope, so this does not invent one.
    It reads the slope from the calibration data and the caller is expected to
    warn when it is absent:

        tuning.vram.compute_mib            measured compute allowance
        tuning.vram.compute_batch          the batch size it was measured at
        tuning.vram.compute_mib_per_batch  measured slope (optional)

    With no slope, batch size is treated as free. That is a known gap, not a
    claim - inventing a coefficient here would make the estimate look precise
    while being fiction, and the OOM would arrive anyway.
    """
    calib = vram_calib or {}
    base = calib.get("compute_mib")
    if base is None:
        base = MOE_COMPUTE_BUF_GB * 1024.0
    slope = calib.get("compute_mib_per_batch")
    if batch and slope:
        base += (batch - (calib.get("compute_batch") or 512)) * slope
    return max(0.5, base / 1024.0)


def estimate_ncpu_moe(meta, weight_gb, ctx, kv_type="q8_0",
                      start_ratio=MOE_START_OFFLOAD_RATIO, extra_gb=0.0,
                      tensors=None, batch=None, vram_calib=None):
    """Pick `--n-cpu-moe` for a MoE checkpoint that does not fit in VRAM.

    Returns a dict, or None when the model is not MoE / too small / unknown:

        {n_init, n_low, n_high, sweep, layers, experts, e_layer_gb, kv_gb}

    Two bounds are computed (DEC-002: write the conservative one, sweep the rest):

      n_low  - byte-budget floor: the smallest offload that still fits, assuming
               MOE_EXPERT_SHARE of the file is expert FFN weights:
                   room = (VRAM - margin) - (W_non + KV + compute)
                   n    = layers - floor(room / bytes_per_layer)
      n_init - evidence-based start: start_ratio x layers. The ratio is a
               per-profile data point taken from measurement, not a guess:
                 gemma4-26b-a4b  -> 0.57  (user measured 17/30 -> 48-52 t/s)
                 qwen36-35b-a3b  -> 0.75  (user measured 30/41 -> 47 t/s,
                                           37/41 -> 48 t/s, 40/41 -> 40 t/s)
               Falls back to MOE_START_OFFLOAD_RATIO when the profile is silent.

    n_init is deliberately the safer (larger) of the two: over-offloading is
    merely slower, under-offloading refuses to load at all.

    NOTE 2026-09-12 - the byte budget was a PLACEHOLDER, and TASK-075 has now
    replaced it. W_non and the per-layer expert bytes come from the GGUF tensor
    table (read_gguf_tensors) when it is supplied; the share heuristic only
    remains as the fallback for a file that cannot be read.
    """
    layers = meta.get("layers") or 0
    if not layers or not weight_gb or not meta.get("experts"):
        return None
    kv = kv_estimate_gb(meta, ctx, kv_type) or 0.0

    # --- expert / non-expert split (TASK-074) --------------------------
    # Prefer the tensor table, which states both numbers exactly. The four local
    # MoE checkpoints measure 88.1 / 90.2 / 90.3 / 92.8 % expert, so the 0.88
    # constant is not wildly wrong - but it is a per-model fact approximated by a
    # constant, and the per-layer curve it implies is worse than that (see below).
    source = "share-estimate"
    n_layers = layers
    if tensors and tensors.get("size_ok") and tensors.get("per_layer"):
        n_layers = min(layers, tensors.get("layers") or layers)
        w_non = tensors["non_expert_gb"] * GIB_TO_GB
        e_total = tensors["expert_gb"] * GIB_TO_GB
        # cum[k] = MiB moved to CPU when the first k layers' experts go there.
        # The real curve, not k x mean: measured per-layer expert bytes spread
        # 0% / 17% / 17% / 31% between the smallest and largest layer, so a flat
        # multiplication can be hundreds of MiB out near the ends of the range.
        pl = tensors["per_layer"]
        cum_mib = [0.0]
        for i in range(n_layers):
            cum_mib.append(cum_mib[-1] + pl.get(i, 0.0) * 1024.0)
        source = "tensor-table"
    else:
        e_total = weight_gb * MOE_EXPERT_SHARE
        w_non = weight_gb - e_total
        per = (e_total / n_layers) * GIB_TO_GB * 1024.0 if n_layers else 0.0
        cum_mib = [i * per for i in range(n_layers + 1)]
    e_layer_gb = (e_total / n_layers) if n_layers else 0.0

    compute_gb = compute_buf_gb(batch, vram_calib)
    fixed_gb = w_non + kv + extra_gb + compute_gb

    def spare_mib(n):
        """Free VRAM left on the card when n layers' experts sit on the CPU."""
        n = max(0, min(n_layers, int(n)))
        remaining = e_total - (cum_mib[n] / 1024.0) * GIB_TO_GB
        return CARD_VRAM_MIB - (fixed_gb + remaining) * GB_TO_MIB

    def smallest_n(want_mib):
        """Smallest offload that still leaves want_mib free on the card."""
        for n in range(0, n_layers + 1):
            if spare_mib(n) >= want_mib:
                return n
        return n_layers

    # Three thresholds, from aggressive to safe. More experts on the GPU means
    # faster, so each is the SMALLEST n meeting its requirement:
    #   n_fast - loads at all, with llama.cpp's own 1 GiB floor
    #   n_opt  - the "3 GB free" rule; the fastest configuration that is safe to
    #            actually use, and the value that gets written
    #   n_safe - double margin, for vision-heavy or agent use where WebUI injects
    #            tools and the compute buffer grows
    n_fast = smallest_n(SPARE_FITS_MIB)
    n_opt = smallest_n(SPARE_MIN_MIB)
    if source != "tensor-table":
        # No exact split available, so fall back to the measured ratio as a
        # floor. With the tensor table this floor would be harmful, because it is
        # a COARSER estimate than the one already in hand. Q5 is the proof: the
        # ratio floor says 30, the exact per-layer curve says 27, and 27 is what
        # the user measured - faster, and still leaving 4.8 GB free.
        n_opt = max(n_opt, int(math.ceil(layers * start_ratio)))
    n_opt = max(n_fast, min(n_layers, n_opt))
    n_safe = max(n_opt, smallest_n(SPARE_SAFE_MIB))
    n_high = min(n_layers, n_safe + 2)
    sweep = sorted({v for v in (n_fast, (n_fast + n_opt) // 2, n_opt,
                                (n_opt + n_safe) // 2, n_safe, n_high)
                    if 0 <= v <= n_layers})
    return {"n_init": n_opt, "n_opt": n_opt, "n_safe": n_safe,
            "n_fast": n_fast, "n_low": n_fast, "n_high": n_high,
            "sweep": sweep, "layers": n_layers, "experts": meta.get("experts"),
            "e_layer_gb": round(e_layer_gb, 3), "w_non_gb": round(w_non, 3),
            "kv_gb": round(kv, 2), "extra_gb": round(extra_gb, 2),
            "compute_gb": round(compute_gb, 2), "source": source,
            "spare_opt_mib": round(spare_mib(n_opt)),
            "spare_safe_mib": round(spare_mib(n_safe))}


GROUP_LETTER = {"gemma": "G", "qwen": "Q"}


def make_model_id(rel, meta, prof, overrides, ctx, mmproj=False, draft=None):
    """Build an ASCII, sortable model id: `Q5 Qwen3.6-35B-A3B-64K-MoE-IMG`.

    Mirrors the manual Phase 3 naming so a freshly scanned model looks like the
    hand-curated ones instead of exposing the raw directory name in the router
    list. The dir name is kept as an `alias`, so nothing that referenced it
    breaks.
    """
    stem = rel.split("/")[-1]
    fam = (prof or {}).get("family") or family_of(rel)
    letter = GROUP_LETTER.get(fam, "O")
    used = set()
    for e in (overrides.get("models") or {}).values():
        m = re.match(r"^([A-Z])(\d+)\b", str(e.get("id") or ""))
        if m:
            used.add((m.group(1), int(m.group(2))))
    n = 1
    while (letter, n) in used:
        n += 1

    m = re.match(r"^([A-Za-z][A-Za-z0-9.]*)", stem)
    parts = [m.group(1) if m else stem[:10]]
    size = find_size(stem)
    if size:
        parts.append(size)
    active = re.search(r"-A(\d+B)\b", stem, re.I)
    if active:
        parts.append("A" + active.group(1).upper())
    if "qat" in stem.lower():
        parts.append("QAT")
    ctx_k = "{}K".format(int(ctx) // 1024) if ctx else "NG"

    tags = []
    if meta.get("experts"):
        tags.append("MoE")
    if draft:
        tags.append("MTP")
    if mmproj:
        tags.append("IMG")
    label = "{} {}-{}".format(letter + str(n), "-".join(parts), ctx_k)
    if tags:
        label += "-" + "-".join(tags)
    label += derivative_suffix(stem)
    return label


# Two checkpoints that behave nothing alike can otherwise render identically in
# a model picker: "Q5 Qwen3.6-35B-A3B-64K-MoE-IMG" says nothing about being an
# uncensored community finetune, so it is indistinguishable from the official
# Q1 build. These markers are therefore part of the id, not the alias.
UNCENSORED_MARKERS = ("uncensored", "heretic", "abliterated", "aggressive",
                      "dolphin", "lewd", "unfiltered")
PUBLISHER_MARKERS = (("hauhaucs", "Hau"),)


def derivative_suffix(stem):
    """`-UNC` / `-Hau` style suffix derived from the directory name.

    Kept as a pure function so the same rule can be re-applied to ids that were
    written before it existed (see plan/_apply_unc_ids.py) - a model scanned
    last week must not end up looking different from one scanned tomorrow.
    """
    low = stem.lower()
    out = ""
    if any(k in low for k in UNCENSORED_MARKERS):
        out += "-UNC"
    for key, tag in PUBLISHER_MARKERS:
        if key in low:
            out += "-" + tag
    return out


def derive_overrides(dirs, mtp_files, profiles, overrides, chat):
    """Create/fill a preset-overrides.json entry for every model on disk.

    Existing values are NEVER overwritten - explicit configuration (especially
    measured values) always wins, so this is safe to re-run after tuning.
    Returns (created, changed) counts.
    """
    models = overrides.setdefault("models", {})
    glob = overrides.setdefault("global", {})
    created = changed = 0
    for rel in sorted(dirs, key=str.lower):
        main = dirs[rel]["mains"][0]
        entry = models.get(rel)
        if entry is None:
            entry = models[rel] = {}
            created += 1
        before = json.dumps(entry, sort_keys=True, ensure_ascii=False)

        prof, prof_how = profile_match(rel, profiles)
        if prof_how == "neighbour":
            # TASK-078: say it out loud. A neighbour match is a guess with a
            # stated basis, and it is the one case where a reviewer should look
            # twice - the value is about to be inherited by a model nobody has
            # measured.
            log("[profile] {}: no exact profile - inherited from '{}' "
                "(same family + parameter size)"
                .format(rel[:44], (prof or {}).get("id")))
        elif prof is None:
            log("[profile] {}: no profile match - using generic defaults"
                .format(rel[:44]))
        gguf = Path(chat) / rel / main
        meta = read_gguf_meta(str(gguf))
        draft = pick_mtp_draft(main, mtp_files, entry.get("mtp"))
        weight_gb = gguf.stat().st_size / 1e9 if gguf.exists() else 0.0
        eff_ctx = int((entry.get("params") or {}).get("ctx-size")
                      or glob.get("ctx-size") or MOE_DEFAULT_CTX)

        if not entry.get("id"):
            entry["id"] = make_model_id(
                rel, meta, prof, overrides, eff_ctx,
                mmproj=bool(dirs[rel].get("mmproj")), draft=draft)
        if not entry.get("alias"):
            entry["alias"] = [rel.split("/")[-1]]

        if not entry.get("note"):
            entry["note"] = describe_model(rel, dirs[rel], prof, meta, draft)

        params = entry.setdefault("params", {})
        if prof:
            kv = prof.get("kv") or {}
            for key, sub in (("cache-type-k", "k"), ("cache-type-v", "v")):
                if kv.get(sub) and key not in params and kv[sub] != glob.get(key):
                    params[key] = kv[sub]
            ctx = prof.get("ctx_recommend")
            if ctx and "ctx-size" not in params and ctx != glob.get("ctx-size"):
                params["ctx-size"] = ctx
            samp = prof.get("sampling") or {}
            for src, dst in (("temp", "temp"), ("top_p", "top-p"),
                             ("top_k", "top-k"), ("min_p", "min-p"),
                             ("presence_penalty", "presence-penalty"),
                             ("repeat_penalty", "repeat-penalty")):
                if src in samp and dst not in params:
                    params[dst] = samp[src]

        est = None
        if "n-cpu-moe" not in params and weight_gb >= MOE_OFFLOAD_MIN_GB:
            kv_type = str(params.get("cache-type-v")
                          or glob.get("cache-type-v") or "f16")
            moe_hint = (prof or {}).get("moe") or {}
            est = estimate_ncpu_moe(
                meta, weight_gb, eff_ctx, kv_type,
                start_ratio=moe_hint.get("start_ratio",
                                         MOE_START_OFFLOAD_RATIO),
                # TASK-074/075: a NEW model gets the exact tensor-table split, not
                # the 88% share guess, and the interval walk that follows from it.
                tensors=read_gguf_tensors(str(gguf)),
                batch=params.get("batch-size") or glob.get("batch-size"),
                vram_calib=(entry.get("tuning") or {}).get("vram"),
                extra_gb=extra_vram_gb(dirs[rel], chat, rel))
            if est:
                params["n-cpu-moe"] = est["n_init"]
                log("[moe] {}: {} layers -> --n-cpu-moe {} "
                    "(range {}..{} | {} MiB free | {})".format(
                        rel[:44], est["layers"], est["n_init"], est["n_fast"],
                        est["n_safe"], est["spare_opt_mib"], est["source"]))

        # FND-068: a model that needs expert offloading is by definition tight
        # on VRAM, and --batch-size sizes the compute buffer directly. Give it a
        # conservative default so Router mode and the .bat launchers agree, and
        # so it does not silently inherit llama.cpp's larger default. Dense
        # models keep the default - they have room and a bigger batch speeds up
        # prompt processing.
        if "n-cpu-moe" in params and "batch-size" not in params:
            params["batch-size"] = MOE_DEFAULT_BATCH

        if draft:
            mtp = entry.setdefault("mtp", {})
            mtp.setdefault("id_suffix", "-MTP")
            mtp.setdefault("draft", draft)
            mtp.setdefault("params", {})
            mtp.setdefault("spec-draft-n-max", 2)

        tuning = entry.setdefault("tuning", {})
        # FND-064/067: measurements may live under measured / measured_realctx /
        # measured_benchctx. Reading only tuning["measured"] used to (a) add an
        # empty `measured` key to entries that already held real measurements and
        # (b) flip their `source` from user-measured back to heuristic.
        have_measured, _ = measured_points(entry)
        if est:
            # DEC-002: the byte-budget floor anchors the sweep, so --tune-sweep can
            # hunt for the real optimum below the conservative starting value.
            tuning.setdefault("sweep", est["sweep"])
            if not have_measured:
                # TASK-078: record WHERE the starting value came from. "the tool
                # computed it" and "it was inherited from a lookalike model" are
                # very different levels of trust, and both --audit and MODELS.md
                # read this field to label the number.
                pid = (prof or {}).get("id")
                if pid:
                    tuning.setdefault("source", "inherited-from:{}".format(pid))
                    if prof_how != "exact":
                        tuning.setdefault("source_pass", prof_how)
                else:
                    tuning.setdefault("source", "vram-budget")
        if "n-cpu-moe" in params:
            cur = int(params["n-cpu-moe"])
            tuning.setdefault("sweep", sorted({max(0, cur - 6), max(0, cur - 3),
                                               cur, cur + 3, cur + 6}))
        tuning.setdefault("source",
                          "user-measured" if have_measured else "heuristic")

        if json.dumps(entry, sort_keys=True, ensure_ascii=False) != before:
            changed += 1
    return created, changed


def report_new_models(unknown, dirs, mtp_files, profiles, overrides, chat):
    """Print what WOULD be written for each unconfirmed model dir (DEC-001).

    Called instead of writing, so the user can approve (or fix the profile)
    before a half-configured preset ever reaches the router.
    """
    log("=" * 64)
    log("== NEW model(s) found on disk - confirmation required (DEC-001) ==")
    log("=" * 64)
    for rel in unknown:
        e = model_entry_for(overrides, rel)
        main = dirs[rel]["mains"][0]
        gguf = Path(chat) / rel / main
        meta = read_gguf_meta(str(gguf))
        weight = gguf.stat().st_size / 1e9 if gguf.exists() else 0.0
        prof = profile_for_dirname(rel, profiles)
        params = e.get("params") or {}
        log("")
        log("  [{}]".format(rel))
        log("      id       : {}".format(
            e.get("id") or default_section_id(main)))
        if meta.get("experts"):
            log("      架构     : {}  MoE  {} 层 / {} 专家  权重 {:.2f} GB".format(
                meta.get("arch", "?"), meta.get("layers"),
                meta.get("experts"), weight))
        else:
            log("      架构     : {}  dense  {} 层  权重 {:.2f} GB".format(
                meta.get("arch", "?"), meta.get("layers"), weight))
        if e.get("alias"):
            log("      兼容别名 : {}".format(", ".join(e["alias"])))
        if params:
            log("      将写入   : " + "  ".join(
                "{}={}".format(k, v) for k, v in sorted_params(params)))
        else:
            log("      将写入   : (无 - 继承 [*] 全局默认)")
        log("      参数来源 : {}   (profile: {})".format(
            (e.get("tuning") or {}).get("source", "?"),
            (prof or {}).get("id") or "无匹配"))
        if "n-cpu-moe" in params:
            sweep = (e.get("tuning") or {}).get("sweep") or []
            log("")
            log("      [!] n-cpu-moe={} 是显存预算的保守估算，未实测。".format(
                params["n-cpu-moe"]))
            log("          候选档位: {}".format(sweep))
            log("          确认后运行: llama-hub.bat --tune-sweep {}".format(rel))
        else:
            sweep = (e.get("tuning") or {}).get("sweep") or []
            if sweep:
                log("      候选档位 : {}".format(sweep))
        if prof is None:
            log("      [!] 没有任何 profile 匹配这个目录 - 上面的参数可能不适用")
        log("      [?] 建议联网核实官方采样 / 上下文 / 已知问题:")
        log("          https://huggingface.co/models?search={}".format(
            rel.split("/")[-1]))
        log("          https://github.com/ggml-org/llama.cpp/issues?q=n-cpu-moe")
    log("")
    log("-" * 64)
    log("[X] {} new model(s) need confirmation - NOTHING was written.".format(
        len(unknown)))
    log("    Review the candidates above, then re-run with --accept-new.")
    log("    (llama-hub.bat menu 1 offers this automatically.)")
    log("")


def audit_moe_section(overrides):
    """Report the n-cpu-moe ladder: configured value vs best measured value."""
    log("== MoE offload audit (n-cpu-moe) ==")
    log("VRAM 16GB - --n-cpu-moe puts the first N layers' expert weights on CPU")
    log("")
    any_row = False
    low = []
    for rel, e in sorted((overrides.get("models") or {}).items()):
        params = e.get("params") or {}
        if "n-cpu-moe" not in params:
            continue
        any_row = True
        cur = params["n-cpu-moe"]
        measured, origin = measured_points(e)
        src = (e.get("tuning") or {}).get("source", "?")
        best = max(measured, key=lambda m: m.get("tps", 0)) if measured else None
        log("  {:<48} current={:<4} source={}".format(rel[:48], cur, src))
        # TASK-088: pair the speed number with the VRAM it costs. A model can
        # clear 35 t/s and still leave ~900 MiB free, which passes every other
        # check here and then OOMs on the first image - exactly what happened to
        # the Q5 model at n-cpu-moe 20 on 2026-09-13.
        ctx = int(params.get("ctx-size") or MOE_DEFAULT_CTX)
        est, head, ok = vram_verdict(e, params, ctx=ctx)
        if est is None:
            log("      VRAM = ?      (no calibration yet - see "
                "plan/_apply_q5_vram_calib.py)")
        else:
            log("      VRAM = {} MiB used, {} MiB free @ {}K ctx   [{}]".format(
                round(est), round(head), ctx // 1024,
                "OK" if ok else "!! under {} MiB - an image request can OOM"
                .format(TUNE_MIN_HEADROOM_MIB)))
            if not ok:
                low.append(rel)
        if best:
            tag = {"realctx": "real-ctx", "measured": "measured",
                   "benchctx": "BENCH-CTX - ordering only"}.get(origin, origin)
            log("      best measured = {} -> {} t/s   [{}]".format(
                best.get("n-cpu-moe"), best.get("tps"), tag))
            # FND-067: only a real-context point may drive the choice. A
            # bench-context number is not comparable - llama-bench's tiny
            # context never reaches the VRAM ceiling, so its ladder keeps
            # improving all the way down and would send you past the wall.
            if origin == "benchctx":
                log("      [i] bench-ctx numbers cannot pick a value - "
                    "verify at the router context first")
            else:
                wall = (e.get("tuning") or {}).get("vram_wall") or {}
                cur_tps = next((m.get("tps") for m in measured
                                if m.get("n-cpu-moe") == cur), None)
                if cur_tps is not None:
                    log("      current = {} t/s".format(cur_tps))
                if cur != best.get("n-cpu-moe") and cur_tps is not None:
                    delta = best.get("tps", 0) - cur_tps
                    best_n = best.get("n-cpu-moe")
                    wall_n = wall.get("ncpu_moe_wall")
                    if wall_n is not None and best_n <= wall_n:
                        # Report the wall finding even when the gain is below the
                        # nag threshold: it is what explains why the fastest value
                        # is not the chosen one. Without it the config looks like
                        # an oversight rather than a deliberate margin decision.
                        log("      [i] fastest is {} (+{:.1f} t/s) but it sits at/below"
                            " the VRAM wall ({}) - no safety margin left; keeping {}"
                            .format(best_n, delta, wall_n, cur))
                    elif delta >= max(MIN_GAIN_TPS, cur_tps * MIN_GAIN_FRAC):
                        log("      [!] {} gives +{:.1f} t/s - consider switching"
                            .format(best_n, delta))
        elif (e.get("tuning") or {}).get("sweep"):
            log("      sweep available: {} (run --tune-sweep {})".format(
                e["tuning"]["sweep"], rel))
    if low:
        log("  [!] {} model(s) leave less than {} MiB free: {}".format(
            len(low), TUNE_MIN_HEADROOM_MIB, ", ".join(low)))
    if not any_row:
        log("  (no model configures n-cpu-moe)")
    log("")


# ==================================================================== main
def load_registry():
    if not REGISTRY.exists():
        raise SystemExit(
            "[X] {} not found - run first:  update_launchers.py --extract"
            .format(REGISTRY.name))
    with REGISTRY.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg):
    if REGISTRY.exists():
        backup_file(REGISTRY)
    with REGISTRY.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def report_ini_changes(old_text, new_text):
    def sections(t):
        return {ln[1:-1] for ln in t.splitlines()
                if ln.startswith("[") and ln.endswith("]")}
    so, sn = sections(old_text), sections(new_text)
    return sorted(sn - so), sorted(so - sn)


def main():
    ap = argparse.ArgumentParser(description="llama.cpp launcher auto-updater")
    ap.add_argument("--extract", action="store_true",
                    help="re-extract launcher-models.json from the 3 launchers")
    ap.add_argument("--check", action="store_true",
                    help="dry run: report + preview only")
    ap.add_argument("--yes", action="store_true",
                    help="apply without confirmation")
    ap.add_argument("--no-scan", action="store_true",
                    help="render registry verbatim (regression test)")
    ap.add_argument("--audit", action="store_true",
                    help="compare registry entries with curated profiles + VRAM estimate")
    ap.add_argument("--fix-mmproj", action="store_true",
                    help="update static mmproj var/body references in the "
                         "registry to the newest on-disk mmproj file "
                         "(dtype change fix)")
    ap.add_argument("--fix-bodies", action="store_true",
                    help="repair stored entry bodies that contain a blank "
                         "line inside a '^' continuation (FND-014: cmd.exe "
                         "silently drops every arg after such a line)")
    ap.add_argument("--fix-moe", action="store_true",
                    help="rewrite menu entries that hard-code -ngl for a MoE "
                         "model (an explicit -ngl disables --fit, so such an "
                         "entry OOMs); converts them to --n-cpu-moe N. Also "
                         "repairs stale --n-cpu-moe values and strips the "
                         "decorative --fit/--fit-ctx that cannot run next to "
                         "--n-cpu-moe (FND-064/FND-066)")
    ap.add_argument("--chat", default=(str(DEFAULT_CHAT) if DEFAULT_CHAT else None),
                    help="models directory; also read from the CHAT_DIR environment "
                         "variable. No built-in default on purpose - the models "
                         "path belongs to the machine, not to this tool.")
    ap.add_argument("--derive-params", action="store_true",
                    help="create/fill preset-overrides.json entries for every "
                         "model on disk (never overwrites existing values)")
    ap.add_argument("--accept-new", action="store_true",
                    help="allow writing when the scan finds model dirs that have "
                         "no preset-overrides.json entry yet (DEC-001: without "
                         "this flag such models abort the run with exit code 2 "
                         "so nothing half-configured reaches the router)")
    ap.add_argument("--ini-diff", action="store_true",
                    help="show the section/key diff of models-config.ini "
                         "without writing anything")
    ap.add_argument("--tune-sweep", nargs="?", const="all", default=None,
                    metavar="DIR",
                    help="benchmark the tuning.sweep ladder of n-cpu-moe for "
                         "one model dir (or 'all') and append to "
                         "preset-overrides.json; makes no other changes")
    ap.add_argument("--reps", type=int, default=3,
                    help="llama-bench repetitions for --tune-sweep "
                         "(default: %(default)s)")
    ap.add_argument("--ctx", type=int, default=None, metavar="N",
                    help="context to sweep at (only useful with --tune-sweep). "
                         "llama-bench has no --ctx-size; it sizes the context "
                         "as n_prompt + n_gen, so this is passed as -p N. "
                         "Results are tagged with the ctx and land in "
                         "tuning.measured_realctx, where they may decide the "
                         "value; without it they land in measured_benchctx "
                         "and may only rank (FND-067). Cost: -p 65536 makes "
                         "every repetition take minutes")
    ap.add_argument("--audit-drafts", action="store_true",
                    help="static MTP drafter report: family pairing matrix, "
                         "orphans, which launcher variable references each "
                         "file, naming traps (loads nothing)")
    ap.add_argument("--validate-drafts", action="store_true",
                    help="verify every MTP (model, draft) pair through "
                         "llama-server and write draft-health.json; "
                         "unhealthy drafts are reported and can be "
                         "blacklisted with --blacklist-drafts")
    ap.add_argument("--configured-only", action="store_true",
                    help="with --validate-drafts: test only the pairs named in "
                         "preset-overrides.json instead of every "
                         "family-consistent combination + controls")
    ap.add_argument("--fast", action="store_true",
                    help="with --validate-drafts: metadata checks only "
                         "(no model loading)")
    ap.add_argument("--blacklist-drafts", action="store_true",
                    help="with --validate-drafts: also write the failing "
                         "drafts into draft-blacklist.json")
    ap.add_argument("--paths", action="store_true",
                    help="print every resolved path the updater uses and exit "
                         "(self-check after moving this folder)")
    args = ap.parse_args()
    # Keep the console's own encoding. Forcing UTF-8 here makes every Chinese
    # character arrive as mojibake on a GBK console (the reporter prints the
    # new-model candidates in Chinese). errors="replace" still guarantees we can
    # never crash on an un-encodable character.
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    if not args.chat:
        ap.error("no models directory. Pass --chat <models-dir>, or set the "
                 "CHAT_DIR environment variable. There is no built-in default: "
                 "the models path belongs to the machine, not to this tool.")
    chat = Path(args.chat)

    if args.paths:
        exe = find_server_exe()
        log("== resolved paths ==")
        log("  script folder (BASE) : {}".format(BASE))
        log("  registry             : {}  exists={}".format(
            REGISTRY, REGISTRY.exists()))
        log("  profiles             : {}  exists={}".format(
            PROFILES_PATH, PROFILES_PATH.exists()))
        log("  overrides            : {}  exists={}".format(
            OVERRIDES_PATH, OVERRIDES_PATH.exists()))
        log("  blacklist            : {}  exists={}".format(
            BLACKLIST_PATH, BLACKLIST_PATH.exists()))
        log("  health               : {}  exists={}".format(
            HEALTH_PATH, HEALTH_PATH.exists()))
        log("  backup / preview     : {}  /  {}".format(BACKUP, PREVIEW_DIR))
        log("  key cache            : {}  exists={}".format(
            KEYS_CACHE, KEYS_CACHE.exists()))
        log("  chat (models)        : {}  exists={}".format(chat, chat.exists()))
        log("  llama-server.exe     : {}".format(
            exe if exe else "NOT FOUND"))
        log("  llama-bench.exe      : {}".format(
            next((p for p in (BASE / "llama-bench.exe",
                              BASE.parent / "llama-bench.exe") if p.exists()),
                 "NOT FOUND")))
        for name, (script, enc) in SPECS.items():
            p = BASE / script
            log("  launcher {:<7}     : {}  exists={}".format(
                name, p, p.exists()))
        return

    if args.extract:
        reg = {
            "version": 1,
            "chat_dir": str(chat),
            "mtp_dir_name": MTP_DIR_NAME,
            "family": {"gemma": "gemma4", "qwen": "qwen", "lfm": "cpu",
                       "phi": "qwen", "glm": "qwen", "gpt": "qwen",
                       "default": "qwen"},
            "launchers": {},
        }
        for name in SPECS:
            reg["launchers"][name] = extract_launcher(name)
            log("[OK] extracted {}  ({})".format(
                SPECS[name][0], len(reg["launchers"][name]["entries"])))
        save_registry(reg)
        log("[OK] registry written: {}".format(REGISTRY))
        return

    reg = load_registry()
    profiles = load_profiles()
    overrides = load_overrides()
    set_blacklist(load_blacklist())
    if _blacklist:
        log("[blacklist] {} draft(s) excluded: {}".format(
            len(_blacklist), ", ".join(sorted(_blacklist))))
        for d, why in sorted(_blacklist.items()):
            log("    {} - {}".format(d, why))
        log("")

    # ---------------- audit mode (read-only)
    if args.audit:
        audit_registry(reg, chat, profiles)
        if overrides:
            audit_moe_section(overrides)
        else:
            log("[!] preset-overrides.json not found - run --derive-params "
                "to create it")
        return

    # ---------------- parameter sweep (writes only preset-overrides.json)
    if args.tune_sweep is not None:
        if not overrides:
            raise SystemExit("[X] preset-overrides.json not found - run "
                             "update-launchers.bat --derive-params first")
        n = tune_sweep(overrides, chat, target=args.tune_sweep, reps=args.reps,
                       ctx=args.ctx)
        if n:
            save_overrides(overrides)
            log("[OK] {} measurement(s) appended to {}".format(
                n, OVERRIDES_PATH.name))
            log("")
            audit_moe_section(overrides)
        else:
            log("[!] nothing measured - check tuning.sweep in {}".format(
                OVERRIDES_PATH.name))
        return

    # ---------------- render verbatim (regression mode)
    if args.no_scan:
        for name, cfg in reg["launchers"].items():
            text = render_launcher(cfg, cfg["entries"], [])
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            write_text(PREVIEW_DIR / cfg["script"], text, cfg["encoding"])
            log("[preview] {} ({} entries, unchanged)".format(
                cfg["script"], len(cfg["entries"])))
        return

    # ---------------- scan and plan
    dirs, mtp_files = scan_chat(chat)
    log("== llama.cpp launcher auto-update  {} ==".format(ts()))
    log("Scan: {} ({} model dirs, {} MTP drafts)".format(
        chat, len(dirs), len(mtp_files)))
    log("")

    # ---------------- mmproj static-variable drift (dtype change F16->F32)
    changed = False
    if args.fix_mmproj:
        n = fix_mmproj_drift(reg, dirs, chat)
        if n:
            changed = True
            log("[fix-mmproj] updated {} static mmproj reference(s) in "
                "registry (re-render below reflects them)".format(n))
    else:
        drift = mmproj_drift_report(reg, dirs, chat)
        if drift:
            log("[!] run 'update-launchers.bat --fix-mmproj' to sync static "
                "mmproj references to the newest on-disk files")
    log("")

    # ---------------- blank-line continuation repair (FND-014)
    if args.fix_bodies:
        n = fix_blank_continuations(reg)
        if n:
            changed = True
            log("[fix-bodies] repaired {} entry body/bodies in the registry "
                "(re-render below reflects them)".format(n))
        else:
            log("[fix-bodies] no blank line inside a caret continuation - "
                "nothing to repair")
        log("")

    # ---------------- derived per-model parameters (REQ-012)
    derived = False
    if overrides is None:
        overrides = {
            "version": 2,
            "ini_path": str(chat / "models-config.ini"),
            "description": ("Per-model tunables for the router preset. Edit "
                            "this file, not models-config.ini (that file is "
                            "generated)."),
            "global": {},
            "models": {},
        }
    if not overrides.get("global"):
        cur_glob = read_ini_sections(chat / "models-config.ini").get("*", {})
        overrides["global"] = {k.split("=")[0].strip(): k.split("=", 1)[1].strip()
                               for k in cur_glob} or {
            "n-gpu-layers": 99, "ctx-size": 65536, "flash-attn": "on",
            "jinja": "true", "cache-type-k": "q8_0", "cache-type-v": "q8_0",
        }
        derived = True
    known = set(overrides.get("models") or {})
    unknown = [rel for rel in dirs if rel not in known]
    if args.derive_params or not OVERRIDES_PATH.exists() or unknown:
        created, touched = derive_overrides(dirs, mtp_files, profiles,
                                            overrides, chat)
        derived = True
        log("[derive] {} entries created, {} updated "
            "(existing values preserved)".format(created, touched))
        log("")
    if derived:
        new_json = json.dumps(overrides, ensure_ascii=False, indent=2) + "\n"
        old_json = (OVERRIDES_PATH.read_text(encoding="utf-8")
                    if OVERRIDES_PATH.exists() else None)
        if old_json != new_json:
            changed = True

    # ---------------- key whitelist gate (FND-029 / CON-007)
    allowed = server_option_keys()
    bad_keys = validate_preset_keys(overrides, allowed)
    if bad_keys:
        for rel, k in bad_keys:
            log("[X] preset key '{}' (in {}) is not recognised by this "
                "llama-server build - the router would refuse to start"
                .format(k, rel))
        raise SystemExit("[X] {} invalid preset key(s) - fix {} before "
                         "writing".format(len(bad_keys), OVERRIDES_PATH.name))
    if allowed:
        log("[keys] whitelist OK ({} llama-server options cached in {})"
            .format(len(allowed), KEYS_CACHE.name))
        log("")

    # ---------------- MoE menu entries that disable --fit (FND-033)
    # Runs BEFORE the new-model gate so one pass reports both classes of problem.
    if args.fix_moe:
        n = fix_moe_entries(reg, dirs, chat, profiles, overrides)
        if n:
            changed = True
            log("[fix-moe] repaired {} MoE entry/entries in the registry "
                "(re-render below reflects them)".format(n))
        else:
            log("[fix-moe] no MoE entry hard-codes -ngl - nothing to repair")
        log("")
    else:
        drift = find_moe_gaps(reg, dirs, chat, profiles, overrides)
        if drift:
            ngl = [d for d in drift if d[6] == "ngl"]
            log("[!] {} MoE menu entr(ies) disagree with "
                "preset-overrides.json:".format(len(drift)))
            for script, label, _e, rel, ncpu, _ctx, kind in drift:
                why = {"ngl": "will OOM", "moe": "no expert offload"}.get(
                    kind, kind)
                log("      [{}] {} :: {}".format(why, script, label))
            if ngl:
                log("[!]   {} of them hard-code -ngl, which disables --fit and "
                    "OOMs on load".format(len(ngl)))
            log("[!] run 'llama-hub.bat --fix-moe' to repair them")
            log("")

    # ---------------- new-model confirmation gate (REQ-025 / DEC-001)
    # A model dir with no preset-overrides.json entry would otherwise be
    # written as a bare [section] carrying only `model =` - on a 16GB card a
    # 22GB MoE then loads with no expert offload and no ctx limit. Refuse.
    if unknown and not args.accept_new:
        report_new_models(unknown, dirs, mtp_files, profiles, overrides, chat)
        raise SystemExit(2)

    # ---------------- MTP drafter static audit (VER-02), ends here
    if args.audit_drafts:
        audit_drafts(reg, dirs, mtp_files, chat, overrides)
        return

    # ---------------- MTP draft health (REQ-006), ends here
    if args.validate_drafts:
        set_blacklist({})          # validate what is on disk, ignore the list
        health, failed = validate_drafts(dirs, mtp_files, chat, overrides,
                                         deep=not args.fast,
                                         configured_only=args.configured_only)
        HEALTH_PATH.write_text(
            json.dumps(health, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        log("")
        log("== draft health ==")
        for d in health["drafts"]:
            log("  {:<40} arch={} L={} meta={}".format(
                d["file"][:40], d.get("arch"), d.get("layers"),
                "ok" if d["meta_ok"] else "BAD: " + str(d.get("meta_note"))))
        for p in health["pairs"]:
            log("  {:<44} {}{}".format(
                "{} + {}".format(p["model"][:24], p["draft"][:18]),
                "[CONTROL] " if p.get("control") else "",
                ("OK  acceptance={:.3f} mean_len={}".format(
                    p["acceptance"], p["mean_len"])
                 if p.get("healthy") else "REJECTED - {}".format(
                     p.get("reason"))) if p.get("ok") else
                "FAIL - {}".format(p.get("reason"))))
        log("")
        log("[OK] wrote {}".format(HEALTH_PATH.name))
        if failed:
            log("")
            log("{} unhealthy draft(s):".format(len(failed)))
            for d, why in sorted(failed.items()):
                log("  [X] {} - {}".format(d, why))
            if args.blacklist_drafts:
                entries = [{"file": d, "reason": w}
                           for d, w in sorted(failed.items())]
                save_blacklist(entries)
                log("")
                log("[OK] wrote {} ({} entr{})".format(
                    BLACKLIST_PATH.name, len(entries),
                    "y" if len(entries) == 1 else "ies"))
                log("     blacklisted drafts are no longer offered by the "
                    "router preset")
        else:
            log("")
            log("all configured drafts are healthy")
        return

    all_dirs = {d for c in reg["launchers"].values()
                for e in c["entries"] for d in e.get("dirs", [])}
    new_dirs = sorted(d for d in dirs if d not in all_dirs)

    plan = {}          # launcher -> (kept, removed, extra_groups, extra_entries)
    for name, cfg in reg["launchers"].items():
        kept, removed = [], []
        for e in cfg["entries"]:
            missing = [d for d in e.get("dirs", []) if d not in dirs]
            if missing:
                removed.append((e, missing))
            else:
                kept.append(e)
        plan[name] = (kept, removed, [], [])
        if removed:
            changed = True
        if kept and mtp_files is not None:
            for e in kept:
                for d in e.get("drafts", []):
                    if d not in mtp_files:
                        log("[!] {}: draft missing on disk: {} (entry '{}')"
                            .format(cfg["script"], d, e["label"]))

    taken_names = set()
    for cfg in reg["launchers"].values():
        for item in cfg["region"]:
            if item.get("t") == "group":
                for vln in item.get("vars") or []:
                    m = VAR_CHAT_RE.match(vln)
                    if m:
                        taken_names.add(m.group(1))
    for rel in new_dirs:
        fam = family_of(rel)
        launcher = reg["family"].get(fam, reg["family"]["default"])
        groups, entries = make_auto(
            launcher, rel, dirs[rel], mtp_files, taken_names, profiles,
            chat=chat, overrides=overrides)
        plan[launcher][2].extend(groups)
        plan[launcher][3].extend(entries)
        changed = True

    # ---------------- ini
    ini_path = chat / "models-config.ini"
    ini_old = None
    ini_new = None
    if ini_path.exists():
        ini_old, _ = read_text(ini_path, "utf-8")
    ini_existing = read_ini_sections(ini_path)
    ini_new, ini_preserved = gen_ini_v2(dirs, mtp_files, chat,
                                        overrides=overrides,
                                        existing=ini_existing)
    if ini_old is not None and ini_old != ini_new:
        changed = True

    # ---------------- ini diff mode (read-only, ends here)
    if args.ini_diff:
        added, dropped, lines = diff_ini_sections(ini_old or "", ini_new)
        log("== models-config.ini diff (preview, nothing written) ==")
        for s in dropped:
            log("  - [{}]".format(s))
        for s in added:
            log("  + [{}]".format(s))
        for op, sec, key, val in lines:
            log("  {} [{}] {} = {}".format(op, sec, key, val))
        log("")
        log("  {} section(s) added / {} dropped / {} key change(s)".format(
            len(added), len(dropped), len(lines)))
        return

    # ---------------- report
    for name, cfg in reg["launchers"].items():
        kept, removed, xg, xe = plan[name]
        log("[{}]".format(cfg["script"]))
        if removed:
            log("  remove {}:".format(len(removed)))
            for e, missing in removed:
                log("    - '{}'  [dir deleted: {}]".format(
                    e["label"], ", ".join(missing)))
        if xe:
            log("  add {} (auto, [NEW]):".format(len(xe)))
            for e in xe:
                src = "  [profile: {}]".format(e["profile"]) if e.get("profile") else ""
                log("    + '{}'{} ".format(e["label"], src))
        if not removed and not xe:
            log("  no change ({} entries)".format(len(kept)))
    log("")
    log("[models-config.ini]")
    if ini_old is None:
        log("  (file missing, will be created)")
    elif ini_old != ini_new:
        added, removed = report_ini_changes(ini_old, ini_new)
        log("  + add {} sections / - drop {} sections".format(
            len(added), len(removed)))
        for s in removed:
            log("    - [{}]".format(s))
        for s in added:
            log("    + [{}]".format(s))
    else:
        log("  no change")
    # hand-tuned key preservation report (FND-015 guard).
    # `ini_preserved` is what gen_ini_v2 actually carried over - keys supplied
    # by preset-overrides.json are not "preserved", they are generated.
    if ini_preserved:
        by_sec = {}
        for sec, ln in ini_preserved:
            by_sec.setdefault(sec, []).append(ln)
        log("  carried over {} key line(s) not managed by "
            "preset-overrides.json:".format(len(ini_preserved)))
        for sec in sorted(by_sec):
            log("    [{}] {}".format(sec, "; ".join(by_sec[sec])))
    else:
        log("  every tunable comes from preset-overrides.json "
            "(nothing carried over)")
    log("")

    # render everything and compare with the files on disk, so that
    # content-level registry edits (body/label tweaks) are detected too
    rendered = {}
    for name, cfg in reg["launchers"].items():
        kept, removed, xg, xe = plan[name]
        entries = kept + xe
        text = render_launcher(cfg, entries, xg, dirs=dirs)
        rendered[name] = (text, entries)
        path = BASE / cfg["script"]
        old, _ = read_text(path, cfg["encoding"])
        if old != text:
            changed = True

    if not changed:
        log("Nothing to update - all launchers and ini are in sync.")
        return

    # ---------------- preview mode
    if args.check:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for name, cfg in reg["launchers"].items():
            text, _ = rendered[name]
            write_text(PREVIEW_DIR / cfg["script"], text, cfg["encoding"])
        write_text(PREVIEW_DIR / "models-config.ini", ini_new, "utf-8")
        with (PREVIEW_DIR / "changes.txt").open("w", encoding="utf-8") as f:
            f.write("dry-run {}\n".format(ts()))
        log("[check] previews written to backup\\preview\\ - nothing applied.")
        return

    # ---------------- apply
    if not args.yes:
        try:
            ans = input("Apply these changes? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            log("Aborted - nothing changed.")
            return

    new_reg = json.loads(json.dumps(reg))
    for name, cfg in new_reg["launchers"].items():
        kept, removed, xg, xe = plan[name]
        cfg["entries"] = kept + xe
        cfg["region"] = cfg["region"] + xg
        if cfg["entries"] != reg["launchers"][name]["entries"] or xg:
            cfg["entries"] = kept + xe
    save_registry(new_reg)

    for name, cfg in reg["launchers"].items():
        text, entries = rendered[name]
        path = BASE / cfg["script"]
        old, _ = read_text(path, cfg["encoding"])
        if old != text:
            backup_file(path)
            write_text(path, text, cfg["encoding"])
            log("[OK] updated {} ({} entries)".format(cfg["script"], len(entries)))
        else:
            log("[=] unchanged {}".format(cfg["script"]))
    if ini_old is None or ini_old != ini_new:
        if ini_path.exists():
            backup_file(ini_path)
        tmp_path = ini_path.with_suffix(ini_path.suffix + ".tmp")
        write_text(tmp_path, ini_new, "utf-8")
        os.replace(str(tmp_path), str(ini_path))
        log("[OK] updated {}".format(ini_path.name))
    if derived and changed:
        save_overrides(overrides)
        log("[OK] updated {} ({} model entries)".format(
            OVERRIDES_PATH.name, len(overrides.get("models") or {})))
    log("")
    log("Done. Registry: {} (next run re-reads it).".format(REGISTRY.name))


if __name__ == "__main__":
    main()
