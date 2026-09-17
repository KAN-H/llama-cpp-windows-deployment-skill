# -*- coding: utf-8 -*-
"""Graft a built-in MTP / NextN head from one GGUF onto another.

Why this exists
---------------
`Qwen3.6-35B-A3B` and a growing number of architectures ship with a built-in
Multi-Token-Prediction head. The official quants carry it as an extra trailing
layer (`blk.40.*`, block_count 41). Community fine-tunes - uncensored merges in
particular - almost never ship it, so `--spec-type draft-mtp` is simply
unavailable for them even though the weights are 99% the same model.

The head is portable across fine-tunes of the same base: it tolerates
substantial weight drift in the target, which is what makes grafting work at all
(ggml-org/llama.cpp#28363 measured +28% decode, draft acceptance 0.85-0.89).

What "graft" means byte-for-byte
--------------------------------
A GGUF is

    header | key-value block | tensor table | pad | tensor data

and only four things change:

    1. `<arch>.block_count`              N -> N+k   (in-place, same 4-byte width)
    2. `<arch>.nextn_predict_layers`     ADDED = k  (this is the switch)
    3. the tensor table                  + the head's entries
    4. the data section                  + the head's bytes, appended

(2) is the non-obvious one, and the reason a naive graft is a silent no-op. The
community post says "rewrite block_count"; it does not mention that the donor
ALSO carries `nextn_predict_layers`, which is what actually tells the loader the
last block is a NextN head rather than an ordinary layer. Copy the tensors
without that key and the model loads beautifully and never speculates.

There is a SECOND silent switch on the runtime side: `mparams.load_mtp`, set by
`--spec-type draft-mtp`. Without that flag the head tensors are not even loaded
into memory. A successful graft plus a command line that omits the flag looks
exactly like a failed graft.

Keys are taken from the TARGET, never the donor: the two checkpoints differ in
their provenance keys, and it is the target's tokenizer, chat template and naming
that must survive.

What this tool refuses to do
----------------------------
Compatibility is a GATE, not a report. A wrong graft does not fail loudly: it
yields a file that either loads and silently never speculates, or fails far from
the cause. Every check below therefore runs BEFORE the multi-GiB write, and a
failure writes nothing at all - there is no half-finished artifact to clean up.

The gate is structural, derived from the two files, and the architecture table is
advisory on top of it:

  structural (hard)
    1.  both files parse as GGUF v2/v3
    2.  same general.architecture
    3.  donor block_count == target block_count + head_blocks
    4.  every TARGET tensor exists in the donor with identical dims
    5.  the donor's extra tensors are exactly the head block range
    6.  the target does not already carry a head
    7.  the donor declares nextn_predict_layers with a usable value
    8.  block_count is a u32 (so it can be spliced in place)
    9.  the head block range is non-empty
    10. every quant type in the head is one we can size
    11. no per-layer array KV desyncs when blocks are added
    12. there is room on disk for the output

  advisory (from the upstream architecture table)
    - architectures whose head tensors are loaded but never executed
    - architectures that reuse nextn_predict_layers for something else entirely
    - architectures whose head lives outside blk.N.* and needs a different writer
    - multi-block heads, where many architectures assert single-block support

Quantisation type differences between the two checkpoints are NOT a reason to
refuse. Two recipes of one base legitimately assign different types to the same
tensor, GGUF stores a type per tensor, and demanding equal types rejects the very
pairs this tool was built for. They are reported as a note.

Usage
-----
    python scripts/mtp_graft.py --check --donor <head.gguf> --target <base.gguf>
    python scripts/mtp_graft.py --go --donor <head.gguf> --target <base.gguf> \
                               --out <new.gguf>
    python scripts/mtp_graft.py --check --preset qwen36-35b-a3b \
                               --models-dir <models-dir>

`--check` is the default and never writes. `--go` requires `--out`; the target is
never modified in place.

After a successful write, verify (all three, they catch different faults):
    * the file loads and the loader reports the higher block count
    * with `--spec-type draft-mtp`, the `draft acceptance` counter is non-null
      and above ~0.3. Near zero means the head runs but predicts nothing useful,
      which is the signature of a bad graft rather than a bad idea.
    * an A/B against the un-grafted model measured IN THE SAME SESSION - an
      absolute number from someone else's box is not comparable.
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# ggml_type enum -> bytes per stored weight.
#
# Block-quantised types pack many weights into one fixed-size block, so the unit
# is bytes-per-weight, not bytes-per-block (Q4_K = 144 bytes per 256 weights =
# 0.5625 B/w). Types 4 and 5 are the deprecated Q4_0/Q4_1 variants; Q5_0 and
# Q5_1 are 6 and 7.
#
# Inlined rather than imported: this file must run with nothing but the standard
# library, on a machine that has no llama.cpp source tree and no gguf-py.
# Provenance: llama.cpp launcher ecosystem GGML_TYPE_BYTES, 34 entries, verified
# against the upstream enum. An entry missing here is refused rather than
# guessed - see gate 10.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Architecture table (advisory; the structural gate above is what decides).
#
# Source: llama.cpp master, surveyed 2026-09. The local build may be older than
# that survey, so these entries choose between "refuse", "warn" and "silent" -
# they are never the only thing standing between a bad graft and a write.
#
#   EXECUTES      loader builds a real draft context for the head
#   INERT         the tensors are declared and then skipped (TENSOR_SKIP), so a
#                 graft produces a file that is larger and buys nothing. Warn,
#                 do not refuse: the upstream list moves, and refusing on a
#                 stale table would block a graft that works.
#   REUSED        nextn_predict_layers means something else entirely
#   TOP_LEVEL     the head lives outside blk.N.* (top-level nextn.* tensors) and
#                 needs a different writer; this one would produce a file with
#                 an increased block_count and no head blocks.
# ---------------------------------------------------------------------------
ARCH_EXECUTES = {
    "qwen35moe", "qwen35", "qwen3next", "deepseek2", "deepseek32", "deepseek4",
    "glm4_moe", "glm_dsa", "bailingmoe3", "cohere2moe", "nemotron_h_moe",
    "step35", "hy_v3", "mimo2",
}
ARCH_INERT = {"glm4", "exaone4", "exaone_moe", "bailingmoe2", "dots3note"}
ARCH_REUSED = {"granite-switch"}          # n_layer_nextn is a router layer here
ARCH_TOP_LEVEL = {"gemma4-assistant"}     # head is nextn.pre_projection / post_projection

# Architectures known to accept more than one head block; everything else in
# ARCH_EXECUTES asserts single-block, so a k>1 graft would GGML_ASSERT at load.
ARCH_MULTI_BLOCK_OK = {"mimo2", "step35"}

# Key-value entries whose value is an array with ONE ENTRY PER LAYER. Adding
# blocks without extending these (or removing them) leaves the loader indexing
# past the end, or - worse - silently using the wrong entry for a layer. Audited
# rather than assumed, because the list grows with every new architecture.
PER_LAYER_ARRAY_SUFFIXES = (
    "compress_ratios",
    "shared_kv_layers",
    "recurrent_layers",
    "deepstack_layers",
    "layer_types",
)

# Documented one-command presets. They only supply defaults for paths and
# expectations - nothing here is required, and no path is baked in.
PRESETS = {
    "qwen36-35b-a3b": {
        "arch": "qwen35moe",
        "head_blocks": 1,
        "donor_dir": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
        "target_dir": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P",
        "note": "The original graft: official UD-Q4_K_XL carries blk.40, the "
                "uncensored merge does not.",
    },
}

# Set to False to stream without progress noise (the write is multi-GiB).
QUIET = False


def say(msg=""):
    if not QUIET:
        print(msg)


def align(n, a):
    return (n + a - 1) // a * a


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "{:.2f} {}".format(n, unit)
        n /= 1024.0


# ===========================================================================
# GGUF reader
# ===========================================================================

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
              7: "?", 10: "Q", 11: "q", 12: "d"}


def enc_kv_u32(key, val):
    """Encode one GGUF key-value entry holding a UINT32 (type 4)."""
    b = key.encode("utf-8")
    return (struct.pack("<Q", len(b)) + b
            + struct.pack("<I", 4) + struct.pack("<I", val))


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class GgufError(Exception):
    pass


class Gguf:
    """A read-only view of a GGUF file: header, KV block, tensor table.

    Keeps the byte SPANS of the KV entries and tensor rows, because the writer
    needs to splice `block_count` in place and copy the table verbatim.
    """

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise GgufError("not a file: {}".format(self.path))
        self.size = self.path.stat().st_size
        self.kv = {}
        self.kv_span = {}
        self.tensors = []
        with open(self.path, "rb") as f:
            head = f.read(24)
            if len(head) < 24 or head[:4] != b"GGUF":
                raise GgufError("not a GGUF file (bad magic): {}".format(self.path))
            self.version, self.n_tensors, self.n_kv = struct.unpack_from("<IQQ", head, 4)
            if self.version < 2:
                raise GgufError("GGUF v{} is too old to splice safely".format(self.version))

            # KV block. Values are SKIPPED by their declared size rather than
            # read: the tokenizer arrays alone run to megabytes on a 256K vocab.
            f.seek(24)
            for _ in range(self.n_kv):
                start = f.tell()
                klen = self._u(f, "Q", 8)
                key = f.read(klen).decode("utf-8", "replace")
                vtype = self._u(f, "I", 4)
                val = self._read_val(f, vtype)
                self.kv[key] = val
                self.kv_span[key] = (start, f.tell())
            self.kv_end = f.tell()

            for _ in range(self.n_tensors):
                tstart = f.tell()
                nlen = self._u(f, "Q", 8)
                name = f.read(nlen).decode("utf-8", "replace")
                nd = self._u(f, "I", 4)
                dims = [self._u(f, "Q", 8) for _ in range(nd)]
                ttype = self._u(f, "I", 4)
                off = self._u(f, "Q", 8)
                self.tensors.append({
                    "name": name, "dims": dims, "type": ttype, "offset": off,
                    "raw_span": (tstart, f.tell()),
                })
            self.tensor_end = f.tell()

        self.alignment = int(self.kv.get("general.alignment") or 32)
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise GgufError("general.alignment={} is not a power of two"
                            .format(self.alignment))
        self.data_start = align(self.tensor_end, self.alignment)
        self.by_name = {t["name"]: t for t in self.tensors}

    # -- primitives ---------------------------------------------------------
    def _u(self, f, fmt, size):
        b = f.read(size)
        if len(b) != size:
            raise GgufError("unexpected end of file in {}".format(self.path))
        return struct.unpack("<" + fmt, b)[0]

    def _read_val(self, f, vtype):
        if vtype == 8:                                    # string
            n = self._u(f, "Q", 8)
            return f.read(n).decode("utf-8", "replace")
        if vtype == 9:                                    # array
            atype = self._u(f, "I", 4)
            alen = self._u(f, "Q", 8)
            # Only the length matters for the audit; keep a prefix for display.
            head = []
            for _ in range(min(alen, 8)):
                head.append(self._read_val(f, atype))
            if atype == 8:
                for _ in range(max(0, alen - 8)):
                    f.seek(self._u(f, "Q", 8), 1)
            elif atype in SCALAR:
                f.seek(SCALAR[atype] * max(0, alen - 8), 1)
            else:
                raise GgufError("unknown gguf array type {}".format(atype))
            return {"_array": atype, "_len": alen, "_head": head}
        if vtype in SCALAR:
            return self._u(f, SCALAR_FMT[vtype], SCALAR[vtype])
        raise GgufError("unknown gguf value type {}".format(vtype))

    # -- derived ------------------------------------------------------------
    def arch(self):
        return self.kv.get("general.architecture", "")

    def k(self, suffix):
        """Look up `<arch>.<suffix>`, e.g. k('block_count')."""
        return self.kv.get("{}.{}".format(self.arch(), suffix))

    def tsize(self, t):
        n = 1
        for d in t["dims"]:
            n *= d
        bpe = GGML_TYPE_BYTES.get(t["type"])
        if bpe is None:
            raise GgufError("tensor {!r} uses ggml type {}, which is not in the "
                            "size table - refusing to guess".format(t["name"], t["type"]))
        return int(n * bpe)

    def layer_of(self, name):
        """The block index a tensor belongs to, or None.

        Parsed from the `blk.<N>.` prefix rather than matched against a naming
        convention: at least one architecture (bailingmoe3) names its head
        tensors with ordinary suffixes, so the NAME cannot identify the head.
        """
        if not name.startswith("blk."):
            return None
        rest = name[4:]
        dot = rest.find(".")
        if dot <= 0:
            return None
        try:
            return int(rest[:dot])
        except ValueError:
            return None

    def layers(self):
        return sorted({n for n in (self.layer_of(t["name"]) for t in self.tensors)
                       if n is not None})

    def per_layer_arrays(self):
        """KV arrays whose length tracks the layer count.

        `(key, length, first_arch_element)` for anything that either uses a
        known per-layer suffix or happens to be exactly layer_count long.
        """
        out = []
        lc = self.k("block_count")
        for key, val in self.kv.items():
            if not isinstance(val, dict) or "_len" not in val:
                continue
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                       "tokenizer.ggml.token_type", "tokenizer.ggml.scores"):
                continue
            if key.rsplit(".", 1)[-1] in PER_LAYER_ARRAY_SUFFIXES:
                out.append((key, val["_len"], "known per-layer key"))
            elif isinstance(lc, int) and val["_len"] == lc and not key.startswith("tokenizer."):
                out.append((key, val["_len"], "length == block_count"))
        return out


# ===========================================================================
# Head detection - by BLOCK INDEX RANGE, never by name
# ===========================================================================

def head_plan(donor, target, head_blocks=None):
    """Work out which donor blocks are the head, and where they land.

    Returns (first, last, nextn, problems).

    The range comes from arithmetic, not from naming: the head is the trailing
    `nextn` blocks. `layer_of()` gives the block index, so an architecture that
    names its head tensors `blk.40.some_ordinary_name` is handled the same as one
    that names them `blk.40.nextn.*`.
    """
    problems = []
    bc_d, bc_t = donor.k("block_count"), target.k("block_count")
    if not isinstance(bc_d, int) or not isinstance(bc_t, int):
        return None, None, None, ["block_count missing or not an integer "
                                  "(donor={!r} target={!r})".format(bc_d, bc_t)]

    nextn = donor.k("nextn_predict_layers")
    if nextn is None:
        # Fall back to the arithmetic the block_count relation implies, but say
        # so: writing the model without the key would be a silent no-op.
        implied = bc_d - bc_t
        return None, None, None, [
            "donor has no {}.nextn_predict_layers - without that key the grafted "
            "tensors are dead weight (the model loads and never speculates). "
            "block_count differs by {}, but that is a hint, not the switch; "
            "refusing to guess.".format(donor.arch(), implied)]
    if not isinstance(nextn, int) or not 1 <= nextn <= 8:
        return None, None, None, ["donor's nextn_predict_layers={!r} is not a "
                                  "usable block count".format(nextn)]

    if head_blocks is not None and head_blocks != nextn:
        problems.append("--head-blocks {} contradicts the donor's declared "
                        "nextn_predict_layers {}".format(head_blocks, nextn))

    first, last = bc_d - nextn, bc_d
    if first < 0:
        problems.append("nextn_predict_layers={} exceeds the donor's block_count={}"
                        .format(nextn, bc_d))
    if bc_d != bc_t + nextn:
        problems.append(
            "block_count relation wrong: donor={} target={} nextn={} "
            "(need donor = target + nextn). Equal means the donor carries no "
            "separate head; a bigger gap means these are different depths and the "
            "head would land at the wrong index.".format(bc_d, bc_t, nextn))
    return first, last, nextn, problems


# ===========================================================================
# Compatibility gate
# ===========================================================================

def check_compat(donor, target, first, last, nextn, args):
    """Structural compatibility. `problems` non-empty means: write nothing.

    Returns (problems, notes, head).
    """
    problems, notes = [], []

    # 1./2. architecture
    if not donor.arch():
        problems.append("donor declares no general.architecture")
    elif donor.arch() != target.arch():
        problems.append("architecture mismatch: donor={!r} target={!r} - a head is "
                        "only portable within one architecture"
                        .format(donor.arch(), target.arch()))

    # 9. the head range must actually contain tensors
    head = [t for t in donor.tensors
            if (lambda l: l is not None and first <= l < last)(donor.layer_of(t["name"]))]
    if not head:
        problems.append("donor blocks [{}, {}) hold no `blk.N.*` tensors - nothing "
                        "to graft".format(first, last))
    head_names = {t["name"] for t in head}

    # 4./5. tensor vocabularies.
    # dims MUST match: a shape difference means these are not the same model and
    # the head would not line up. The quant TYPE is deliberately not a
    # requirement - see the module docstring.
    missing, dim_bad, type_diff = [], [], 0
    for t in target.tensors:
        o = donor.by_name.get(t["name"])
        if o is None:
            missing.append(t["name"])
            continue
        if tuple(o["dims"]) != tuple(t["dims"]):
            dim_bad.append((t["name"], t["dims"], o["dims"]))
        elif o["type"] != t["type"]:
            type_diff += 1
    if missing:
        problems.append("{} target tensor(s) absent from the donor, e.g. {} - the "
                        "two checkpoints are not the same base"
                        .format(len(missing), ", ".join(sorted(missing)[:4])))
    if dim_bad:
        problems.append("{} tensor(s) have different SHAPES - these are not the "
                        "same model, e.g. {}".format(
                            len(dim_bad),
                            "; ".join("{} {} vs {}".format(n, list(a), list(b))
                                      for n, a, b in dim_bad[:3])))
    if type_diff:
        notes.append("{} tensor(s) use a different quant type in the two recipes "
                     "(informational - GGUF stores a type per tensor, and two "
                     "recipes of one base legitimately differ)".format(type_diff))

    dnames = {t["name"] for t in target.tensors}
    extra = sorted(n for n in donor.by_name if n not in dnames and n not in head_names)
    if extra:
        problems.append("{} donor tensor(s) are neither in the target nor part of "
                        "the head, e.g. {}".format(len(extra), ", ".join(extra[:4])))

    # 6. refuse to double-add
    already = sorted(n for n in head_names if n in dnames)
    if already:
        problems.append("target ALREADY carries {} tensor(s) inside the head block "
                        "range, e.g. {} - it appears to ship a head of its own; "
                        "refusing to double-add".format(len(already), ", ".join(already[:4])))
    if isinstance(target.k("nextn_predict_layers"), int):
        problems.append("target already declares {}.nextn_predict_layers={} - it is "
                        "not a headless model".format(target.arch(),
                                                      target.k("nextn_predict_layers")))

    # 8. block_count must be a u32 so it can be spliced in place
    bc_key = "{}.block_count".format(target.arch())
    span = target.kv_span.get(bc_key)
    if span is None:
        problems.append("target has no {} - cannot enable the extra layer".format(bc_key))
    else:
        a, b = span
        vlen = (b - a) - 8 - len(bc_key) - 4
        if vlen != 4:
            problems.append("{} is {} bytes wide, not a u32 - it cannot be spliced "
                            "in place".format(bc_key, vlen))
        elif isinstance(target.k("block_count"), int) \
                and target.k("block_count") > 0xFFFFFFFF:
            problems.append("{} is out of u32 range".format(bc_key))

    # 10. every head quant type must be sizeable
    unknown_types = sorted({t["type"] for t in head if t["type"] not in GGML_TYPE_BYTES})
    if unknown_types:
        problems.append("head uses ggml type(s) {} which are not in the size table - "
                        "the layout cannot be computed, and a wrong layout produces a "
                        "file that still parses".format(unknown_types))

    # 11. per-layer array KV audit. Adding blocks makes any array that is exactly
    #     block_count long inconsistent with the new count. Some loaders index it
    #     by layer and would read past the end; others silently use the wrong
    #     entry. Neither is acceptable, so this is a refusal with the fix stated.
    audit = target.per_layer_arrays()
    desync = [(k, n, why) for k, n, why in audit if n == target.k("block_count")]
    if desync:
        problems.append(
            "{} target KV array(s) have one entry per layer and would desync when "
            "blocks are added: {}. Extend each to the new count (or delete it if the "
            "loader can infer the value) before grafting - the keys are listed by "
            "--audit.".format(
                len(desync),
                ", ".join("{} [{}]".format(k, n) for k, n, _ in desync[:4])))
    if audit:
        notes.append("per-layer KV arrays audited: {} candidate(s), {} exactly the "
                     "current block count".format(len(audit), len(desync)))

    # advisory: architecture table
    arch = donor.arch()
    if arch in ARCH_REUSED:
        problems.append("architecture {!r} reuses nextn_predict_layers for something "
                        "other than an MTP head (it is a router layer there), so "
                        "writing it would be semantic pollution, not a graft"
                        .format(arch))
    if arch in ARCH_TOP_LEVEL:
        problems.append("architecture {!r} keeps its head OUTSIDE blk.N.* (top-level "
                        "nextn.pre_projection / post_projection); this writer only "
                        "moves block tensors and would raise block_count without "
                        "adding a head".format(arch))
    if arch in ARCH_INERT:
        notes.append("WARNING: upstream lists {!r} in the group whose head tensors are "
                     "declared and then skipped (TENSOR_SKIP) - the grafted file would "
                     "load and never speculate. Verify against your build before "
                     "trusting it.".format(arch))
    if arch and arch not in ARCH_EXECUTES and arch not in ARCH_INERT \
            and arch not in ARCH_REUSED and arch not in ARCH_TOP_LEVEL:
        notes.append("architecture {!r} is not in the surveyed table - it may or may "
                     "not execute the head. A successful graft is not evidence that it "
                     "does; check the loader log.".format(arch))

    # multi-block head
    if nextn and nextn > 1 and arch not in ARCH_MULTI_BLOCK_OK:
        problems.append("this must add {} head blocks, but upstream asserts "
                        "single-block support for {!r} - loading it would "
                        "GGML_ASSERT".format(nextn, arch))
    elif nextn and nextn > 1:
        notes.append("multi-block head: {} blocks [{}, {}), which {!r} supports"
                     .format(nextn, first, last, arch))

    if not problems:
        notes.append("all {} target tensors matched in the donor (naming + shape)"
                     .format(len(target.tensors)))
        notes.append("the donor's only extra content is the {}-tensor head in "
                     "block range [{}, {})".format(len(head), first, last))
    return problems, notes, head


# ===========================================================================
# Self-check: does the tensor table add up to the file size?
# ===========================================================================

def verify_layout(path):
    """Walk a GGUF and compare the summed tensor table against the real length.

    This is the ONLY check that catches a wrong tensor offset. A file with every
    offset shifted still parses: the header is fine, the table is fine, and the
    tensors simply point at padding or at each other. Summing the table and
    comparing against `filesize - data_start` fails loudly for exactly that class,
    which a "does it load?" test does not.

    Inlined rather than imported. Returns {} if the file is unreadable, else
    {n_tensors, total_bytes, delta_pct, size_ok, layers, head_layers}.
    """
    total = 0
    layer_ids = set()
    try:
        f = open(path, "rb")
    except OSError:
        return {}
    try:
        head = f.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            return {}
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
        if version < 2:
            return {}

        def rd(fmt, size):
            b = f.read(size)
            if len(b) != size:
                raise EOFError
            return struct.unpack("<" + fmt, b)[0]

        for _ in range(n_kv):
            f.seek(rd("Q", 8), 1)
            vtype = rd("I", 4)
            if vtype == 8:
                f.seek(rd("Q", 8), 1)
            elif vtype == 9:
                atype = rd("I", 4)
                alen = rd("Q", 8)
                if atype == 8:
                    for _ in range(alen):
                        f.seek(rd("Q", 8), 1)
                elif atype in SCALAR:
                    f.seek(SCALAR[atype] * alen, 1)
                else:
                    return {}
            elif vtype in SCALAR:
                f.seek(SCALAR[vtype], 1)
            else:
                return {}

        for _ in range(n_tensors):
            nlen = rd("Q", 8)
            name = f.read(nlen).decode("utf-8", "replace")
            nd = rd("I", 4)
            n_elem = 1
            for _ in range(nd):
                n_elem *= rd("Q", 8)
            ttype = rd("I", 4)
            f.seek(8, 1)                                   # data offset
            bpe = GGML_TYPE_BYTES.get(ttype)
            if bpe is None:
                return {}                                  # unknown quant: no estimate
            total += int(n_elem * bpe)
            if name.startswith("blk."):
                rest = name[4:]
                dot = rest.find(".")
                if dot > 0:
                    try:
                        layer_ids.add(int(rest[:dot]))
                    except ValueError:
                        pass

        data_off = align(f.tell(), 32)
        size = os.path.getsize(path)
        real = size - data_off
        delta = abs(total - real) / real * 100 if real else 100.0
        return {
            "n_tensors": n_tensors,
            "total_bytes": total,
            "delta_pct": delta,
            "size_ok": delta < 1.0,
            "layers": sorted(layer_ids),
        }
    except (EOFError, struct.error, OSError):
        return {}
    finally:
        f.close()


# ===========================================================================
# Writer
# ===========================================================================

def write_graft(donor, target, first, last, nextn, head, out):
    """Write donor's head blocks onto target, producing `out`.

    Order matters and two earlier versions of this got it wrong, so:

      * the KV block is copied from the TARGET. Reading the donor's KV bytes
        while writing the target's n_kv made the parser stop early and read the
        rest of the KV as a tensor table; every offset after that was garbage.
      * the head data starts at `align(target_data_len)` - NOT at
        `target_data_len + align(head_bytes)`. Writing the whole head region as
        padding first shifted every head offset by ~0.49 GiB and pointed the head
        tensors at zeros. The file still parsed.
    """
    bc_key = "{}.block_count".format(target.arch())
    npl_key = "{}.nextn_predict_layers".format(target.arch())

    with open(target.path, "rb") as f:
        f.seek(24)
        kv_raw = bytearray(f.read(target.kv_end - 24))

    # 1. block_count: in place, same width.
    a, b = target.kv_span[bc_key]
    kv_raw[b - 24 - 4:b - 24] = struct.pack("<I", target.k("block_count") + nextn)
    say("  {} {} -> {}".format(bc_key, target.k("block_count"),
                               target.k("block_count") + nextn))

    # 2. THE SWITCH. Absent from the target by construction (gate 6), so it is
    #    appended. Without it the grafted tensors are dead weight.
    added = 0
    if npl_key not in target.kv:
        kv_raw += enc_kv_u32(npl_key, nextn)
        added = 1
        say("  + {} = {}   (from donor - this is what enables the head)"
            .format(npl_key, nextn))
    n_kv_out = target.n_kv + added

    # 3. tensor table: the target's rows verbatim, then the head's rows with
    #    offsets recomputed against the new data section.
    with open(target.path, "rb") as f:
        f.seek(target.kv_end)
        d_table = f.read(target.tensor_end - target.kv_end)
    existing_data_len = target.size - target.data_start

    new_entries = b""
    cursor = align(existing_data_len, donor.alignment)
    placed = []
    for t in head:
        sz = donor.tsize(t)
        new_entries += (enc_str(t["name"])
                        + struct.pack("<I", len(t["dims"]))
                        + b"".join(struct.pack("<Q", x) for x in t["dims"])
                        + struct.pack("<I", t["type"])
                        + struct.pack("<Q", cursor))
        placed.append((t, cursor, sz))
        cursor = align(cursor + sz, donor.alignment)

    header = struct.pack("<4sIQQ", b"GGUF", target.version,
                         target.n_tensors + len(head), n_kv_out)
    body_table = d_table + new_entries
    tensor_end_new = 24 + len(kv_raw) + len(body_table)
    data_start_new = align(tensor_end_new, target.alignment)

    say("\nwriting {} ...".format(out))
    with open(out, "wb") as w:
        w.write(header)
        w.write(kv_raw)
        w.write(body_table)
        w.write(b"\x00" * (data_start_new - tensor_end_new))
        with open(target.path, "rb") as f:                 # target data verbatim
            f.seek(target.data_start)
            remaining = existing_data_len
            while remaining > 0:
                chunk = f.read(min(1 << 24, remaining))
                if not chunk:
                    break
                w.write(chunk)
                remaining -= len(chunk)
        # Pad only to the next alignment boundary.
        w.write(b"\x00" * (align(existing_data_len, donor.alignment)
                           - existing_data_len))
        with open(donor.path, "rb") as f:                  # head data
            for t, _off, sz in placed:
                f.seek(donor.data_start + t["offset"])
                got = f.read(sz)
                if len(got) != sz:
                    raise GgufError("short read for {}".format(t["name"]))
                w.write(got)
                w.write(b"\x00" * (align(len(got), donor.alignment) - len(got)))
    return out.stat().st_size


# ===========================================================================
# CLI
# ===========================================================================

def pick_main_gguf(directory):
    """The single main gguf in a model directory, or None.

    Mirrors what the model scanner does: one main gguf per directory, mmproj
    excluded. Passing a directory with two mains would make the two disagree.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    cands = sorted(p for p in d.glob("*.gguf") if "mmproj" not in p.name.lower())
    return cands[0] if cands else None


def resolve_pair(args):
    """(donor_path, target_path) from either explicit files or a preset + dir."""
    if args.donor and args.target:
        return Path(args.donor), Path(args.target)

    preset = PRESETS.get(args.preset) if args.preset else None
    if preset is None:
        return None, None
    models = args.models_dir or os.environ.get("CHAT_DIR")
    if not models:
        return None, None
    dd = Path(models) / (args.donor or preset["donor_dir"])
    td = Path(models) / (args.target or preset["target_dir"])
    return pick_main_gguf(dd), pick_main_gguf(td)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report only (this is the default; nothing is written)")
    mode.add_argument("--go", action="store_true", help="write the grafted file")
    mode.add_argument("--audit", action="store_true",
                      help="print every per-layer KV array the target declares, then exit")
    ap.add_argument("--donor", default=None,
                    help="GGUF that carries the head (or a directory name with --preset)")
    ap.add_argument("--target", default=None,
                    help="GGUF to graft onto (or a directory name with --preset)")
    ap.add_argument("--models-dir", default=None,
                    help="root for --preset directory lookups (else $CHAT_DIR)")
    ap.add_argument("--out", default=None, help="output path (required with --go)")
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="fill donor/target from a documented example")
    ap.add_argument("--head-blocks", type=int, default=None,
                    help="assert the number of head blocks (default: use the "
                         "donor's nextn_predict_layers)")
    ap.add_argument("--force", action="store_true",
                    help="write even when the space check fails (never bypasses "
                         "the structural gate)")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    global QUIET
    QUIET = args.quiet
    if args.go and not args.out:
        ap.error("--go requires --out <path> (the target is never written in place)")

    donor_p, target_p = resolve_pair(args)
    if not donor_p or not target_p:
        ap.error("give --donor and --target, or --preset with --models-dir/$CHAT_DIR")
    for p in (donor_p, target_p):
        if not Path(p).is_file():
            ap.error("not found: {}".format(p))

    try:
        donor, target = Gguf(donor_p), Gguf(target_p)
    except GgufError as e:
        print("[X] {}".format(e))
        return 2

    say("donor : {}\n        {}".format(donor_p.name, donor_p))
    say("target: {}\n        {}".format(target_p.name, target_p))
    say("  arch {} | block_count donor={} target={} | tensors {} / {}"
        .format(donor.arch() or "?", donor.k("block_count"), target.k("block_count"),
                donor.n_tensors, target.n_tensors))

    if args.audit:
        rows = target.per_layer_arrays()
        say("\nper-layer KV arrays in the target ({} candidate(s)):".format(len(rows)))
        for key, n, why in rows:
            say("  {:<44} len={:<6} [{}]  {}".format(
                key, n, "DESYNC" if n == target.k("block_count") else "ok", why))
        return 0

    first, last, nextn, problems = head_plan(donor, target, args.head_blocks)
    if problems:
        for p in problems:
            print("  [X] {}".format(p))
        print("\n[X] REFUSING TO GRAFT: the head cannot be located. Nothing was "
              "written.")
        return 1

    say("\nhead = donor blocks [{}, {})  nextn={}".format(first, last, nextn))
    head_probe = [t for t in donor.tensors
                  if (lambda l: l is not None and first <= l < last)(donor.layer_of(t["name"]))]
    head_bytes = sum(donor.tsize(t) for t in head_probe)
    say("  tensors={}  bytes={} ({:.2f} GiB)"
        .format(len(head_probe), human(head_bytes), head_bytes / (1 << 30)))
    if not QUIET:
        for t in head_probe:
            say("    {:<44} dims={:<2} type={:<3} {:>10}".format(
                t["name"], len(t["dims"]), t["type"], human(donor.tsize(t))))

    problems, notes, head = check_compat(donor, target, first, last, nextn, args)
    say("\ncompatibility:")
    for n in notes:
        say("    . {}".format(n))
    if problems:
        for p in problems:
            print("  [X] {}".format(p))
        print("\n[X] REFUSING TO GRAFT: {} compatibility problem(s). Nothing was "
              "written; there is nothing to clean up.".format(len(problems)))
        return 1
    say("  [OK] donor and target line up structurally")

    need = target.size + head_bytes + (1 << 26)
    free = shutil.disk_usage(str(Path(target_p).parent)).free
    say("\nspace: need ~{} , free {} -> {}".format(
        human(need), human(free), "OK" if free > need else "NOT ENOUGH"))
    if free <= need and not args.force:
        print("  [X] not enough free space (use --force only if you know the "
              "estimate is pessimistic)")
        return 1

    if args.json:
        print(json.dumps({
            "donor": str(donor_p), "target": str(target_p),
            "arch": donor.arch(), "head_blocks": nextn,
            "head_range": [first, last], "head_tensors": len(head),
            "head_bytes": head_bytes, "problem_count": 0, "notes": notes,
        }, indent=2))

    if not args.go:
        say("\n[check only] pass --go --out <path> to write")
        return 0

    out = Path(args.out)
    if out.exists():
        print("[X] {} already exists - refusing to overwrite".format(out))
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)

    size = write_graft(donor, target, first, last, nextn, head, out)
    say("done: {} ({})".format(out, human(size)))

    v = verify_layout(str(out))
    if not v or not v.get("size_ok"):
        print("[X] SELF-CHECK FAILED: the tensor table does not add up to the file "
              "size (delta={}%). The artifact is not internally consistent - do NOT "
              "use it, delete it and re-run."
              .format(v.get("delta_pct") if v else "n/a"))
        return 1
    say("[OK] self-check: {} tensors, delta {:.4f}%, blocks up to blk.{}"
        .format(v["n_tensors"], v["delta_pct"], max(v["layers"]) if v["layers"] else -1))
    say("\nNEXT - verify all three, they catch different faults:")
    say("  1. load it and confirm the loader reports {} blocks, not {}"
        .format(target.k("block_count") + nextn, target.k("block_count")))
    say("  2. run with --spec-type draft-mtp and check `draft acceptance` is "
        "non-null and > 0.3")
    say("  3. A/B it against the un-grafted model IN THE SAME SESSION")
    say("  (remember: the MTP path costs ~3x the head's weight bytes in VRAM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
