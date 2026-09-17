# -*- coding: utf-8 -*-
"""Tests for scripts/mtp_graft.py.

Two things are being tested, and they are different in kind:

  * the POSITIVE path, which must accept a pair that lines up - a gate that
    refuses everything is not a gate, it is a wall.
  * the NEGATIVE paths, one per gate. Each fixture is a real, parseable GGUF, so
    the refusal has to come from the check and not from a parse error. That
    distinction matters: "it crashed" is not "it refused".

The fixtures are synthetic and tiny (a few hundred bytes), so this runs anywhere
in under a second - unlike the 22 GiB models the tool is actually for. The
synthetic positive case exists for the same reason: the real pair proved the
layout arithmetic (see the Stage 2 report), but it cannot be a unit test.

    python scripts/tests/test_mtp_graft.py
"""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "mtp_graft.py"

# ggml type ids used by the fixtures; see GGML_TYPE_BYTES in the tool.
F32, Q8_0, Q4_1 = 0, 8, 3

ALIGN = 32


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


def _enc_kv_str(key, val):
    kb = key.encode()
    vb = val.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + \
        struct.pack("<Q", len(vb)) + vb


def _enc_kv_u32(key, val):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", val)


def _enc_kv_array_u32(key, vals):
    kb = key.encode()
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9)
            + struct.pack("<I", 4) + struct.pack("<Q", len(vals))
            + b"".join(struct.pack("<I", v) for v in vals))


def build_gguf(path, tensors, kv):
    """Write a minimal but genuinely valid GGUF.

    `tensors` is [(name, dims, ggml_type)]. `kv` is a list of already-encoded
    key-value blobs, so a test can emit a key of any width - including a
    deliberately wrong one.
    """
    kv_bytes = b"".join(kv)
    n_tensors = len(tensors)

    # Sizes are computed with the same table the tool uses, so the file passes
    # the tool's own size cross-check when the fixture is meant to be valid.
    # An UNKNOWN type falls back to a nominal 2 B/weight: the fixture still has
    # to be laid out, and the point of that case is that the TOOL refuses it.
    sys.path.insert(0, str(HERE.parent))
    import mtp_graft as MG
    sizes = []
    for _name, dims, ttype in tensors:
        n = 1
        for d in dims:
            n *= d
        sizes.append(int(n * MG.GGML_TYPE_BYTES.get(ttype, 2.0)))

    table = b""
    offsets = []
    cur = 0
    for (name, dims, ttype), sz in zip(tensors, sizes):
        nb = name.encode()
        table += (struct.pack("<Q", len(nb)) + nb
                  + struct.pack("<I", len(dims))
                  + b"".join(struct.pack("<Q", d) for d in dims)
                  + struct.pack("<I", ttype)
                  + struct.pack("<Q", cur))
        offsets.append(cur)
        cur = align(cur + sz)

    body = struct.pack("<4sIQQ", b"GGUF", 3, n_tensors, len(kv))
    body += kv_bytes + table
    pad = align(len(body)) - len(body)
    data = b"".join(b"\x11" * sz + b"\x00" * (align(sz) - sz) for sz in sizes)

    Path(path).write_bytes(body + b"\x00" * pad + data)
    return path


def std_kv(arch="qwen35moe", block_count=4, nextn=None, extra=None):
    kv = [
        _enc_kv_str("general.architecture", arch),
        _enc_kv_u32("{}.block_count".format(arch), block_count),
        _enc_kv_u32("general.alignment", ALIGN),
    ]
    if nextn is not None:
        kv.append(_enc_kv_u32("{}.nextn_predict_layers".format(arch), nextn))
    if extra:
        kv.extend(extra)
    return kv


def base_tensors(blocks=4, arch_prefix=""):
    """Layers 0..blocks-1 with one weight each, plus the usual globals."""
    t = [("token_embd.weight", [8, 8], Q8_0),
         ("output_norm.weight", [8], F32),
         ("output.weight", [8, 8], Q8_0)]
    for i in range(blocks):
        t.append(("blk.{}.attn.weight".format(i), [8, 8], Q8_0))
    return t


def make_pair(tmp, donor_over=None, target_over=None):
    """A generative donor/target pair that the tool accepts, plus overrides."""
    d_tensors = base_tensors(4) + [("blk.4.nextn.eh_proj.weight", [8, 8], Q8_0)]
    d_kv = std_kv(block_count=5, nextn=1)
    t_tensors = list(base_tensors(4))
    t_kv = std_kv(block_count=4)

    # (donor_tensors, donor_kv, target_tensors, target_kv) overrides
    if donor_over:
        d_tensors = donor_over.get("tensors", d_tensors)
        d_kv = donor_over.get("kv", d_kv)
    if target_over:
        t_tensors = target_over.get("tensors", t_tensors)
        t_kv = target_over.get("kv", t_kv)

    d = tmp / "donor.gguf"
    t = tmp / "target.gguf"
    build_gguf(d, d_tensors, d_kv)
    build_gguf(t, t_tensors, t_kv)
    return d, t


def run_tool(donor, target, *extra):
    cmd = [sys.executable, str(TOOL), "--check", "--donor", str(donor),
           "--target", str(target), "--quiet"] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(HERE.parent), timeout=120)
    return p.returncode, p.stdout + p.stderr


def run_tool_verbose(donor, target):
    """Same, but with the report on - the positive case asserts on its wording."""
    cmd = [sys.executable, str(TOOL), "--check", "--donor", str(donor),
           "--target", str(target)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(HERE.parent), timeout=120)
    return p.returncode, p.stdout + p.stderr


def expect_refuse(name, donor, target, needle, extra=()):
    rc, out = run_tool(donor, target, *extra)
    ok = rc == 1 and needle.lower() in out.lower()
    detail = ""
    if rc == 0:
        detail = "tool ACCEPTED (rc=0) - the gate did not fire"
    elif rc not in (0, 1):
        detail = "expected a clean refusal (rc=1), got rc={}".format(rc)
    elif needle.lower() not in out.lower():
        detail = "refused, but not for the expected reason (looking for {!r})".format(needle)
    return ok, detail


def main():
    failures = []
    cases = []

    with tempfile.TemporaryDirectory(prefix="mtp-graft-tests-") as td:
        tmp = Path(td)

        # ---------------------------------------------------------------- positive
        d, t = make_pair(tmp)
        rc, out = run_tool_verbose(d, t)
        ok = rc == 0 and "donor and target line up structurally" in out
        cases.append(("positive: a pair that lines up is ACCEPTED", ok,
                      "" if ok else "rc={} out={}".format(rc, out[-400:])))
        if not ok:
            failures.append("positive")

        # ------------------------------------------------- negative, one per gate
        # 2/3/9: architecture
        dd = tmp / "d_arch.gguf"
        build_gguf(dd, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], Q8_0)],
                   std_kv(arch="qwen35", block_count=5, nextn=1))
        ok, why = expect_refuse("arch mismatch", dd, t, "architecture mismatch")
        cases.append(("arch mismatch is refused", ok, why))
        ok or failures.append("arch")

        # 3: block_count relation (donor not target+nextn)
        dd = tmp / "d_bc.gguf"
        build_gguf(dd, base_tensors(4) + [("blk.3.nextn.x.weight", [8, 8], Q8_0)],
                   std_kv(block_count=4, nextn=1))
        ok, why = expect_refuse("block_count relation", dd, t, "block_count relation")
        cases.append(("block_count relation wrong is refused", ok, why))
        ok or failures.append("block_count")

        # 4: a target tensor absent from the donor
        d2, t2 = make_pair(tmp, target_over={
            "tensors": list(base_tensors(4)) + [("blk.1.only_in_target.weight", [8, 8], Q8_0)]})
        ok, why = expect_refuse("missing target tensor", d2, t2, "absent from the donor")
        cases.append(("a target tensor missing from the donor is refused", ok, why))
        ok or failures.append("missing")

        # 4b: shape mismatch
        d3, t3 = make_pair(tmp, target_over={
            "tensors": [("token_embd.weight", [8, 8], Q8_0),
                        ("output_norm.weight", [8], F32),
                        ("output.weight", [8, 8], Q8_0),
                        ("blk.0.attn.weight", [8, 16], Q8_0)] +
                       [("blk.{}.attn.weight".format(i), [8, 8], Q8_0) for i in (1, 2, 3)]})
        ok, why = expect_refuse("shape mismatch", d3, t3, "different SHAPES")
        cases.append(("a shape mismatch is refused", ok, why))
        ok or failures.append("shape")

        # 5: donor extra tensor outside the head range
        d4, t4 = make_pair(tmp)
        build_gguf(d4, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], Q8_0),
                                          ("stray.extra.weight", [8, 8], Q8_0)],
                   std_kv(block_count=5, nextn=1))
        ok, why = expect_refuse("stray donor tensor", d4, t4, "neither in the target nor part of")
        cases.append(("a donor tensor outside the head is refused", ok, why))
        ok or failures.append("stray")

        # 6: target already carries a head. The tensor must have the SAME name
        #    as the donor's head tensor - otherwise gate 4 (a target tensor the
        #    donor lacks) fires first, and the refusal is correct but for the
        #    wrong reason, which would make this case prove nothing.
        d5, _t5 = make_pair(tmp)
        t5 = tmp / "t_head.gguf"
        build_gguf(t5, base_tensors(4) + [("blk.4.nextn.eh_proj.weight", [8, 8], Q8_0)],
                   std_kv(block_count=4))
        ok, why = expect_refuse("target already has a head", d5, t5, "ALREADY carries")
        cases.append(("a target that already has a head is refused", ok, why))
        ok or failures.append("double-add")

        # 6b: target declares nextn_predict_layers
        t6 = tmp / "t_npl.gguf"
        build_gguf(t6, base_tensors(4), std_kv(block_count=4, nextn=1))
        ok, why = expect_refuse("target declares nextn", d5, t6, "not a headless model")
        cases.append(("a target declaring nextn_predict_layers is refused", ok, why))
        ok or failures.append("target-nextn")

        # 7: donor has no nextn_predict_layers
        d7 = tmp / "d_nonpl.gguf"
        build_gguf(d7, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], Q8_0)],
                   std_kv(block_count=5, nextn=None))
        ok, why = expect_refuse("donor lacks the switch", d7, t, "no qwen35moe.nextn_predict_layers")
        cases.append(("a donor with no nextn_predict_layers is refused", ok, why))
        ok or failures.append("no-switch")

        # 8: block_count is not a u32
        kv_bad_width = [
            _enc_kv_str("general.architecture", "qwen35moe"),
            struct.pack("<Q", len(b"qwen35moe.block_count")) + b"qwen35moe.block_count"
            + struct.pack("<I", 10) + struct.pack("<Q", 4),   # u64, not u32
            _enc_kv_u32("general.alignment", ALIGN),
        ]
        t8 = tmp / "t_u64.gguf"
        build_gguf(t8, base_tensors(4), kv_bad_width)
        ok, why = expect_refuse("block_count not u32", d5, t8, "not a u32")
        cases.append(("a non-u32 block_count is refused", ok, why))
        ok or failures.append("u32")

        # 10: an unknown quant type in the head
        sys.path.insert(0, str(HERE.parent))
        import mtp_graft as MG
        unknown = max(MG.GGML_TYPE_BYTES) + 100
        d10 = tmp / "d_badtype.gguf"
        build_gguf(d10, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], unknown)],
                   std_kv(block_count=5, nextn=1))
        ok, why = expect_refuse("unknown quant type", d10, t, "not in the size table")
        cases.append(("an unknown quant type in the head is refused", ok, why))
        ok or failures.append("quant")

        # advisory: granite-switch reuses the key
        for arch, needle, label in (
                ("granite-switch", "router layer", "granite-switch (reused key)"),
                ("gemma4-assistant", "OUTSIDE blk", "gemma4-assistant (top-level head)")):
            dd = tmp / "d_{}.gguf".format(arch)
            build_gguf(dd, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], Q8_0)],
                       std_kv(arch=arch, block_count=5, nextn=1))
            tt = tmp / "t_{}.gguf".format(arch)
            build_gguf(tt, base_tensors(4), std_kv(arch=arch, block_count=4))
            ok, why = expect_refuse(label, dd, tt, needle)
            cases.append(("{} is refused".format(label), ok, why))
            ok or failures.append(arch)

        # advisory: multi-block head on a single-block-asserting arch
        d13 = tmp / "d_multi.gguf"
        build_gguf(d13, base_tensors(4) + [("blk.4.nextn.x.weight", [8, 8], Q8_0),
                                           ("blk.5.nextn.y.weight", [8, 8], Q8_0)],
                   std_kv(block_count=6, nextn=2))
        t13 = tmp / "t_6blk.gguf"
        build_gguf(t13, base_tensors(4), std_kv(block_count=4))
        ok, why = expect_refuse("multi-block on single-block arch", d13, t13,
                                "asserts")
        cases.append(("a multi-block head on a qwen35moe target is refused", ok, why))
        ok or failures.append("multi")

        # 11: a per-layer array KV would desync
        d14, _ = make_pair(tmp)
        t14 = tmp / "t_desync.gguf"
        build_gguf(t14, base_tensors(4),
                   std_kv(block_count=4, extra=[_enc_kv_array_u32(
                       "qwen35moe.compress_ratios", [1, 1, 1, 1])]))
        ok, why = expect_refuse("per-layer KV desync", d14, t14, "desync")
        cases.append(("a per-layer KV array that would desync is refused", ok, why))
        ok or failures.append("desync")

        # --go without --out
        rc, out = run_tool(d, t, "--go")
        ok = rc == 2 and "--out" in out
        cases.append(("--go without --out exits 2", ok, "" if ok else "rc={}".format(rc)))
        ok or failures.append("no-out")

        # --go must not overwrite an existing file
        existing = tmp / "already.gguf"
        existing.write_bytes(b"do not clobber me")
        p = subprocess.run([sys.executable, str(TOOL), "--go", "--donor", str(d),
                            "--target", str(t), "--out", str(existing), "--quiet"],
                           capture_output=True, text=True, cwd=str(HERE.parent))
        ok = p.returncode == 1 and existing.read_bytes() == b"do not clobber me"
        cases.append(("--go refuses to overwrite an existing output", ok,
                      "" if ok else "rc={} content={!r}".format(p.returncode, existing.read_bytes()[:20])))
        ok or failures.append("overwrite")

    # ------------------------------------------------------------------ report
    width = max(len(c[0]) for c in cases)
    print()
    print("=" * (width + 10))
    print("mtp_graft.py gate tests  ({}/{} passed)".format(
        sum(1 for c in cases if c[1]), len(cases)))
    print("=" * (width + 10))
    for name, ok, why in cases:
        print("  {} {}".format("PASS" if ok else "FAIL", name))
        if not ok and why:
            print("       {}".format(why))
    print()
    if failures:
        print("FAILED: {}".format(", ".join(failures)))
        return 1
    print("all gates behaved as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
