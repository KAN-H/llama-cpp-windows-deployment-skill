#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q6 Phase-0 metadata probe: Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.

Answers, before any GPU time is spent:
  1. Is the MTP head really there?       -> nextn_predict_layers KV + blk.<N>.* tensors
  2. What geometry do we budget on?      -> block_count, head_count_kv, key/value len,
                                            sliding_window / full_attention_interval
  3. Is the file layout sane?            -> sum(tensor table) vs (file size - data offset)
     The graft-era lesson: "actual size != expected size" is the earliest and most
     reliable signal that a layout assumption is broken; parse success is NOT proof
     (a mis-offset file still parses). Check it first, trust it most.

Also dumps the per-layer byte curve and the quant-type histogram: GSQ-RCO assigns a
quant type PER TENSOR, so "IQ3_S" is a budget label, not a uniform type - the real
mix matters for CUDA kernel coverage and for the per-layer VRAM math.

Usage (script in <llama-cpp-dir>/plan/, or anywhere with LLAMA_DIR set):
    .\\.venv\\Scripts\\python.exe plan\\_q6_metadata.py [model_dir]

model_dir defaults to <chat root>/<DIRNAME>; the chat root is CHAT_DIR if
set, else the chat_dir the launcher registry remembers (registry found via
LAUNCHER_DIR, default <LLAMA_DIR>/launcher).

Writes _q6_metadata.json next to this script and prints a summary.
"""
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


_CHAT = _chat_root()
DEFAULT_DIR = (_CHAT / DIRNAME) if _CHAT else Path("<models-dir>") / DIRNAME

# GGML tensor type -> bytes per block (kept in sync with
# launcher/update_launchers.py GGML_TYPE_BYTES; the size cross-check below is the
# independent verification that this table is right for this file).
TYPE_BYTES = {
    0: 4.0, 1: 2.0, 2: 18 / 32, 3: 20 / 32, 6: 22 / 32, 7: 24 / 32,
    8: 34 / 32, 9: 36 / 32, 10: 84 / 256, 11: 110 / 256, 12: 144 / 256,
    13: 176 / 256, 14: 210 / 256, 15: 292 / 256, 16: 66 / 256, 17: 74 / 256,
    18: 98 / 256, 19: 50 / 256, 20: 18 / 32, 21: 110 / 256, 22: 82 / 256,
    23: 136 / 256, 24: 1.0, 25: 2.0, 26: 4.0, 27: 8.0, 28: 8.0, 29: 56 / 256,
    30: 2.0, 34: 54 / 256, 35: 66 / 256, 39: 17 / 32,
}
TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}
SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
          5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
          12: ("d", 8)}


def scan(path):
    """Walk a GGUF: all scalar KVs (strings truncated), then the tensor table."""
    out = {"file": str(path), "size": path.stat().st_size, "kv": {}, "kv_arrays": {},
           "tensors": None, "error": None}
    try:
        f = open(path, "rb")
    except OSError as e:
        out["error"] = "open: %s" % e
        return out
    with f:
        head = f.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            out["error"] = "not a GGUF file"
            return out
        ver, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
        out["version"] = ver
        out["n_tensors_declared"] = n_tensors
        out["n_kv"] = n_kv

        def rd(fmt, size):
            b = f.read(size)
            if len(b) != size:
                raise EOFError
            return struct.unpack("<" + fmt, b)[0]

        for _ in range(n_kv):
            klen = rd("Q", 8)
            key = f.read(klen).decode("utf-8", "replace")
            vtype = rd("I", 4)
            if vtype == 8:
                slen = rd("Q", 8)
                raw = f.read(slen)
                out["kv"][key] = raw.decode("utf-8", "replace")[:300] if slen <= 4096 \
                    else "<string %d bytes>" % slen
            elif vtype == 9:
                atype = rd("I", 4)
                alen = rd("Q", 8)
                out["kv_arrays"][key] = {"type": atype, "len": alen}
                if atype == 8:
                    for _ in range(alen):
                        f.seek(rd("Q", 8), 1)
                elif atype in SCALAR:
                    f.seek(SCALAR[atype][1] * alen, 1)
                else:
                    raise ValueError("array type %d" % atype)
            elif vtype in SCALAR:
                fmt, size = SCALAR[vtype]
                out["kv"][key] = rd(fmt, size)
            else:
                raise ValueError("value type %d" % vtype)

        tensors = []
        for _ in range(n_tensors):
            nlen = rd("Q", 8)
            name = f.read(nlen).decode("utf-8", "replace")
            nd = rd("I", 4)
            dims = [rd("Q", 8) for _ in range(nd)]
            ttype = rd("I", 4)
            _off = rd("Q", 8)
            seen = 1
            for d in dims:
                seen *= d
            bpe = TYPE_BYTES.get(ttype)
            nbytes = seen * bpe if bpe is not None else None
            tensors.append({"name": name, "dims": dims, "type": ttype,
                            "bytes": nbytes})
        data_off = (f.tell() + 31) // 32 * 32
        out["tensors"] = tensors
        out["data_offset"] = data_off
        total = sum(t["bytes"] for t in tensors if t["bytes"] is not None)
        real = out["size"] - data_off
        out["tensor_bytes_sum"] = total
        out["delta_pct"] = abs(total - real) / real * 100 if real else 100.0
        out["size_ok"] = out["delta_pct"] < 1.0
    return out


def analyze(scan_result):
    kv = scan_result["kv"]
    a = {}
    arch = kv.get("general.architecture", "")
    a["arch"] = arch

    def g(suffix, default=None):
        # llama.cpp has shipped two spellings over time; the GSQ-RCO file uses
        # qwen35.full_attention_interval (no 'attention.' segment), the older
        # UD-era files used qwen35.attention.sliding_window. Accept both.
        for k in ("%s.%s" % (arch, suffix),
                  "%s.attention.%s" % (arch, suffix),
                  "general." + suffix):
            if k in kv:
                return kv[k]
        return default

    a["name"] = kv.get("general.name", "")
    a["block_count"] = g("block_count")
    a["nextn_predict_layers"] = g("nextn_predict_layers")
    a["context_length"] = g("context_length")
    a["embedding_length"] = g("embedding_length")
    a["head_count"] = g("attention.head_count", kv.get("%s.head_count" % arch))
    a["head_count_kv"] = g("attention.head_count_kv", kv.get("%s.head_count_kv" % arch))
    a["key_length"] = g("attention.key_length", kv.get("%s.key_length" % arch))
    a["value_length"] = g("attention.value_length", kv.get("%s.value_length" % arch))
    a["sliding_window"] = g("sliding_window")
    a["sliding_window_pattern"] = g("sliding_window_pattern")
    a["full_attention_interval"] = g("full_attention_interval")
    a["rope_freq_base"] = g("rope.freq_base")
    a["expert_count"] = g("expert_count")
    a["vocab_size"] = kv.get("tokenizer.ggml.tokens.count")
    a["quantization_version"] = kv.get("general.quantization_version")
    a["file_type"] = kv.get("general.file_type")

    tensors = scan_result["tensors"] or []
    per_layer = {}
    types = {}
    mtp_tensors = []
    blk_ids = set()
    n_nextn = a["nextn_predict_layers"] or 0
    mtp_start = (a["block_count"] - n_nextn) if (n_nextn and a["block_count"]) else None
    for t in tensors:
        name, nb = t["name"], t["bytes"]
        tn = TYPE_NAME.get(t["type"], "type%d" % t["type"])
        types[tn] = types.get(tn, 0) + (nb or 0)
        if name.startswith("blk."):
            try:
                lay = int(name.split(".", 2)[1])
                blk_ids.add(lay)
                per_layer[lay] = per_layer.get(lay, 0) + (nb or 0)
            except ValueError:
                pass
            if mtp_start is not None:
                try:
                    if int(name.split(".", 2)[1]) >= mtp_start:
                        mtp_tensors.append({"name": name, "type": tn, "bytes": nb})
                except (ValueError, IndexError):
                    pass
        elif "nextn" in name or "mtp" in name.lower():
            mtp_tensors.append({"name": name, "type": tn, "bytes": nb})

    a["n_tensors"] = len(tensors)
    a["unknown_type_tensors"] = sum(1 for t in tensors if t["bytes"] is None)
    a["blk_max"] = max(blk_ids) if blk_ids else None
    a["blk_count_seen"] = len(blk_ids)
    a["per_layer_mib"] = {k: round(v / (1 << 20), 1) for k, v in sorted(per_layer.items())}
    # layers at/above mtp_start are the MTP head block(s)
    main_cut = mtp_start if mtp_start is not None else (a["block_count"] or 0)
    main_layers = [v for k, v in per_layer.items() if k < main_cut]
    a["main_layer_mib_mean"] = round(sum(main_layers) / len(main_layers) / (1 << 20), 1) if main_layers else None
    a["main_layer_mib_min"] = round(min(main_layers) / (1 << 20), 1) if main_layers else None
    a["main_layer_mib_max"] = round(max(main_layers) / (1 << 20), 1) if main_layers else None
    a["mtp_tensor_count"] = len(mtp_tensors)
    a["mtp_tensor_bytes_mib"] = round(sum(t["bytes"] or 0 for t in mtp_tensors) / (1 << 20), 1)
    a["mtp_tensors"] = mtp_tensors
    gb = 1 << 30
    a["weight_gib_total"] = round(scan_result["tensor_bytes_sum"] / gb, 3)
    a["type_histogram_gib"] = {k: round(v / gb, 3) for k, v in
                               sorted(types.items(), key=lambda x: -x[1])}
    return a


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not model_dir.is_dir():
        print("[X] model dir not found: %s" % model_dir)
        if _CHAT is None:
            print("    set CHAT_DIR (models root), or LLAMA_DIR so the launcher "
                  "registry's chat_dir can be read")
        return 2
    report = {"model_dir": str(model_dir), "files": {}}
    for f in sorted(model_dir.iterdir()):
        if f.suffix.lower() != ".gguf":
            continue
        print("== scanning %s (%.2f GiB) ==" % (f.name, f.stat().st_size / (1 << 30)))
        try:
            s = scan(f)
        except Exception as e:  # noqa: BLE001 - report, do not crash the probe
            print("   FAILED: %s" % e)
            report["files"][f.name] = {"error": str(e)}
            continue
        if s["error"]:
            print("   FAILED: %s" % s["error"])
            report["files"][f.name] = {"error": s["error"]}
            continue
        a = analyze(s)
        report["files"][f.name] = {"scan": {k: v for k, v in s.items()
                                            if k not in ("tensors", "kv_arrays")},
                                   "analysis": a}

        print("   version=%s  n_tensors=%d  n_kv=%d" %
              (s.get("version"), s.get("n_tensors_declared"), s.get("n_kv")))
        print("   size check: sum=%d bytes  real=%d bytes  delta=%.3f%%  -> %s" %
              (s["tensor_bytes_sum"], s["size"] - s["data_offset"], s["delta_pct"],
               "OK" if s["size_ok"] else "** MISMATCH **"))
        if a["arch"]:
            print("   arch=%s  block_count=%s  nextn_predict_layers=%s  ctx_train=%s" %
                  (a["arch"], a["block_count"], a["nextn_predict_layers"],
                   a["context_length"]))
            print("   kv heads=%s key=%s value=%s  sliding_window=%s  interval=%s  vocab=%s" %
                  (a["head_count_kv"], a["key_length"], a["value_length"],
                   a["sliding_window"], a["full_attention_interval"], a["vocab_size"]))
            print("   weights=%.3f GiB  layers seen=%s (max blk=%s)  mean layer=%.1f MiB" %
                  (a["weight_gib_total"], a["blk_count_seen"], a["blk_max"],
                   a["main_layer_mib_mean"] or 0))
            print("   MTP: %d tensors, %.1f MiB %s" %
                  (a["mtp_tensor_count"], a["mtp_tensor_bytes_mib"],
                   ("(nextn_predict_layers=%s)" % a["nextn_predict_layers"])
                   if a["nextn_predict_layers"] else "** NONE FOUND **"))
            for t in a["mtp_tensors"][:20]:
                print("      %s  [%s]  %s B" % (t["name"], t["type"], t["bytes"]))
        print("   type mix (GiB): %s" % a["type_histogram_gib"])
        print()

    out = HERE / "_q6_metadata.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("JSON written to %s" % out)


if __name__ == "__main__":
    sys.exit(main())
