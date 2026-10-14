#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_refactor2_body_probe.py - prove a generated launch block survives cmd.exe
=========================================================================
Extracts a `:RUN_*` block from a generated launcher, swaps `llama-server.exe`
for an argv-echo stub, runs it under cmd.exe and compares the argv actually
received with the argv written in the file.

This is the regression that would have caught FND-014: an empty line inside a
`^` continuation makes cmd.exe silently drop every argument after it.

Usage (script in <llama-cpp-dir>/plan/, or anywhere with LLAMA_DIR set):
    .\\.venv\\Scripts\\python.exe _refactor2_body_probe.py
    .\\.venv\\Scripts\\python.exe _refactor2_body_probe.py --dir <backup-copy>

Environment: LLAMA_DIR (llama.cpp dir; default = parent.parent, i.e. the
script sits in <llama-cpp-dir>/plan/), LAUNCHER_DIR (generated launchers;
default <LLAMA_DIR>/launcher).
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or HERE.parent)
LAUNCHER_DIR = Path(os.environ.get("LAUNCHER_DIR") or (LLAMA_DIR / "launcher"))
SPECS = {
    "start-Qwen-Launcher.bat": "gbk",
    "start-Gemma4-Launcher.bat": "gbk",
    "start-CPU-Toolcall-Launcher.bat": "ascii",
}


def blocks_of(text):
    """Yield (label, [lines]) for every ':RUN_x ... pause & goto menu' block."""
    lines = text.split("\r\n")
    out, cur, label = [], None, None
    for ln in lines:
        if re.match(r"^:RUN_[A-Za-z0-9_]+$", ln):
            if cur is not None:
                out.append((label, cur))
            cur, label = [], ln
            continue
        if cur is None:
            continue
        cur.append(ln)
        if re.match(r"^\s*pause & goto menu\s*$", ln):
            out.append((label, cur))
            cur = None
    if cur is not None:
        out.append((label, cur))
    return out


def block_varmap(lines):
    """Resolve `set "X=..."` lines declared inside the block itself, so that
    `%REASONING_ARG%` etc. expand to their real (multi-token) value. Vars not
    declared in the block are stubbed as V_<name>."""
    raw = {}
    for ln in lines:
        # greedy value: the value itself may contain embedded double quotes,
        # e.g. set "X=--a %B% --msg "hello" --c"
        m = re.match(r'^\s*set "(\w+)=(.*)"\s*$', ln)
        if m:
            raw[m.group(1)] = m.group(2)

    def resolve(value, depth=0):
        if depth > 6:
            return value
        return re.sub(
            r"%(\w+)%",
            lambda m: resolve(raw[m.group(1)], depth + 1)
            if m.group(1) in raw else "V_" + m.group(1),
            value)

    return {k: resolve(v) for k, v in raw.items()}


def expected_argv(label, lines):
    """argv as written in the file (the `^` markers removed, %VAR% resolved).

    Surrounding double quotes are stripped, matching the stub's `%%~A`.
    """
    keep = [l for l in lines
            if l.strip() and not re.match(r"^\s*pause & goto menu\s*$", l)]
    varmap = block_varmap(keep)
    joined = " ".join(keep).replace("^", " ")
    joined = re.sub(r"%(\w+)%",
                    lambda m: varmap.get(m.group(1), "V_" + m.group(1)),
                    joined)
    rest = joined.split("llama-server.exe", 1)[1]
    return [t.strip('"') for t in rest.split()]


def probe(script, label, lines, tmp):
    stub = tmp / "args_stub.bat"
    stub.write_bytes(b"@echo off\r\necho ARGV_BEGIN\r\n"
                     b"for %%A in (%*) do echo [%%~A]\r\n"
                     b"echo ARGV_END\r\n")
    body = []
    for ln in lines[1:]:                       # drop the :RUN_ label
        if "call :check_file" in ln:
            continue
        # skip the interactive confirmation guard - it would `goto menu`
        # before the launch line (the probe has no menu and no stdin)
        if re.match(r"^\s*set /p ok=", ln):
            continue
        if re.match(r'^\s*if /i not "%ok%"=="Y" goto menu\s*$', ln):
            continue
        if re.match(r"^\s*pause & goto menu\s*$", ln):
            continue
        body.append(ln.replace("llama-server.exe",
                               'call "{}"'.format(stub)))
    # define every %VAR% used in this block with a non-empty stub value, so
    # nothing expands to an empty argument (which `for` would skip)
    used = set()
    for ln in body:
        used.update(re.findall(r"%(\w+)%", ln))
    head = ["@echo off"]
    for k in sorted(used):
        head.append('set "{0}=V_{0}"'.format(k))
    bat = tmp / "probe.bat"
    bat.write_bytes("\r\n".join(head + body).encode("ascii", errors="replace"))

    proc = subprocess.run(["cmd", "/c", str(bat)], capture_output=True,
                          text=True, errors="replace",
                          cwd=str(LLAMA_DIR), timeout=60)
    got = []
    inside = False
    for ln in proc.stdout.splitlines():
        s = ln.strip()
        if s == "ARGV_BEGIN":
            inside = True
            continue
        if s == "ARGV_END":
            break
        if inside and s.startswith("[") and s.endswith("]"):
            got.append(s[1:-1])
    return got


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(LAUNCHER_DIR),
                    help="directory holding the generated launchers "
                         "(default: LAUNCHER_DIR, else "
                         "<llama-cpp-dir>\\launcher). Point it at a backup "
                         "copy for a negative control.")
    args = ap.parse_args()

    problems = 0
    checked = 0
    base = Path(args.dir)
    for script, enc in SPECS.items():
        path = base / script
        if not path.exists():
            print("[X] missing: {}".format(path))
            problems += 1
            continue
        text = path.read_bytes().decode(enc, errors="replace")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for label, lines in blocks_of(text):
                if not any("llama-server.exe" in l for l in lines):
                    continue                    # goto-alias items, no launch
                want = expected_argv(label, lines)
                got = probe(script, label, lines, tmp)
                checked += 1
                want_norm = [re.sub(r"%(\w+)%", lambda m: m.group(0), w) for w in want]
                if len(got) != len(want_norm):
                    print("[X] {} {}: argv length {} != {} (written)"
                          .format(script, label, len(got), len(want_norm)))
                    print("      written: {}".format(" ".join(want_norm)))
                    print("      received: {}".format(" ".join(got)))
                    problems += 1
                    continue
                for a, b in zip(got, want_norm):
                    if a != b:
                        print("[X] {} {}: argv mismatch '{}' != '{}'"
                              .format(script, label, a, b))
                        problems += 1
                        break
    print("")
    if problems:
        print("FAIL - {} block(s) mismatched out of {} checked"
              .format(problems, checked))
        return 1
    print("PASS - {} launch block(s): cmd.exe receives every argument "
          "written in the file".format(checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
