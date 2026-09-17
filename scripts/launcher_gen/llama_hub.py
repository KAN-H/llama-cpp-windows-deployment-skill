#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama-hub - the friendly front door of the llama.cpp launcher ecosystem.

Lives in launcher\\, driven by ..\\llama-hub.bat at the repo root.

  llama-hub.bat                  interactive menu
  llama-hub.bat --guide          model picker table, then exit
  llama-hub.bat --diagnose       environment health check, then exit
  llama-hub.bat --write-docs     rebuild launcher\\docs\\*.md
  llama-hub.bat <anything else>  forwarded verbatim to update_launchers.py

Standard library only. Every path is derived from this file's location, so the
whole launcher\\ folder can be moved without editing anything.
"""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import unicodedata
from pathlib import Path

# Where the launcher DATA lives (preset-overrides.json, model-profiles.json,
# docs/, ...). Defaults to this file's own directory, so an existing install
# keeps working; set LAUNCHER_DIR when the tool lives somewhere else, as it does
# once installed in the skill, where the data belongs to the user.
BASE = Path(os.environ.get("LAUNCHER_DIR") or Path(__file__).resolve().parent).resolve()
ROOT = BASE.parent                              # ...\llama.cpp
sys.path.insert(0, str(BASE))

import update_launchers as U                    # noqa: E402  (shared helpers)

OVERRIDES = BASE / "preset-overrides.json"
PROFILES = BASE / "model-profiles.json"
HEALTH = BASE / "draft-health.json"
BLACKLIST = BASE / "draft-blacklist.json"
DOCS = BASE / "docs"

PORTS = [
    (8080, "Gemma4 菜单启动器", "start-Gemma4-Launcher.bat"),
    (8081, "向量 / 嵌入服务", "start-embedding.bat"),
    (8082, "Router 多模型统一入口（推荐）", "models-config.bat"),
    (8084, "Qwen 菜单启动器", "start-Qwen-Launcher.bat"),
    (8086, "纯 CPU 工具调用", "start-CPU-Toolcall-Launcher.bat"),
]

LAUNCHERS = {
    "router": ("models-config.bat", 8082),
    "gemma4": ("start-Gemma4-Launcher.bat", 8080),
    "qwen": ("start-Qwen-Launcher.bat", 8084),
    "cpu": ("start-CPU-Toolcall-Launcher.bat", 8086),
    "embed": ("start-embedding.bat", 8081),
}

HUB_FLAGS = {"--write-docs", "--guide", "--diagnose", "--audit",
             "--menu", "--params", "--params-preview", "--help", "-h"}


# ------------------------------------------------------------------ api key
# The key the servers are started with. This file ships in a public repository,
# so it is never written here - a key inside a print() is a key inside the repo,
# and this one would end up in every generated guide as well.
def api_key_display():
    """The configured key, or an explicit note that none is configured."""
    return os.environ.get("LLAMA_API_KEY") or "(未设置 - 请设 LLAMA_API_KEY)"


# ------------------------------------------------------------------ console

def setup_console():
    """Never crash on a code page that cannot represent a character, and keep
    our own output interleaved correctly with child processes when piped."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except Exception:
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def hr(ch="-", width=78):
    print(ch * width)


def disp(s):
    """Display width: CJK glyphs occupy two columns in a console."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in str(s))


def pad(s, width, align="<"):
    s = str(s)
    gap = width - disp(s)
    if gap <= 0:
        return s
    return s + " " * gap if align == "<" else " " * gap + s


def head(title):
    print()
    hr("=")
    print("  " + title)
    hr("=")


def ask(prompt, default=""):
    try:
        got = input("{} ".format(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return got or default


def confirm(prompt, default=False):
    got = ask("{} [{}]:".format(prompt, "Y/n" if default else "y/N"), "")
    if not got:
        return default
    return got.lower().startswith("y")


def pause():
    try:
        input("\n按回车返回菜单...")
    except (EOFError, KeyboardInterrupt):
        print()


# -------------------------------------------------------------- data access

def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except Exception:
        return default


def load_state():
    ovr = read_json(OVERRIDES, {})
    return {
        "overrides": ovr,
        "models": (ovr.get("models") or {}),
        "global": (ovr.get("global") or {}),
        "health": read_json(HEALTH, {}),
        "blacklist": set(
            (read_json(BLACKLIST, {}).get("drafts") or [])),
        "profiles": read_json(PROFILES, {}),
    }


def _gb(path):
    try:
        return path.stat().st_size / (1 << 30)
    except OSError:
        return 0.0


def _tps_now(entry, params):
    """Speed of the setting that is actually in use; falls back to the best.

    FND-067: measurements may live under measured / measured_realctx /
    measured_benchctx, so ask update_launchers rather than reading the legacy
    key directly. bench-ctx numbers are skipped entirely - they are not
    comparable with a router-context configuration.
    """
    meas, origin = U.measured_points(entry)
    if origin == "benchctx":
        return None
    meas = [m for m in meas if m.get("tps")]
    if not meas:
        return None
    cur = params.get("n-cpu-moe")
    if cur is not None:
        for m in meas:
            if m.get("n-cpu-moe") == cur:
                return m["tps"]
    return max(meas, key=lambda m: m["tps"])["tps"]


def _caps(mid, note, multimodal, mtp_variant, is_mtp_row):
    """Capability tags, ASCII only (Chinese labels render fine, tags do not)."""
    got = []
    if multimodal:
        got.append("IMG")
    if "MoE" in note:
        got.append("MoE")
    if is_mtp_row or mid.endswith("-MTP"):
        got.append("MTP")
    elif mtp_variant:
        got.append("+MTP")
    if "不支持工具调用" in note or "NoTools" in mid:
        got.append("NoTools")
    return got


def moe_cell(entry):
    """The MoE offload cell: the value plus where it came from.

    A bare "17" is not enough. The same number means four very different
    things depending on its provenance, and the difference is exactly what bit
    us - a small-context ladder keeps rewarding lower values right up to the
    point where the card runs out of VRAM:
      realctx     - measured at a real context; the only kind worth setting
      benchctx    - measured with llama-bench's tiny context, where the VRAM
                    wall is unreachable, so it can rank but never decide
      budget      - from the VRAM budget equation, not yet confirmed
      inherited   - taken from the family profile
    """
    params = entry.get("params") or {}
    n = params.get("n-cpu-moe")
    if n is None:
        return "-"
    src = str((entry.get("tuning") or {}).get("source") or "")
    if src.startswith("inherited-from:") or src == "inherited":
        return "{} 继承".format(n)
    _pts, origin = U.measured_points(entry)
    if origin == "benchctx":
        return "{} 实测小ctx".format(n)
    if origin in ("realctx", "measured") or "measured" in src:
        return "{} 实测".format(n)
    return "{} 预算".format(n)


def model_rows(state, chat=U.DEFAULT_CHAT):
    """One dict per routable entry: every model, plus its MTP variant if any.

    The Router exposes both, so the guide must show both - otherwise users
    never discover the -MTP IDs even though they are the fast ones.
    """
    try:
        dirs, mtp_files = U.scan_chat(chat)
    except Exception:
        dirs, mtp_files = {}, []
    rows = []
    for rel, entry in state["models"].items():
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if entry["id"].lower().startswith("auto"):
            continue                                    # stale registry junk
        note = entry.get("note") or ""
        params = entry.get("params") or {}
        mtp = entry.get("mtp") or {}
        info = dirs.get(rel) or {}
        mains = info.get("mains") or []
        weight = _gb(chat / rel / mains[0]) if mains else 0.0
        multimodal = bool(info.get("mmproj"))
        missing = bool(mains) and not (chat / rel / mains[0]).exists()
        tps = _tps_now(entry, params)

        rows.append({
            "kind": "base",
            "rel": rel,
            "id": entry["id"],
            "alias": entry.get("alias") or [],
            "note": note,
            "params": params,
            "ctx": params.get("ctx-size"),
            "weight": weight,
            "multimodal": multimodal,
            "draft": None,
            "moe": moe_cell(entry),
            "tps": tps,
            "flags": _caps(entry["id"], note, multimodal,
                           bool(mtp.get("id")), False),
            "missing": missing,
        })

        if mtp.get("id"):
            draft = None
            stem = rel.split("/")[-1]
            for cand in (mtp.get("draft"), stem + "-MTP"):
                if cand and any(cand == f for f in mtp_files):
                    draft = cand
                    break
            if draft is None:
                draft = mtp.get("draft")
            dw = _gb(chat / U.MTP_DIR_NAME / draft) if draft else 0.0
            m_params = mtp.get("params") or {}
            rows.append({
                "kind": "mtp",
                "rel": rel,
                "id": mtp["id"],
                "alias": mtp.get("alias") or [],
                "note": mtp.get("note") or ("{} 的 MTP 加速版".format(entry["id"])),
                "params": m_params,
                "ctx": m_params.get("ctx-size") or params.get("ctx-size"),
                "weight": weight + dw,
                "multimodal": multimodal,
                "draft": draft,
                "draft_gb": dw,
                "moe": moe_cell(entry),
                "tps": _tps_now(entry, params),
                "flags": _caps(mtp["id"], note, multimodal, True, True),
                "missing": missing,
            })
    rows.sort(key=lambda r: r["id"])
    return rows, mtp_files


def health_of(state, draft_name):
    pairs = (state["health"] or {}).get("pairs") or []
    for p in pairs:
        if p.get("draft") == draft_name:
            return p
    return None


# ------------------------------------------------------------------ guide

def cmd_guide(state=None):
    state = state or load_state()
    rows, mtp_files = model_rows(state)
    base = [r for r in rows if r["kind"] == "base"]
    mtps = [r for r in rows if r["kind"] == "mtp"]
    head("模型选择指南")
    print("  该选哪个模型？看这一页就够了。")
    print()
    print("  共 {} 个可选用条目 = {} 个模型 + {} 个 MTP 加速版"
          .format(len(rows), len(base), len(mtps)))
    print("  启动 Router（端口 8082）后，它们全都在同一个地址里，")
    print("  在网页下拉框或 API 的 model 字段里写 ID 即可，不用重启服务。")
    print()

    idw = min(max([disp(r["id"]) for r in rows] + [26]), 34)
    print("  {} {} {} {} {} {}".format(
        pad("模型 ID", idw), pad("GB", 6, ">"), pad("ctx", 5, ">"),
        pad("MoE 卸载", 11), pad("能力", 16), pad("实测", 8, ">")))
    hr()
    for r in rows:
        ctx = r["ctx"]
        ctx_s = "{:.0f}K".format(ctx / 1024) if isinstance(ctx, (int, float)) else "?"
        tps = "{:.0f} t/s".format(r["tps"]) if r["tps"] else "-"
        caps = " ".join(r["flags"]) or "-"
        mark = "  " if r["kind"] == "base" else "> "
        print("{} {} {} {} {} {}".format(
            mark + pad(r["id"], idw - 2), pad("{:.2f}".format(r["weight"]), 6, ">"),
            pad(ctx_s, 5, ">"), pad(r.get("moe") or "-", 11),
            pad(caps, 16), pad(tps, 8, ">")))
        if r["kind"] == "mtp" and r["draft"]:
            print("    {}草稿 {} ({:.2f} GB)".format(
                " " * (idw - 2), r["draft"], r.get("draft_gb") or 0.0))
    hr()
    print("  能力说明（列表里只用英文，避免乱码）：")
    print("    IMG      可以读图片（图文多模态），能看懂截图和照片")
    print("    MoE      稀疏专家模型，长上下文更省显存")
    print("    MTP      本身就是投机解码加速版，回答更快")
    print("    +MTP     主模型可以挂草稿加速，下面 > 那行就是它的加速版")
    print("    NoTools  不支持 function calling（不能给 Agent 当工具模型）")
    print("  MoE 卸载 = 专家层放 CPU 的数量 + 来源：实测 / 实测小ctx / 预算 / 继承。")
    print("    只有「实测」（真实上下文）能用来定值；「实测小ctx」只能给排序 ——")
    print("    llama-bench 的小上下文撞不到显存墙，照着改会 OOM（FND-067）。")
    print("  > 开头的行是加速版；GB 列是模型权重占用（不含 KV 缓存）。")
    print()

    print("  【按需选型】")
    for label, sid in (
            ("日常聊天 / 看图          ", "G1 "),
            ("中文长文 / 高质量推理    ", "Q1 "),
            ("破限对话（纯文本）       ", "G3 "),
            ("要更快：选 > 开头的行    ", "G1m"),
            ("超小体积 / 低配兜底      ", "Q4 "),
            ("纯 CPU（无显卡）         ", None)):
        if sid is None:
            print("    {} -> 菜单选 5，端口 8086".format(label))
            continue
        hit = next((r for r in rows if r["id"].startswith(sid)), None)
        print("    {} -> {}".format(label, hit["id"] if hit else sid))
    print()
    print("  重要：列表里看不到某个模型？磁盘上有 ≠ 能被列出。")
    print("        Router 每个子目录只暴露 1 个条目，且同名的 preset 节会覆盖它。")
    print()

    print("  【旧 ID 仍然可用】别名映射（旧 ID -> 实际 ID）")
    hr()
    any_alias = False
    for r in rows:
        for a in r["alias"]:
            any_alias = True
            print("    {} -> {}".format(pad(a, 56), r["id"]))
    if not any_alias:
        print("    （无）")
    print()
    print("  别名是「追加」不是「改名」：新旧 ID 都有效，老客户端配置不用动。")
    print("  同一个模型的多个别名可以写在 alias 的同一行，用逗号隔开。")
    return rows


# --------------------------------------------------------------- diagnose

def port_busy(port, host="127.0.0.1"):
    s = socket.socket()
    s.settimeout(0.35)
    try:
        return s.connect_ex((host, port)) == 0
    except Exception:
        return False
    finally:
        s.close()


def _capture(cmd, timeout=25):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, shell=False)
        return p.stdout.decode("utf-8", "replace").strip()
    except Exception as exc:
        return "<failed: {}>".format(exc)


def cmd_diagnose():
    state = load_state()
    head("环境诊断")
    problems, warnings = [], []

    # 1. python
    print("  [1] 运行环境")
    print("      python        : {} {}".format(
        sys.version.split()[0], "(64-bit)" if sys.maxsize > 2**32 else "(32-bit)"))
    print("      script folder : {}".format(BASE))
    print("      repo root     : {}".format(ROOT))

    # 2. binaries
    exe = U.find_server_exe()
    print("  [2] llama.cpp 二进制")
    if exe:
        ver = _capture([str(exe), "--version"])
        m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", ver)
        shown = "build {} ({})".format(m.group(1), m.group(2)[:9]) if m else ver.splitlines()[0] if ver else "ok"
        print("      llama-server  : {}  [{}]".format(exe, shown))
    else:
        print("      llama-server  : 未找到")
        problems.append("找不到 llama-server.exe（应位于 {} 或其上级）".format(BASE))
    bench = next((p for p in (BASE / "llama-bench.exe",
                              ROOT / "llama-bench.exe") if p.exists()), None)
    print("      llama-bench   : {}".format(bench if bench else "未找到（调优扫描不可用）"))
    if not bench:
        warnings.append("缺 llama-bench.exe：--tune-sweep 无法运行")

    keys = read_json(BASE / ".llama-server-keys.json", {})
    if keys:
        print("      option 白名单 : {} 个 key（针对 build {}）".format(
            len(keys.get("keys") or []), keys.get("build", "?")))

    # 3. GPU
    print("  [3] GPU / 驱动")
    smi = _capture(["nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader"], timeout=20)
    if smi.startswith("<failed") or "not recognized" in smi.lower():
        print("      nvidia-smi    : 不可用 -> 只能跑纯 CPU 模式（端口 8086）")
        warnings.append("nvidia-smi 不可用：GPU 加速与 --audit 显存估算失效")
    else:
        for line in smi.splitlines():
            print("      " + line.strip())

    # 4. models
    print("  [4] 模型文件")
    chat = U.DEFAULT_CHAT
    try:
        dirs, mtp_files = U.scan_chat(chat)
    except Exception as exc:
        dirs, mtp_files = {}, []
        problems.append("扫描模型目录失败: {}".format(exc))
    total_gb = 0.0
    seen = set()
    for rel, info in dirs.items():
        for f in info["mains"]:
            p = chat / rel / f
            if p.exists():
                total_gb += p.stat().st_size / (1 << 30)
                seen.add(str(p))
    print("      models dir    : {}  {}".format(
        chat, "OK" if chat.exists() else "缺失"))
    print("      模型目录数    : {} 个（Router 每个目录 = 1 个条目）".format(len(dirs)))
    print("      权重合计     : {:.1f} GB".format(total_gb))
    print("      MTP 草稿     : {} 个".format(len(mtp_files)))

    rows, _ = model_rows(state, chat)
    ini_path = Path((state["overrides"] or {}).get(
        "ini_path") or (chat / "models-config.ini"))
    ini_secs = U.read_ini_sections(ini_path) if ini_path.exists() else {}
    print("      overrides    : {} 个模型条目".format(len(rows)))
    print("      preset ini   : {} 个节 ({})".format(len(ini_secs), ini_path))
    if not ini_path.exists():
        problems.append("找不到 models-config.ini: {}".format(ini_path))

    miss = [r["id"] for r in rows if r["missing"]]
    if miss:
        problems.append("以下条目的权重文件在磁盘上找不到: " + ", ".join(miss))
    orphan = set(dirs) - set(state["models"])
    if orphan:
        warnings.append("磁盘上有 {} 个目录不在 preset-overrides.json 中: {}".format(
            len(orphan), ", ".join(sorted(orphan)[:4])))
    unlisted = [r["id"] for r in rows if r["id"].lower().startswith("auto")]
    if unlisted:
        warnings.append("残留 AUTO_* 条目 {} 个（可运行 llama-hub.bat --extract 清理）"
                        .format(len(unlisted)))

    # 5. drafts
    print("  [5] MTP 草稿健康度（实测接受率，阈值 30%）")
    hl = (state["health"] or {}).get("pairs") or []
    if not hl:
        print("      （无记录，运行 llama-hub.bat --validate-drafts 生成）")
    for p in hl:
        acc = p.get("acceptance") or p.get("acc")
        ok = (acc is not None and acc >= 0.30)
        print("      {:<28} -> {:<34} acc={} {} ".format(
            str(p.get("model", p.get("main", "?")))[:28],
            str(p.get("draft", "?"))[:34],
            "{:.3f}".format(acc) if acc is not None else "?", "OK" if ok else "FAIL"))
        if acc is not None and not ok:
            problems.append("草稿 {} 接受率过低（{:.3f}）".format(p.get("draft"), acc))
    bl = state["blacklist"]
    if bl:
        print("      黑名单       : {} 个（不会被自动选中）".format(len(bl)))

    # 6. ports
    print("  [6] 端口占用")
    for port, label, bat in PORTS:
        busy = port_busy(port)
        print("      {:<6} {:<34} {}".format(
            port, label, "已占用（服务在跑）" if busy else "空闲"))
    hot = [p for p, _, _ in PORTS if port_busy(p)]
    if len(hot) == 1:
        print("      提示: 端口 {} 有服务，可直接用 http://127.0.0.1:{}/ 访问。".format(
            hot[0], hot[0]))
    elif len(hot) > 1:
        warnings.append("同时有 {} 个端口在跑，16GB 显存可能不够：{}".format(
            len(hot), hot))

    # summary
    print()
    hr("=")
    if problems:
        print("  [X] 发现 {} 个必须处理的问题:".format(len(problems)))
        for p in problems:
            print("      - " + p)
    else:
        print("  [OK] 未发现阻塞性问题。")
    if warnings:
        print("  [!] {} 个提醒:".format(len(warnings)))
        for w in warnings:
            print("      - " + w)
    hr("=")
    return not problems


# --------------------------------------------------------------- actions

def run_updater(args):
    cmd = [sys.executable, str(BASE / "update_launchers.py")] + list(args)
    print("  $ " + " ".join(cmd[1:]), flush=True)
    hr()
    sys.stdout.flush()
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    sys.stdout.flush()
    return rc


def act_update():
    head("更新模型清单 / 同步启动器")
    print("  第 1 步：预览（不会改动任何文件）")
    hr()
    rc = run_updater(["--check"])
    print()
    if rc == 2:
        print("  " + "!" * 58)
        print("  发现需要你确认的新模型（候选参数已在上面列出）。")
        print("  在你确认之前，程序不会写入任何文件（DEC-001）。")
        print("  " + "!" * 58)
        if not confirm("  确认接受上面列出的新模型参数并写入？", default=False):
            print("  已取消，文件未改动。")
            print("  若参数不合适，可以：")
            print("    - 编辑 launcher\\model-profiles.json 补充该型号的参数画像，或")
            print("    - 编辑 launcher\\preset-overrides.json 手工指定，然后重试")
            return
        args = ["--fix-moe", "--accept-new", "--yes"]
    else:
        if not confirm("  第 2 步：把上面的改动正式写入？", default=False):
            print("  已取消，文件未改动。")
            return
        args = ["--fix-moe", "--yes"]
    hr()
    rc = run_updater(args)
    if rc == 0:
        print("\n  [OK] 完成。备份在 {}".format(BASE / "backup"))
    else:
        print("\n  [X] 更新失败（退出码 {}）".format(rc))


def start_bat(key):
    name, port = LAUNCHERS[key]
    bat = BASE / name
    if not bat.exists():
        print("  [X] 找不到 {}".format(bat))
        return
    if port_busy(port):
        print("  [!] 端口 {} 已被占用。".format(port))
        if not confirm("      继续启动（可能失败）？", default=False):
            print("      已取消。")
            return
    head("启动 {}（端口 {}）".format(name, port))
    print("  浏览器访问  : http://127.0.0.1:{}/".format(port))
    print("  API 地址     : http://127.0.0.1:{}/v1".format(port))
    print("  会话密钥     : {}".format(api_key_display()))
    print("  启动器窗口会保持打开；关掉那个窗口即停止服务。")
    hr()
    subprocess.Popen('start "" "{}"'.format(bat), shell=True, cwd=str(ROOT))
    print("  已在新窗口启动。本菜单可以关掉，服务不受影响。")


def act_tune():
    state = load_state()
    head("参数调优 / 扫描")
    rows, _ = model_rows(state)
    moe = [r for r in rows if r["kind"] == "base" and (
        "MoE" in r["note"] or any(k == "n-cpu-moe" for k in (r["params"] or {})))]
    if moe:
        print("  当前 MoE 模型的 n-cpu-moe 实测记录：")
        hr()
        for r in moe:
            meas = (state["models"][r["rel"]].get("tuning") or {}).get("measured") or []
            cur = (r["params"] or {}).get("n-cpu-moe")
            best = max(meas, key=lambda m: m.get("tps") or 0) if meas else None
            line = "    {:<34} 当前={}".format(r["id"][:34], cur)
            if meas:
                line += "  实测: " + ", ".join(
                    "{}->{}t/s".format(m.get("n-cpu-moe"), m.get("tps"))
                    for m in meas)
            print(line)
            if best and cur is not None and best.get("n-cpu-moe") != cur:
                print("      [!] 最优是 {}（{} t/s），当前 {} 不是最优。".format(
                    best.get("n-cpu-moe"), best.get("tps"), cur))
        hr()
    print("  [A] 审计（只读）：显存估算 / 参数冲突 / 实测对比")
    print("  [S] 扫描 n-cpu-moe 阶梯（耗时，需关闭其它服务）")
    print("  [M] 校验全部 MTP 草稿（逐个加载实测）")
    print("  [B] 返回")
    choice = ask("  选择 [B]:").lower()
    if choice == "a":
        cmd_audit()
    elif choice == "s":
        target = ask("  目标（模型目录名，或 all）:", "all")
        reps = ask("  每个档位重复次数（默认 3，越高质量越稳）:", "3")
        run_updater(["--tune-sweep", target, "--reps", reps])
    elif choice == "m":
        run_updater(["--validate-drafts"])
    print()


def act_docs():
    head("文档与 Web UI")
    print("  服务已启动时，直接用浏览器打开对应端口即可（见下表）。")
    hr()
    for port, label, bat in PORTS:
        mark = "运行中" if port_busy(port) else "  --  "
        print("    [{}]  http://127.0.0.1:{}/   {:<32} {}".format(
            mark, port, label, bat))
    hr()
    print("  本地文档：")
    for f in sorted(DOCS.glob("*")):
        print("    {}".format(f))
    print()
    print("  [O] 用默认程序打开 docs 文件夹")
    print("  [M] 用默认程序打开模型选择指南 (MODELS.md)")
    print("  [B] 返回")
    choice = ask("  选择 [B]:").lower()
    if choice in ("o", "m"):
        target = DOCS if choice == "o" else (DOCS / "MODELS.md")
        if not target.exists():
            print("  [!] {} 不存在，先运行 --write-docs。".format(target))
            return
        try:
            os.startfile(str(target))
        except Exception as exc:
            print("  [X] 打开失败: {}".format(exc))
    print()


# ------------------------------------------------------------------- audit

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def model_budget(chat, rel, entry, gparams, dirs, mtp_files):
    """Estimate the VRAM split for one model without loading it."""
    info = dirs.get(rel) or {}
    main = (info.get("mains") or [None])[0]
    out = {"rel": rel, "main": main, "ok": False, "why": "no gguf on disk"}
    if not main:
        return out
    path = chat / rel / main
    meta = U.read_gguf_meta(path)
    if not meta:
        out["why"] = "unreadable gguf header"
        return out

    params = entry.get("params") or {}
    ctx = _num(params.get("ctx-size")) or _num(gparams.get("ctx-size")) or 8192
    kv_type = params.get("cache-type-v") or gparams.get("cache-type-v") or "f16"
    kv = U.kv_estimate_gb(meta, int(ctx), str(kv_type))
    weight = _gb(path)

    mm = 0.0
    for f in (info.get("mmproj") or [])[:1]:
        mm = _gb(chat / rel / f)

    draft_name = (entry.get("mtp") or {}).get("draft")
    draft = _gb(chat / U.MTP_DIR_NAME / draft_name) if draft_name else 0.0
    name_ok = bool(draft_name) and any(draft_name == f for f in mtp_files)

    layers = meta.get("layers") or 0
    is_moe = bool(meta.get("experts"))
    ncpu = _num(params.get("n-cpu-moe"))
    if ncpu is None:
        ncpu = _num(gparams.get("n-cpu-moe")) or 0
    ncpu = int(ncpu)
    share = (ncpu / layers) if (is_moe and layers) else 0.0
    # the MoE expert FFN is roughly 90% of an MoE checkpoint's bytes
    cpu_gb = weight * share * 0.90
    gpu_gb = (weight - cpu_gb) + (kv or 0.0) + mm + draft

    out.update({
        "ok": True,
        "ctx": int(ctx),
        "ctx_train": meta.get("ctx_train"),
        "layers": layers,
        "experts": meta.get("experts"),
        "is_moe": is_moe,
        "ncpu": ncpu,
        "weight": weight,
        "cpu_gb": cpu_gb,
        "gpu_gb": gpu_gb,
        "total_gb": gpu_gb + cpu_gb,
        "kv": kv,
        "mm": mm,
        "draft": draft,
        "draft_name": draft_name,
        "draft_ok": name_ok,
        "kv_type": kv_type,
        "arch": meta.get("arch"),
    })
    return out


def cmd_audit(state=None):
    state = state or load_state()
    gparams = state["global"]
    rows, mtp_files = model_rows(state)
    bases = [r for r in rows if r["kind"] == "base"]
    head("配置审计（只读，不加载模型）")
    print("  含两部分：显存/参数审计 + 注册表一致性审计")

    try:
        smi = _capture(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits"], timeout=20)
        total, used = [float(x) for x in smi.splitlines()[0].split(",")[:2]]
        vram = total / 1024.0
        free = (total - used) / 1024.0
    except Exception:
        vram, free = None, None

    print("  全局默认  : " + ", ".join(
        "{}={}".format(k, v) for k, v in sorted(gparams.items())))
    if vram:
        print("  显存      : {:.1f} GB 总量 / {:.1f} GB 空闲".format(vram, free))
    else:
        print("  显存      : 读不到 nvidia-smi，无法做显存判定")
    print()

    try:
        dirs, mtp_files = U.scan_chat(U.DEFAULT_CHAT)
    except Exception:
        dirs, mtp_files = {}, []

    Budget = type("_B", (), {})
    budget = Budget()
    over, near = [], []
    print("  {} {} {} {} {} {}".format(
        pad("模型 ID", 30), pad("权重", 6, ">"), pad("KV缓存", 7, ">"),
        pad("MoE卸载", 7, ">"), pad("估算显存", 8, ">"), "判定"))
    hr()
    for r in bases:
        checks = []
        b = model_budget(U.DEFAULT_CHAT, r["rel"], state["models"][r["rel"]],
                         gparams, dirs, mtp_files)
        if not b["ok"]:
            print("  {}  [X] {}".format(pad(r["id"], 30), b["why"]))
            over.append((r["id"], b["why"]))
            continue
        verdict = "OK"
        meas = (state["models"][r["rel"]].get("tuning") or {}).get("measured") or []
        if vram:
            if b["gpu_gb"] > vram:
                verdict = "超显存 +{:.1f}G".format(b["gpu_gb"] - vram)
                over.append((r["id"], verdict))
            elif b["gpu_gb"] > vram - 0.5:
                verdict = "临界，可能 OOM"
                near.append(r["id"] + " :: 估算显存贴近卡容量上限")
            elif meas:
                verdict = "已实测"
            elif b["is_moe"] and b["ncpu"]:
                verdict = "未实测，建议扫描"
                near.append(r["id"] + " :: 有 MoE 卸载值但无实测记录")
        print("  {} {} {} {} {} {}".format(
            pad(r["id"] + (" *" if meas else ""), 30),
            pad("{:.2f}".format(b["weight"]), 6, ">"),
            pad("{:.2f}".format(b["kv"] or 0), 7, ">"),
            pad("{}/{}".format(b["ncpu"], b["layers"]) if b["is_moe"] else "-", 7, ">"),
            pad("{:.2f}".format(b["gpu_gb"]), 8, ">"), verdict))

        # ---- per-model parameter checks
        p = dict(gparams)
        p.update(state["models"][r["rel"]].get("params") or {})
        if b["ctx_train"] and b["ctx"] > b["ctx_train"]:
            checks.append("ctx-size {} 超过模型训练上限 {}，会被静默截断".format(
                b["ctx"], b["ctx_train"]))
        if "mlock" in p and "load-mode" in p:
            checks.append("mlock 与 load-mode 同时存在，后者会覆盖前者")
        if p.get("flash-attn") in (None, "off", "false", "0") and \
                str(b["kv_type"]).startswith("q"):
            checks.append("KV 量化 {} 但 flash-attn 未开（可能失败或极慢）"
                          .format(b["kv_type"]))
        bs = _num(p.get("batch-size"))
        if bs and bs > b["ctx"]:
            checks.append("batch-size {} 大于 ctx-size {}".format(int(bs), b["ctx"]))
        tp = _num(p.get("threads"))
        if tp and tp > 12:
            checks.append("threads {} 超过物理核 12，上下文切换会拖慢".format(int(tp)))
        if b["is_moe"] and not b["ncpu"]:
            checks.append("是 MoE 模型但没设 n-cpu-moe，专家层会挤爆显存")
        if b["draft_name"] and not b["draft_ok"]:
            checks.append("MTP 草稿 {} 不在磁盘上".format(b["draft_name"]))
        for c in checks:
            print("      [!] " + c)
            near.append(r["id"] + " :: " + c)

        # ---- measured vs current (only report a change worth chasing)
        # FND-067: prefer the real-context ladder. A llama-bench ladder runs its
        # own tiny context, never reaches the VRAM ceiling and so keeps looking
        # better all the way down - it may only be used for ordering.
        tbl = state["models"][r["rel"]]
        meas, origin = U.measured_points(tbl)
        if len(meas) >= 2 and origin != "benchctx":
            best = max(meas, key=lambda m: m.get("tps") or 0)
            cur = b["ncpu"]
            wall = (tbl.get("tuning") or {}).get("vram_wall") or {}
            if best.get("n-cpu-moe") not in (None, cur) and best.get("tps"):
                cur_tps = next((m["tps"] for m in meas
                                if m.get("n-cpu-moe") == cur), None)
                if cur_tps:
                    gain = best["tps"] - cur_tps
                    if gain >= max(3.0, cur_tps * 0.08):
                        if (wall.get("ncpu_moe_wall") is not None
                                and best["n-cpu-moe"] <= wall["ncpu_moe_wall"]):
                            print("      [!] 实测最优 n-cpu-moe {} -> {} t/s 更快，"
                                  "但已超过显存墙（墙={}），余量不足，不建议切换".format(
                                      best["n-cpu-moe"], best["tps"],
                                      wall["ncpu_moe_wall"]))
                        else:
                            print("      [!] n-cpu-moe 当前 {} -> {} t/s；"
                                  "实测最优 {} -> {} t/s（+{:.0f} t/s）".format(
                                      cur, cur_tps, best["n-cpu-moe"], best["tps"], gain))
                            print("          这是你自己实测的数据，是否采用由你决定；"
                                  "复核用: llama-hub.bat --tune-sweep all")
                            near.append("{} :: n-cpu-moe 有更优档位".format(r["id"]))

    hr()
    if over:
        print("  [X] {} 个模型跑不起来或会 OOM：".format(len(over)))
        for mid, why in over:
            print("      {}  -> {}".format(mid, why))
    else:
        print("  [OK] 所有模型估算显存都在卡容量内。")
    if near:
        print("  [!] {} 条提醒（上面带 [!] 的行）：".format(len(near)))
    else:
        print("  [OK] 未发现冲突参数。")
    print()
    print("  注意：估算值按权重 + KV + mmproj + 草稿推算，不含计算缓冲，")
    print("        实际占用通常略高。要拿到真值请启动一次并看 /metrics。")
    print("        * = 该值来自你的实测记录，不是估算。")
    print()

    print()
    print("  ---- 追加：注册表 vs 模型画像一致性审计 ----")
    run_updater(["--audit"])
    return not over


# ------------------------------------------------------------- write docs

def write_models_md(state):
    rows, mtp_files = model_rows(state)
    hl = (state["health"] or {}).get("pairs") or []
    L = []
    L.append("# 模型选择指南")
    L.append("")
    L.append("> 本文件由 `llama-hub.bat --write-docs` 自动生成，**请勿手工编辑**。")
    L.append("> 要改内容，请编辑 `launcher/preset-overrides.json` 后重新生成。")
    L.append("")
    L.append("生成时间: {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    L.append("")
    L.append("## 30 秒上手")
    L.append("")
    L.append("1. 双击根目录的 `llama-hub.bat`")
    L.append("2. 选 `2` 启动 Router 模式（端口 8082）")
    L.append("3. 浏览器打开 <http://127.0.0.1:8082/>")
    L.append("4. 在页面顶部的模型下拉框里选一个，开始聊天")
    L.append("")
    L.append("**API Key**: `{}`".format(api_key_display()))
    L.append("")
    L.append("## 全部模型（Router 模式下都在同一个端口 8082）")
    L.append("")
    L.append("| 模型 ID | 类型 | 权重 | 上下文 | MoE 卸载 | 能力 | 实测 | 说明 |")
    L.append("|---|---|---:|---:|---:|---|---:|---|")
    for r in rows:
        ctx = r["ctx"]
        ctx_s = "{:.0f}K".format(ctx / 1024) if isinstance(ctx, (int, float)) else "?"
        caps = " ".join(r["flags"]) or "-"
        tps = "{:.0f} t/s".format(r["tps"]) if r["tps"] else "-"
        note = r["note"].split("·")[0].replace("用途:", "").strip()
        note = note.replace("|", "/")
        kind = "MTP 加速版" if r["kind"] == "mtp" else "主模型"
        L.append("| `{}` | {} | {:.2f} GB | {} | {} | {} | {} | {} |".format(
            r["id"], kind, r["weight"], ctx_s, r.get("moe") or "-",
            caps, tps, note))
    L.append("")
    L.append("能力图例（表中只用英文标签，避免乱码）：")
    L.append("")
    L.append("- **IMG** = 可以读图片 · **MoE** = 稀疏专家，长上下文省显存")
    L.append("- **MTP** = 这一行本身就是投机解码加速版 · "
             "**+MTP** = 主模型可挂草稿，表里紧跟着的 MTP 行就是它的加速版")
    L.append("- **NoTools** = 不支持 function calling，不能给 Agent 当工具模型")
    L.append("")
    L.append("权重列是模型本体占用（GB），不含 KV 缓存；MTP 行已包含草稿体积。")
    L.append("")
    L.append("### MoE 卸载列怎么读")
    L.append("")
    L.append("数字 = 放多少层专家到 CPU（`--n-cpu-moe`）。**数字越小越快，但显存越紧。**")
    L.append("但比数字更重要的是它的来源 —— 同一个数字，来源不同可信度完全不同：")
    L.append("")
    L.append("| 来源 | 含义 | 能不能用来定值 |")
    L.append("|---|---|---|")
    L.append("| **实测** | 在真实上下文下扫描出的最优值 | 可以 —— 唯一权威来源 |")
    L.append("| **实测小ctx** | 用 llama-bench 的小上下文扫出来的 | **不可以** —— 只能给排序。"
             "小上下文撞不到显存墙，扫描会一路「越低越快」，照着改直接 OOM（FND-067） |")
    L.append("| **预算** | 由显存预算方程算出的初值，尚未实测 | 作起点可以，需实测确认 |")
    L.append("| **继承** | 来自同家族 profile 的推荐值 | 同上 |")
    L.append("")
    L.append("定值判据：**稳定解码 ≥ 35 t/s 且显存余量 > 3 GB**。"
             "3 GB 不是拍脑袋 —— 一次视觉请求就会吃掉 ~500 MiB 的 "
             "CUDA graph 缓存且不释放。")
    L.append("")
    L.append("改这个值请用 `llama-hub.bat` 菜单 11（参数速调），"
             "它会先算预计显存再写入。")
    L.append("")
    L.append("## 旧 ID 仍可用（别名映射）")
    L.append("")
    L.append("别名是**追加**的，不是改名。新旧 ID 同时有效，老客户端配置不用动。")
    L.append("")
    L.append("| 旧 ID（别名） | 现在指向 |")
    L.append("|---|---|")
    for r in rows:
        for a in r["alias"]:
            L.append("| `{}` | `{}` |".format(a, r["id"]))
    L.append("")
    L.append("## MTP 投机解码：草稿选哪个")
    L.append("")
    L.append("`gemma4_mtp\\` 里同一模型的草稿有多个精度。**越大越准，但更占显存**——"
             "16GB 卡上选错了会直接挤爆显存。")
    L.append("")
    L.append("| 主模型 | 出题草稿 | 实测接受率 | 平均接受长度 | 判定 |")
    L.append("|---|---|---:|---:|---|")
    for p in hl:
        acc = p.get("acceptance") or p.get("acc")
        ok = (acc is not None and acc >= 0.30)
        L.append("| `{}` | `{}` | {} | {} | {} |".format(
            p.get("model", p.get("main", "?")), p.get("draft", "?"),
            "{:.3f}".format(acc) if acc is not None else "?",
            p.get("mean_len", "?"),
            "健康" if ok else "**不要用**（低于 30% 阈值）"))
    L.append("")
    L.append("已核验的草稿文件（结构必须为 `gemma4-assistant` / 4 层）：")
    L.append("")
    for d in (state["health"].get("drafts") or []):
        L.append("- `{}` — {} / {} 层 / {:.2f} GB / {}".format(
            d.get("file", "?"), d.get("arch", "?"), d.get("layers", "?"),
            (d.get("size") or 0) / (1 << 30),
            "OK" if d.get("meta_ok") else "**不合格**"))
    L.append("")
    L.append("## 三条硬规则")
    L.append("")
    L.append("1. **同一个模型只会显示一个主条目**（外加可选的 `-MTP` 条目）。"
             "磁盘上同一目录里的多个 gguf 只会暴露 1 个 ID。")
    L.append("2. **16GB 显存同时只跑 1 个模型**。Router 已设 `--models-max 1`，"
             "切换时自动卸载上一个，不要手动改大。")
    L.append("3. **参数改 `preset-overrides.json`，不要改 `models-config.ini`**。"
             "ini 是生成物，下次同步会被覆盖（但手写的参数会被保留，见下）。")
    L.append("")
    L.append("## 参数怎么写才不会被覆盖")
    L.append("")
    L.append("`update_launchers.py` 会保留 ini 里**不由脚本管理**的键。"
             "被管理的键只有这些，其余手写内容一律原样保留：")
    L.append("")
    L.append("`model` · `mmproj` · `alias` · `spec-type` · `spec-draft-model` · "
             "`spec-draft-n-max`")
    L.append("")
    L.append("也就是说，直接在 `models-config.ini` 的某个 `[节]` 里加 "
             "`temperature = 0.8` 这类行是**安全**的。")
    L.append("")
    L.append("## 实测调优记录（用户实测，已在配置中生效）")
    L.append("")
    L.append("| 模型 | 参数 | 档位 -> 速度 |")
    L.append("|---|---|---|")
    for r in rows:
        if r["kind"] != "base":
            continue
        meas = (state["models"][r["rel"]].get("tuning") or {}).get("measured") or []
        if not meas:
            continue
        key = "n-cpu-moe" if any("n-cpu-moe" in m for m in meas) else "?"
        cells = ", ".join("{} -> {} t/s".format(
            m.get(key, m.get("value", "?")), m.get("tps", "?")) for m in meas)
        L.append("| `{}` | `{}` = {} | {} |".format(
            r["id"], key, (r["params"] or {}).get(key, "?"), cells))
    L.append("")
    L.append("## 出问题了怎么办")
    L.append("")
    L.append("| 症状 | 处理 |")
    L.append("|---|---|")
    L.append("| Router 启动即退出 | 某个 `[节]` 里有非法参数名，运行 "
             "`llama-hub.bat --ini-diff` 看差异 |")
    L.append("| 列表里出现不认识的模型 | `llama-hub.bat --extract` 重新生成清单 |")
    L.append("| 模型报了但加载失败 | 权重路径不对，`llama-hub.bat --fix-mmproj` |")
    L.append("| MTP 变慢或乱码 | `llama-hub.bat --validate-drafts` 重新实测 |")
    L.append("| 改坏了想还原 | `launcher\\backup\\` 里有带时间戳的每个旧版本 |")
    L.append("")
    return "\n".join(L) + "\n"


def write_arch_md(state):
    rows, mtp_files = model_rows(state)
    L = []
    L.append("# 启动器生态结构")
    L.append("")
    L.append("> 由 `llama-hub.bat --write-docs` 自动生成。")
    L.append("")
    L.append("生成时间: {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    L.append("")
    L.append("## 文件关系")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("  HUB[\"llama-hub.bat<br/>根目录唯一入口\"] --> PY[\"launcher/llama_hub.py\"]")
    L.append("  PY -->|菜单 1 更新| UPD[\"launcher/update_launchers.py\"]")
    L.append("  PY -->|菜单 2| R[\"launcher/models-config.bat<br/>Router :8082\"]")
    L.append("  PY -->|菜单 3| G[\"launcher/start-Gemma4-Launcher.bat<br/>:8080\"]")
    L.append("  PY -->|菜单 4| Q[\"launcher/start-Qwen-Launcher.bat<br/>:8084\"]")
    L.append("  PY -->|菜单 5| C[\"launcher/start-CPU-Toolcall-Launcher.bat<br/>:8086\"]")
    L.append("  PY -->|菜单 10| E[\"launcher/start-embedding.bat<br/>:8081\"]")
    L.append("")
    L.append("  OVR[\"launcher/preset-overrides.json<br/>人工调参唯一入口\"] --> UPD")
    L.append("  PROF[\"launcher/model-profiles.json<br/>模型家族/显存画像\"] --> UPD")
    L.append("  REG[\"launcher/launcher-models.json<br/>启动器菜单结构\"] --> UPD")
    L.append("  BLK[\"launcher/draft-blacklist.json\"] --> UPD")
    L.append("  HLTH[\"launcher/draft-health.json\"] --> UPD")
    L.append("  KEYS[\"launcher/.llama-server-keys.json<br/>合法参数名白名单\"] --> UPD")
    L.append("")
    L.append("  UPD -->|生成| R")
    L.append("  UPD -->|生成| G")
    L.append("  UPD -->|生成| Q")
    L.append("  UPD -->|生成| C")
    L.append("  UPD -->|生成| INI[\"D:/dev/models/chat/models-config.ini\"]")
    L.append("  UPD -->|每次覆盖前| BK[\"launcher/backup/\"]")
    L.append("  UPD -->|dry-run| PV[\"launcher/backup/preview/\"]")
    L.append("  INI --> R")
    L.append("```")
    L.append("")
    L.append("## 端口分配")
    L.append("")
    L.append("| 端口 | 用途 | 启动器 | 当前状态 |")
    L.append("|---:|---|---|---|")
    for port, label, bat in PORTS:
        L.append("| {} | {} | `{}` | {} |".format(
            port, label, bat, "占用中" if port_busy(port) else "空闲"))
    L.append("")
    L.append("API Key 统一为 `{}`。".format(api_key_display()))
    L.append("Router 模式下**所有模型共用 8082**，靠模型 ID 切换，不需要换端口。")
    L.append("")
    L.append("## 文件编码约定（改错会乱码或闪退）")
    L.append("")
    L.append("| 文件 | 编码 | BOM | 说明 |")
    L.append("|---|---|---|---|")
    L.append("| `llama-hub.bat` | ASCII | 无 | 纯 ASCII，无 `chcp`，任何代码页都不会乱 |")
    L.append("| `launcher/update-launchers.bat` | ASCII | 无 | 同上 |")
    L.append("| `launcher/models-config.bat` | GBK(936) | 无 | 含中文 `REM` 注释，"
             "**不要转成 UTF-8** |")
    L.append("| `launcher/start-Gemma4-Launcher.bat` | GBK(936) | 无 | 中文界面文案 |")
    L.append("| `launcher/start-Qwen-Launcher.bat` | GBK(936) | 无 | 中文界面文案 |")
    L.append("| `launcher/start-CPU-Toolcall-Launcher.bat` | ASCII | 无 | 纯 ASCII |")
    L.append("| `launcher/*.json` | UTF-8 | 无 | 无 BOM |")
    L.append("| `D:/dev/models/chat/models-config.ini` | UTF-8 | 无 | 中文 `#` 注释 |")
    L.append("")
    L.append("## 事实源优先级")
    L.append("")
    L.append("想改东西时，先在这张表里找到「该改哪一层」，不要直接改生成物。")
    L.append("")
    L.append("| 层级 | 文件 | 修改方式 | 会被谁覆盖 |")
    L.append("|---:|---|---|---|")
    L.append("| 1 数据源 | `preset-overrides.json` | 手工编辑 | 无人覆盖，**推荐改这里** |")
    L.append("| 2 数据源 | `model-profiles.json` | 手工编辑 | 无人覆盖 |")
    L.append("| 3 数据源 | 磁盘上的 gguf | 增删模型目录 | 无人覆盖 |")
    L.append("| 4 生成物 | `start-*.bat` | 不要手改 | `update_launchers.py` |")
    L.append("| 5 生成物 | `models-config.ini` | 可加自定义参数键 | 只覆盖 6 个受管键 |")
    L.append("| 6 缓存 | `.llama-server-keys.json` | 自动 | `--help` 探测结果 |")
    L.append("| 7 历史 | `backup/` | 只读 | 每次写入前追加 |")
    L.append("")
    L.append("## 常用命令")
    L.append("")
    L.append("```bat")
    L.append("llama-hub.bat                     :: 交互式菜单")
    L.append("llama-hub.bat --guide             :: 看模型表")
    L.append("llama-hub.bat --diagnose          :: 环境体检")
    L.append("llama-hub.bat --check             :: 预览差异，不改文件")
    L.append("llama-hub.bat --yes               :: 正式写入")
    L.append("llama-hub.bat --audit             :: 参数/显存审计")
    L.append("llama-hub.bat --ini-diff          :: ini 变化明细")
    L.append("llama-hub.bat --validate-drafts   :: 实测所有 MTP 草稿")
    L.append("llama-hub.bat --tune-sweep all    :: n-cpu-moe 阶梯扫描")
    L.append("llama-hub.bat --write-docs        :: 重新生成本目录文档")
    L.append("llama-hub.bat --paths             :: 打印所有解析后的路径")
    L.append("```")
    L.append("")
    L.append("## 当前清单")
    L.append("")
    L.append("- 模型条目（含 MTP 变体）: **{}**".format(len(rows)))
    L.append("- MTP 草稿文件: **{}**".format(len(mtp_files)))
    L.append("- 已列入黑名单的草稿: **{}**".format(len(state["blacklist"])))
    L.append("")
    return "\n".join(L) + "\n"


def cmd_write_docs():
    state = load_state()
    head("生成文档")
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, text in (("MODELS.md", write_models_md(state)),
                       ("launcher-architecture.md", write_arch_md(state))):
        p = DOCS / name
        data = text.encode("utf-8")
        if p.exists() and p.read_bytes() == data:
            print("  [=] {} 无变化".format(p))
            continue
        p.write_bytes(data)
        print("  [w] {}  ({} B)".format(p, len(data)))
    print()
    print("  其余文档（手工维护，不会被覆盖）：")
    for f in sorted(DOCS.glob("*")):
        if f.name not in ("MODELS.md", "launcher-architecture.md"):
            print("    {}".format(f.name))


# --------------------------------------------------- tuning guard rails
# Moved into update_launchers so the audit and the tune menu share one
# implementation and one set of coefficients. These are aliases, not copies.
TUNE_MIN_TPS = U.TUNE_MIN_TPS
TUNE_MIN_HEADROOM_MIB = U.TUNE_MIN_HEADROOM_MIB

TUNABLE = [
    ("ctx-size",     "上下文长度",   "int"),
    ("n-cpu-moe",    "MoE 卸载层数", "int"),
    ("cache-type-k", "KV 量化 K",   "kv"),
    ("cache-type-v", "KV 量化 V",   "kv"),
    ("batch-size",   "批尺寸",       "int"),
]


def vram_estimate(entry, n_cpu_moe, ctx, kv_k="q8_0"):
    """See update_launchers.vram_estimate - one implementation, shared."""
    return U.vram_estimate(entry, n_cpu_moe, ctx, kv_k)


def vram_verdict(entry, params, ctx=None, n=None, kv_k=None):
    """See update_launchers.vram_verdict."""
    return U.vram_verdict(entry, params, ctx=ctx, n=n, kv_k=kv_k)


def retag_ctx(model_id, ctx):
    """Rewrite the `NNNK` context token inside a model id. None when absent.

The id is written once, when the model is first scanned, and nothing used to
put it back in step afterwards - so changing ctx-size to 262144 left the model
listed as `...-128K-...`, which contradicts its own configuration and is
exactly the kind of mismatch that makes a picker untrustworthy.

    `Q5 Qwen3.6-35B-A3B-64K-MoE-IMG-UNC-Hau` + 262144
      -> `Q5 Qwen3.6-35B-A3B-256K-MoE-IMG-UNC-Hau`

    Guards against the other numbers in an id: `35B`, `A3B`, `0.6B` and `F16`
    never match because the token must be digits followed by `K` and then a
    non-word character.
    """
    if not model_id or not ctx:
        return None
    token = "{}K".format(int(ctx) // 1024)
    new, n = re.subn(r"(?<![\w.])\d+K(?![\w])", token, model_id, count=1)
    return new if n else None


def _gguf_meta(rel):
    d = Path(U.DEFAULT_CHAT) / rel.replace("/", os.sep)
    if not d.is_dir():
        return {}
    cands = sorted(f for f in d.glob("*.gguf")
                   if "mmproj" not in f.name.lower())
    return U.read_gguf_meta(str(cands[0])) if cands else {}


def _eff(params, g, key, default=None):
    """Effective value of a tunable: model override wins, then global."""
    v = params.get(key)
    if v is None:
        v = (g or {}).get(key)
    return default if v is None else v


def act_tune_params(state=None, preview_only=False):
    """Interactive per-model parameter editor.

    Exists because the only way to change, say, a context length used to be
    hand-editing launcher/preset-overrides.json - which is a reasonable thing to
    ask of whoever wrote the generator and an unreasonable thing to ask of
    anybody else. Writes ONLY that json (the generated .bat / .ini are rebuilt
    afterwards), so the existing architecture is untouched.

    Every edit is pre-checked against the two guard rails decided on
    2026-09-13: >= 35 t/s stable decode and > 3 GB VRAM headroom.
    """
    state = state or load_state()
    models = state["models"]
    g = state.get("global") or {}
    rels = sorted(models)

    while True:
        head("参数速调  -  不手改 json，直接改参数")
        print("  判据：稳定解码 >= {:.0f} t/s   且   显存余量 > {:.1f} GB".format(
            TUNE_MIN_TPS, TUNE_MIN_HEADROOM_MIB / 1024.0))
        print("  （余量要求来自实测：视觉请求会额外吃掉 ~500 MiB 的 CUDA graph")
        print("    缓存且不释放，浏览器/桌面还会再抢一部分）")
        print()
        print("  {:<3}{:<33}{:>6} {:>3} {:>10} {:>11}  {}".format(
            "#", "模型", "ctx", "n", "预计显存", "余量", ""))
        hr()
        for i, rel in enumerate(rels, 1):
            e = models[rel]
            p = e.get("params") or {}
            ctx = int(_eff(p, g, "ctx-size", 65536))
            n = _eff(p, g, "n-cpu-moe")
            est, headroom, ok = vram_verdict(e, p)
            if est is None:
                est_s, head_s, mark = "?", "无法预估", "  "
            else:
                est_s = "{:,}".format(round(est))
                head_s = "{:+,}".format(round(headroom))
                mark = "OK" if ok else "!!"
            mid = str(e.get("id") or rel)
            print("  {:<3}{:<33}{:>6} {:>3} {:>10} {:>11}  {}".format(
                i, mid[:33], ctx, n if n is not None else "-", est_s,
                head_s, mark))
        print()
        if preview_only:
            return
        print("  0   返回主菜单")
        hr()
        c = ask("  选模型 [1-{}] : ".format(len(rels))).strip()
        if c in ("0", "", "q"):
            return
        if not c.isdigit() or not (1 <= int(c) <= len(rels)):
            print("  无效编号。"); pause(); continue
        rel = rels[int(c) - 1]
        _edit_one_model(state, rel, g)


def _edit_one_model(state, rel, g):
    models = state["models"]
    while True:
        e = models[rel]
        p = e.setdefault("params", {})
        meta = _gguf_meta(rel)
        ctx_train = int(meta.get("ctx_train") or 0)
        layers = int(meta.get("layers") or 0)
        est, headroom, ok = vram_verdict(e, p)

        head("参数速调 · {}".format(str(e.get("id") or rel)[:56]))
        print("  目录      : {}".format(rel))
        print("  模型上限  : ctx {}  层数 {}  专家 {}".format(
            ctx_train or "?", layers or "?", meta.get("experts") or "?"))
        if est is None:
            print("  显存预估  : 无标定数据（改完仍需实测）")
        else:
            print("  当前预计  : {:,} MiB   余量 {:+,} MiB  [{}]".format(
                round(est), round(headroom), "OK" if ok else "!! 低于 "
                "{:.1f} GB".format(TUNE_MIN_HEADROOM_MIB / 1024.0)))
        tps = _tps_now(e, p)
        if tps:
            print("  最近实测  : {:.1f} t/s  {}".format(
                tps, "OK" if tps >= TUNE_MIN_TPS else
                "!! 低于 {:.0f} t/s".format(TUNE_MIN_TPS)))
        print()
        print("   #   参数              说明            当前值")
        for i, (key, label, kind) in enumerate(TUNABLE, 1):
            cur = p.get(key)
            if cur is None:
                shown = "{}  (继承全局)".format(_eff(p, g, key, "未设置"))
            elif g.get(key) == cur:
                shown = "{}  (=全局)".format(cur)
            else:
                shown = str(cur)
            print("   {:<3} {:<17} {:<14} {}".format(i, key, label, shown))
        print("   9   温度 / top-p / top-k 等采样参数")
        print("   8   清除某项的本地覆盖（回退到全局）")
        print("   7   改完 → 立即重建启动器与 Router 配置")
        print("   0   返回模型列表")
        hr()
        c = ask("  改哪个？ : ").strip().lower()

        if c in ("0", ""):
            return
        if c == "7":
            _rebuild_after_tune(); pause(); continue
        if c == "9":
            _edit_sampling(state, rel); continue
        if c == "8":
            keys = [k for k, _, _ in TUNABLE]
            print("  可清除的本地覆盖：")
            for i, k in enumerate(keys, 1):
                mark = "有" if k in p else "-"
                print("    {}) {}   [{}]".format(i, k, mark))
            s = ask("  清除哪个 [1-{}] : ".format(len(keys))).strip()
            if s.isdigit() and 1 <= int(s) <= len(keys):
                k = keys[int(s) - 1]
                if k in p:
                    p.pop(k)
                    U.save_overrides(state["overrides"])
                    print("  [OK] 已清除 {} 的本地覆盖".format(k))
                else:
                    print("  该项本来就没有本地覆盖。")
            pause(); continue
        if not c.isdigit() or not (1 <= int(c) <= len(TUNABLE)):
            print("  无效编号。"); pause(); continue

        key, label, kind = TUNABLE[int(c) - 1]
        _edit_one_param(state, rel, g, key, label, kind, meta, p)
        pause()


def _edit_one_param(state, rel, g, key, label, kind, meta, p):
    ctx_train = int(meta.get("ctx_train") or 0)
    layers = int(meta.get("layers") or 0)
    cur = _eff(p, g, key)
    cur_s = "未设置" if cur is None else str(cur)

    if kind == "kv":
        opts = ["q8_0", "q4_0", "f16"]
        print()
        print("  {} 可选：{}".format(key, " / ".join(opts)))
        print("    q8_0 = 默认，质量最好；q4_0 ≈ 一半显存，长上下文才需要")
        v = ask("  新值 [当前 {}] : ".format(cur)).strip()
        if not v:
            return
        if v not in opts:
            print("  [X] 只支持 {}".format(" / ".join(opts))); return
        new = v
    else:
        lo, hi, hint = 1, 1 << 20, ""
        if key == "ctx-size":
            lo = 4096
            hi = ctx_train or 262144
            hint = "模型上限 {}".format(hi)
        elif key == "n-cpu-moe":
            lo, hi = 0, layers or 99
            hint = "0-{}（越大越省显存、越慢）".format(hi)
        elif key == "batch-size":
            lo, hi, hint = 32, 4096, "越小越省显存，512 是常用值"
        print()
        print("  {}  当前 {}   {}".format(key, cur_s, hint))
        v = ask("  新值 : ").strip()
        if not v:
            return
        if not v.lstrip("-").isdigit():
            print("  [X] 请输入整数。"); return
        new = int(v)
        if not (lo <= new <= hi):
            print("  [X] 超出范围 {}-{}".format(lo, hi)); return

    # ---- pre-check against the guard rails -------------------------------
    kw = {}
    if key == "ctx-size":
        kw["ctx"] = new
    elif key == "n-cpu-moe":
        kw["n"] = new
    elif key == "cache-type-k":
        kw["kv_k"] = new
    est, headroom, ok = vram_verdict(state["models"][rel], p, **kw)

    print()
    print("  即将写入 : {}  {} -> {}".format(key, cur_s, new))
    id_new, id_old = None, None
    if key == "ctx-size":
        id_old = str(state["models"][rel].get("id") or "")
        id_new = retag_ctx(id_old, new)
        if id_new and id_new != id_old:
            print("  名称同步 : {}  ->  {}".format(id_old, id_new))
            print("             （旧名称保留为 alias，不会被引用断裂）")
        else:
            print("  名称同步 : 这个 id 里没有 xxxK 片段，只能手动改")
    if est is not None and key != "batch-size":
        verdict = "OK" if ok else "!! 余量不足"
        print("  预计显存 : {:,} MiB   余量 {:+,} MiB  [{}]".format(
            round(est), round(headroom), verdict))
        if not ok:
            print("  ⚠ 这会让显存余量低于 {:.1f} GB。".format(
                TUNE_MIN_HEADROOM_MIB / 1024.0))
            print("    （实测视觉请求会再吃掉 ~500 MiB 且不释放）")
    if key == "n-cpu-moe" and layers:
        print("  专家层数 : {} / {} 层留在 GPU 上".format(layers - new, layers))
    if key == "batch-size":
        print("  注意     : 改 batch 的显存影响未标定，改完请实测确认余量。")
    print()
    if not confirm("  确认写入 preset-overrides.json ？", default=False):
        print("  已取消。"); return
    p[key] = new
    if id_new and id_new != id_old:
        e2 = state["models"][rel]
        e2["id"] = id_new
        al = e2.get("alias") or []
        if isinstance(al, str):
            al = [al]
        if id_old and id_old not in al:
            al.append(id_old)
        e2["alias"] = al
    U.save_overrides(state["overrides"])
    print("  [OK] 已写入 {} = {}".format(key, new))
    if id_new and id_new != id_old:
        print("  [OK] 模型名称已同步为 {}".format(id_new))
    if confirm("  现在重建启动器与 Router 配置？", default=True):
        _rebuild_after_tune()


def _edit_sampling(state, rel):
    e = state["models"][rel]
    p = e.setdefault("params", {})
    print()
    for k in ("temp", "top-p", "top-k", "repeat-penalty"):
        cur = p.get(k)
        if cur is None:
            continue
        v = ask("  {} [当前 {}] : ".format(k, cur)).strip()
        if not v:
            continue
        try:
            p[k] = float(v) if "." in v else int(v)
        except ValueError:
            print("  [X] {} 不是数字，跳过。".format(v))
    U.save_overrides(state["overrides"])
    print("  [OK] 采样参数已保存（未填写的保持不变）")


def _rebuild_after_tune():
    print()
    print("  正在重建启动器与 models-config.ini ...")
    try:
        r = subprocess.run(
            [sys.executable, str(BASE / "update_launchers.py"), "--yes"],
            cwd=str(ROOT), capture_output=True, text=True,
            errors="replace", timeout=1800)
    except Exception as exc:
        print("  [X] 重建失败：{}".format(exc)); return
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()][-6:]
    for ln in tail:
        print("    " + ln)
    if r.returncode != 0:
        print("  [X] 重建返回码 {}".format(r.returncode))
        for ln in (r.stderr or "").splitlines()[-8:]:
            print("    " + ln)
    else:
        print("  [OK] 重建完成。重启服务后生效。")


# ------------------------------------------------------------------ menu

def menu():
    state = load_state()
    rows, _ = model_rows(state)
    exe = U.find_server_exe()
    while True:
        head("llama-hub  -  llama.cpp 模型启动中心")
        print("  仓库根目录 : {}".format(ROOT))
        print("  模型目录   : {}  ({} 个模型 + {} 个 MTP 加速版)".format(
            U.DEFAULT_CHAT,
            len([r for r in rows if r["kind"] == "base"]),
            len([r for r in rows if r["kind"] == "mtp"])))
        print("  llama-server: {}".format(exe if exe else "!!! 未找到 !!!"))
        running = [str(p) for p, _, _ in PORTS if port_busy(p)]
        print("  正在运行   : {}".format(
            "端口 " + ", ".join(running) if running else "无"))
        hr()
        print()
        print("   1   更新模型清单 / 同步启动器（先预览再确认）")
        print("   2   启动 Router 模式 · 端口 8082 · ★推荐★")
        print("   3   启动 Gemma4 菜单 · 端口 8080（含图文/MTP 选项）")
        print("   4   启动 Qwen 菜单 · 端口 8084")
        print("   5   启动 纯 CPU 工具调用 · 端口 8086（无显卡也能跑）")
        print("   6   模型选择指南（该选哪个一看就懂）")
        print("   7   环境诊断（显卡/显存/端口/模型文件体检）")
        print("   8   参数调优 / 扫描（n-cpu-moe 阶梯实测）")
        print("   9   打开文档与 Web UI 地址")
        print("   10  启动 向量/嵌入服务 · 端口 8081")
        print("   11  参数速调（改上下文/MoE 卸载/KV 量化，不用碰 json）★")
        print("   0   退出")
        print()
        hr()
        choice = ask("  请输入编号 [2]:").strip().lower() or "2"
        try:
            if choice in ("0", "q", "exit", "quit"):
                print("\n  再见。")
                return 0
            if choice == "1":
                act_update(); pause()
            elif choice == "2":
                start_bat("router"); pause()
            elif choice == "3":
                start_bat("gemma4"); pause()
            elif choice == "4":
                start_bat("qwen"); pause()
            elif choice == "5":
                start_bat("cpu"); pause()
            elif choice == "6":
                cmd_guide(state)
                if confirm("\n  是否把这份表写成文档？", default=False):
                    cmd_write_docs()
                pause()
            elif choice == "7":
                cmd_diagnose(); pause()
            elif choice == "8":
                act_tune(); pause()
            elif choice == "9":
                act_docs(); pause()
            elif choice == "10":
                start_bat("embed"); pause()
            elif choice == "11":
                act_tune_params(state)
            else:
                print("  无效编号：{}".format(choice)); pause()
        except KeyboardInterrupt:
            print("\n  已中断。")


# ------------------------------------------------------------------- main

def main(argv=None):
    setup_console()
    argv = list(sys.argv[1:] if argv is None else argv)

    if any(a in HUB_FLAGS for a in argv):
        ap = argparse.ArgumentParser(
            prog="llama-hub",
            description="llama.cpp 启动中心。无参数时进入交互菜单。",
            epilog="其它任何参数都会原样转发给 update_launchers.py，例如:\n"
                   "  --check             预览差异，不改文件\n"
                   "  --yes               正式写入启动器与 ini\n"
                   "  --ini-diff          只看 preset 变化明细\n"
                   "  --extract           重新扫描磁盘并重建模型清单\n"
                   "  --derive-params     为新模型补默认参数\n"
                   "  --tune-sweep all    实测 n-cpu-moe 阶梯（慢，先关服务）\n"
                   "  --reps N            配合 --tune-sweep 指定重复次数\n"
                   "  --validate-drafts   逐个加载实测所有 MTP 草稿\n"
                   "  --blacklist-drafts  配合上面那条，把失败的草稿拉黑\n"
                   "  --fix-bodies        修复启动器里的空行断行\n"
                   "  --fix-moe           给 MoE 菜单条目补 --n-cpu-moe\n"
                   "  --fix-mmproj        按磁盘实际文件修正 mmproj 路径\n"
                   "  --accept-new        确认接受扫描到的新模型（DEC-001）\n"
                   "  --paths             打印所有解析后的路径\n"
                   "  --no-scan           不做磁盘扫描，按注册表现样渲染\n",
            formatter_class=argparse.RawDescriptionHelpFormatter)
        g = ap.add_mutually_exclusive_group()
        g.add_argument("--guide", action="store_true",
                       help="打印模型选择指南")
        g.add_argument("--diagnose", action="store_true",
                       help="环境体检（显卡/显存/端口/模型文件/草稿）")
        g.add_argument("--audit", action="store_true",
                       help="显存与参数审计：估算每个模型能否跑起来（只读）")
        g.add_argument("--write-docs", action="store_true",
                       help="重新生成 launcher\\docs\\MODELS.md 与 "
                            "launcher-architecture.md")
        ap.add_argument("--menu", action="store_true", help="进入交互菜单")
        ap.add_argument("--params", action="store_true",
                        help="参数速调：直接进入可交互的模型参数编辑器")
        ap.add_argument("--params-preview", action="store_true",
                        help="只打印参数速调表（含显存预估）后退出，不修改任何文件")
        args = ap.parse_args(argv)
        if args.params_preview:
            act_tune_params(preview_only=True)
            return 0
        if args.params:
            act_tune_params()
            return 0
        if args.guide:
            cmd_guide()
            return 0
        if args.diagnose:
            return 0 if cmd_diagnose() else 1
        if args.audit:
            return 0 if cmd_audit() else 1
        if args.write_docs:
            cmd_write_docs()
            return 0
        return menu()

    if argv:
        # anything else is an update_launchers.py option -> transparent forward
        return run_updater(argv)

    return menu()


if __name__ == "__main__":
    sys.exit(main())
