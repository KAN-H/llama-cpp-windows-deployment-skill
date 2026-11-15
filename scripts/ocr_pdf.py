#!/usr/bin/env python3
"""ocr_pdf.py - 用 llama.cpp 上的视觉 OCR 模型（如 GLM-OCR）逐页 OCR 整本 PDF。

为什么需要它：llama.cpp 自带 WebUI **上传整本 PDF 时会先把每一页渲染成图片、再全部塞进
同一条 chat 请求**，于是同时撞两堵墙：
  * llama-server 请求体硬上限 **100 MiB**（编译期常量，无参数可调）—— 超了回 413 并直接关连接，
    浏览器侧只会看到 "Unable to connect to server"；
  * 视觉模型 ctx 有限（GLM-OCR 常见配置 16K ≈ 12 页）。
本脚本改为分页请求，并按页落盘、可断点续跑。

用法：
  python scripts/ocr_pdf.py <book.pdf>                     # 逐页 OCR（断点续跑）
  python scripts/ocr_pdf.py <book.pdf> --from 1 --to 20
  python scripts/ocr_pdf.py <book.pdf> --dry-run            # 只看计划
  python scripts/ocr_pdf.py <book.pdf> --batch 8            # ⚠ 批量会漏页，见下

环境：
  LLAMA_API_KEY   服务端 --api-key（也可用 --api-key）
  LLAMA_API       chat/completions 地址，默认 http://127.0.0.1:8080/v1/chat/completions

依赖：pymupdf  （pip install pymupdf）—— 仅标准库以外这一个依赖。

⚠️ --batch > 1 实测不可靠：模型不输出页分隔符，且会漏页（曾 20 页只回 316 token）。
   默认 --batch 1 是唯一能保证「一页一文件」的模式。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API = os.environ.get("LLAMA_API", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_PROMPT = "Text Recognition:"
PAYLOAD_BUDGET = 90 * 1024 * 1024      # 服务端硬上限 100 MiB，留 10 MiB 余量


def die(msg: str, code: int = 1):
    print(f"[x] {msg}", file=sys.stderr)
    raise SystemExit(code)


def http_json(url: str, key: str, payload: dict | None = None, timeout: int = 600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw or "(empty body - 很可能是请求体超 100 MiB 的 413)"
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def root_of(api: str) -> str:
    return api.rsplit("/v1/", 1)[0]


def pick_model(api: str, key: str, want: str | None) -> str:
    if want:
        return want
    status, data = http_json(root_of(api) + "/v1/models", key)
    ids = []
    if status == 200 and isinstance(data, dict):
        ids = [d.get("id", "") for d in data.get("data", [])]
    for i in ids:
        if "ocr" in i.lower():
            return i
    die(f"无法自动识别 OCR 模型，请用 --model 指定。可用：{ids}")


def server_ctx(api: str, key: str, model: str) -> int:
    """Router 模式下 /props 的 n_ctx 恒为 0，故优先读 /v1/models 的 meta / status.args。"""
    root = root_of(api)
    status, data = http_json(root + "/props", key)
    if status == 200 and isinstance(data, dict):
        n = int((data.get("default_generation_settings") or {}).get("n_ctx") or 0)
        if n > 0:
            return n
    status, data = http_json(root + "/v1/models", key)
    if status != 200 or not isinstance(data, dict):
        return 0
    for m in data.get("data", []):
        if m.get("id") != model:
            continue
        n = int((m.get("meta") or {}).get("n_ctx") or 0)
        if n > 0:
            return n
        args = (m.get("status") or {}).get("args") or []
        for i, a in enumerate(args):
            if a in ("--ctx-size", "-c") and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    pass
    return 0


def open_pdf(path: Path):
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf      # 旧版本 PyMuPDF
        except ImportError:
            die("缺少 pymupdf：pip install pymupdf")
    return pymupdf.open(path), pymupdf


def render_page(doc, pymupdf, index0: int, scale: float) -> bytes:
    pix = doc[index0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png")


def ocr_once(api, key, model, images, prompt, max_tokens, timeout):
    content = [{"type": "text", "text": prompt}]
    for png in images:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    payload = {"model": model,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "stream": False, "temperature": 0.0}
    t0 = time.time()
    status, data = http_json(api, key, payload, timeout=timeout)
    return status, data, time.time() - t0


def is_ctx_error(status: int, data) -> bool:
    if status != 400:
        return False
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return "exceed_context_size_error" in text or "exceeds the available context size" in text


def extract(data):
    if not isinstance(data, dict):
        return "", "", 0, 0
    ch = (data.get("choices") or [{}])[0]
    u = data.get("usage") or {}
    return ((ch.get("message") or {}).get("content") or "", ch.get("finish_reason") or "",
            int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))


def main() -> int:
    ap = argparse.ArgumentParser(description="逐页 OCR 整本 PDF（llama.cpp /v1/chat/completions）")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="输出目录（默认 <pdf目录>/<pdf名>_ocr）")
    ap.add_argument("--from", dest="start", type=int, default=1, help="起始页（1 起，含）")
    ap.add_argument("--to", dest="end", type=int, default=0, help="结束页（含；0=末页）")
    ap.add_argument("--batch", type=int, default=1,
                    help="每请求页数（默认 1 最稳；>1 实测会漏页且无页边界）")
    ap.add_argument("--scale", type=float, default=1.5, help="渲染缩放（WebUI 用 1.5）")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    ap.add_argument("--model", default=None, help="模型 ID（默认自动挑含 ocr 的）")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--tokens-per-page", type=int, default=1350, help="每页 prompt token 估计（限批用）")
    ap.add_argument("--ctx", type=int, default=0, help="手动指定上下文（默认自动探测）")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="忽略已有结果重跑")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.pdf.exists():
        die(f"PDF 不存在：{a.pdf}")
    out_dir = a.out or (a.pdf.parent / f"{a.pdf.stem}_ocr")
    out_dir.mkdir(parents=True, exist_ok=True)

    doc, pymupdf = open_pdf(a.pdf)
    total = doc.page_count
    end = a.end or total
    if a.start < 1 or end > total or a.start > end:
        die(f"页范围非法：{a.start}-{end}（共 {total} 页）")

    model = pick_model(a.api, a.api_key, a.model)
    ctx = a.ctx or server_ctx(a.api, a.api_key, model)
    batch = max(1, a.batch)
    if ctx:
        fit = max(1, (ctx - a.max_tokens - 64) // max(1, a.tokens_per_page))
        if batch > fit:
            print(f"[i] 批大小 {batch} 超出 ctx 预算，自动降为 {fit}")
            batch = fit

    print(f"[i] PDF   : {a.pdf}  ({total} 页)")
    print(f"[i] 输出  : {out_dir}")
    print(f"[i] API   : {a.api}")
    print(f"[i] model : {model}   ctx={ctx or '?'}   batch={batch}   scale={a.scale}")
    if batch > 1:
        print("[!] batch>1：模型不输出页分隔符且会漏页，结果的页边界不可靠；要逐页请用 --batch 1")
    if a.dry_run:
        return 0

    todo = [p for p in range(a.start, end + 1)
            if a.force or not (out_dir / f"{p:04d}.txt").exists()]
    print(f"[i] 待处理: {len(todo)} 页")

    ok = fail = 0
    t0 = time.time()
    i = 0
    while i < len(todo):
        images, size, keep = [], 0, []
        for p in todo[i:i + batch]:
            png = render_page(doc, pymupdf, p - 1, a.scale)
            if images and size + len(png) * 4 / 3 > PAYLOAD_BUDGET:
                break
            images.append(png)
            size += len(png) * 4 / 3
            keep.append(p)

        attempt = 0
        while True:
            status, data, dt = ocr_once(a.api, a.api_key, model, images, a.prompt,
                                        a.max_tokens, a.timeout)
            if is_ctx_error(status, data) and len(keep) > 1:
                keep = keep[:max(1, len(keep) // 2)]
                images = images[:len(keep)]
                print(f"[!] ctx 超限，批大小降为 {len(keep)} 重试")
                continue
            if status == 0 or status >= 500:
                attempt += 1
                if attempt > a.retries:
                    break
                print(f"[!] p{keep[0]}-{keep[-1]} 失败({status})：{str(data)[:160]}，重试 {attempt}")
                time.sleep(min(10, 2 * attempt))
                continue
            break

        text, finish, ptok, ctok = extract(data)
        tag = f"{keep[0]}" if len(keep) == 1 else f"{keep[0]}-{keep[-1]}"
        if status == 200 and text.strip():
            name = f"{keep[0]:04d}.txt" if len(keep) == 1 else f"{keep[0]:04d}-{keep[-1]:04d}.txt"
            (out_dir / name).write_text(text, encoding="utf-8")
            ok += 1
            print(f"[+] p{tag:<9} {dt:5.1f}s  in={ptok:<6} out={ctok:<5}"
                  + ("  [截断!]" if finish == "length" else ""))
        else:
            fail += 1
            print(f"[x] p{tag:<9} status={status} {str(data)[:300]}")
        i += len(keep)

    if not a.no_merge:
        parts = sorted(out_dir.glob("*.txt"))
        merged = out_dir.parent / f"{a.pdf.stem}_ocr.md"
        if parts:
            with merged.open("w", encoding="utf-8") as f:
                f.write(f"# {a.pdf.stem}\n")
                for p in parts:
                    f.write(f"\n\n## {p.stem}\n\n" + p.read_text(encoding="utf-8").strip())
            print(f"[i] 合并 -> {merged}")

    print(f"\n[=] 完成 {ok} 组 / 失败 {fail} 组，用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
