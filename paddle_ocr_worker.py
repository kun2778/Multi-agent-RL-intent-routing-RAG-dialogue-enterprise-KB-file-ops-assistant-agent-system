"""PaddleOCR-VL 常驻识别 worker。

在 venv-paddle 环境中运行，模型只加载一次，通过 stdin/stdout 的 JSON-lines 协议复用。

协议（每行一个 JSON）：
  请求: {"image_path": "..."}  或  {"image_path": "...", "image_b64": "..."}
  响应: {"ok": true, "text": "...", "confidence": 0.97, "blocks": [...], "time_ms": 123}
        {"ok": false, "error": "..."}

特殊命令：{"cmd": "ping"} -> {"ok": true, "pong": true}；{"cmd": "quit"} 退出。
"""
import os
import sys
import json
import time
import io

# --- 必须在 import paddle 前设置 DLL 与环境 ---
_TORCH_LIB = r"C:\Program Files\Python312\Lib\site-packages\torch\lib"
if os.path.isdir(_TORCH_LIB):
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# CPU 是 6GB 显存卡能稳定跑通的模式；有 >=8GB 显存可改 "gpu"
DEVICE = os.environ.get("PADDLE_OCR_DEVICE", "cpu")


def _log(msg):
    # 走 stderr，避免污染 stdout 的 JSON 协议
    print(msg, file=sys.stderr, flush=True)


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def extract_blocks(res_json):
    blocks = res_json.get("res", {}).get("parsing_res_list", []) or []
    out = []
    for b in blocks:
        content = (b.get("block_content") or "").strip()
        if content:
            out.append({
                "label": b.get("block_label"),
                "text": content,
                "bbox": b.get("block_bbox"),
            })
    return out


def main():
    _log(f"[paddle-worker] device={DEVICE}, loading pipeline...")
    t_load = time.time()
    from paddleocr import PaddleOCRVL
    pipe = PaddleOCRVL(pipeline_version="v1", device=DEVICE)
    _log(f"[paddle-worker] loaded in {time.time()-t_load:.1f}s, ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"ok": False, "error": f"bad json: {e}"})
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            _emit({"ok": True, "bye": True})
            break
        if cmd == "ping":
            _emit({"ok": True, "pong": True})
            continue

        image_path = req.get("image_path")
        if not image_path or not os.path.exists(image_path):
            _emit({"ok": False, "error": f"image not found: {image_path}"})
            continue

        t0 = time.time()
        try:
            result_text = ""
            blocks = []
            for res in pipe.predict(image_path):
                blocks = extract_blocks(res.json)
                result_text = "\n".join(b["text"] for b in blocks)
            conf = _estimate_conf(blocks)
            _emit({
                "ok": True,
                "text": result_text,
                "confidence": conf,
                "blocks": blocks,
                "time_ms": int((time.time() - t0) * 1000),
            })
        except Exception as e:
            _log(f"[paddle-worker] error: {e}")
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _estimate_conf(blocks):
    """PaddleOCR-VL 不返回置信度，用有内容的块数做启发式（有文字=高置信）。"""
    if not blocks:
        return 0.0
    return 0.95


if __name__ == "__main__":
    main()
