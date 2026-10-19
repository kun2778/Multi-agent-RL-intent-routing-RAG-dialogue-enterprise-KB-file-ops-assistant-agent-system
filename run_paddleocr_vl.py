"""
PaddleOCR-VL (FP16) 本地 OCR 启动器
用法:
  python run_paddleocr_vl.py <图片路径>          # 识别单张图
  python run_paddleocr_vl.py <目录>              # 识别目录下所有图
说明:
  - 模型权重已从 BF16 转为 FP16 (Turing 架构只支持 FP16)
  - GTX 1660 Ti 6GB 显存不足以跑该 VLM pipeline，默认走 CPU
  - 若你有 >=8GB 显存的卡, 把下面 DEVICE 改为 "gpu" 即可加速数十倍
"""
import os, sys, time, json, glob

# --- 必须在 import paddle 前设置 ---
# 复用全局 torch 自带的 CUDA/cuDNN 9 DLL (同 cuDNN 9.x, ABI 兼容)
_torch_lib = r"C:\Program Files\Python312\Lib\site-packages\torch\lib"
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

DEVICE = "cpu"  # 改为 "gpu" 需 >=8GB 显存

from paddleocr import PaddleOCRVL

from paddle_ocr_worker import extract_blocks

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]

    if os.path.isdir(target):
        files = [f for f in glob.glob(os.path.join(target, "*")) if f.lower().endswith(IMG_EXTS)]
    else:
        files = [target]
    if not files:
        print("未找到图片文件:", target)
        sys.exit(1)

    print(f"设备: {DEVICE} | 共 {len(files)} 张图 | 加载模型中(首次较慢)...", flush=True)
    t0 = time.time()
    pipe = PaddleOCRVL(pipeline_version="v1", device=DEVICE)
    print(f"模型加载完成, 耗时 {time.time()-t0:.1f}s\n", flush=True)

    for fp in files:
        t0 = time.time()
        for res in pipe.predict(fp):
            dt = time.time() - t0
            items = extract_blocks(res.json)
            print("=" * 60)
            print(f"文件: {os.path.basename(fp)}  |  识别块: {len(items)}  |  耗时: {dt:.1f}s")
            print("=" * 60)
            for it in items:
                print(f"[{it['label']}] {it['text']}")
            print()


if __name__ == "__main__":
    main()
