# -*- mode: python ; coding: utf-8 -*-
"""AgentRL 桌面版 PyInstaller 打包配置（onedir 模式）。

设计要点：
- onedir：启动快（1-2s），比 onefile 避免每次自解压几十 GB 依赖
- torch / faiss / sentence_transformers 用 collect_all 全量收集（含原生 DLL）
- 排除云端/密钥/Vault 等桌面版用不到的重依赖，控制体积
- config/、意图语料 CSV、聊天页面作为数据文件打入，运行时用 app_dir()/_MEIPASS 定位
- 统一打包 VC++ 运行库，避免目标机缺 VCRUNTIME140.dll
- UPX 默认排除 torch/cuBLAS 类 DLL（压缩后易崩）
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH 由 PyInstaller 注入 = spec 文件所在目录（packaging/），项目根为上一级
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# --------------------------------------------------------------- 数据文件
datas = [
    (os.path.join(ROOT, "config"), "config"),
    (os.path.join(ROOT, "file_control.html"), "."),
    (os.path.join(ROOT, "intent_corpus.csv"), "."),
    (os.path.join(ROOT, "datasets", "intent_corpus.csv"), "datasets"),
]
_bulk = os.path.join(ROOT, "datasets", "intent_corpus_bulk.csv")
if os.path.exists(_bulk):
    datas.append((_bulk, "datasets"))

# 可选：内置 HuggingFace 模型缓存（离线环境构建时手动放置 hf_cache/）
_hf = os.path.join(ROOT, "hf_cache")
if os.path.isdir(_hf):
    datas.append((_hf, "hf_cache"))

# 可选：已训练模型权重（约 100MB；不放则目标机首次运行随机初始化并在线学习）
_models = os.path.join(ROOT, "models")
if os.path.isdir(_models):
    for _f in ("agent_rl.pth", "marl_agents.pth"):
        _p = os.path.join(_models, _f)
        if os.path.exists(_p):
            datas.append((_p, "models"))

# --------------------------------------------------------------- 原生库收集
binaries = []
hiddenimports = [
    # uvicorn 按需加载的子模块（PyInstaller 静态分析常漏）
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # asyncio / multiprocessing 冻结支持
    "asyncio",
    "multiprocessing.spawn",
    "multiprocessing.forkserver",
    "multiprocessing.resource_tracker",
]

for pkg in ("torch", "faiss", "sentence_transformers", "transformers",
            "tokenizers", "huggingface_hub", "safetensors", "tiktoken",
            "sklearn", "scipy"):
    try:
        _d, _b, _h = collect_all(pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception as e:  # 缺包时跳过，构建日志可见
        print(f"[spec] collect_all({pkg}) skipped: {e}")

hiddenimports += collect_submodules("yaml")
hiddenimports += collect_submodules("dotenv")
hiddenimports += collect_submodules("httpx")
hiddenimports += collect_submodules("openai")
hiddenimports += collect_submodules("bs4")
hiddenimports += collect_submodules("lxml")
hiddenimports += collect_submodules("pypdf")
hiddenimports += collect_submodules("requests")

# --------------------------------------------------------------- 排除项
# 桌面聊天版用不到：GUI 学习框架 / 测试框架 / 云端密钥与 SDK / MCP 服务
excludes = [
    "tkinter", "test",
    "IPython", "jupyter", "jupyterlab", "notebook",
    "matplotlib", "numpy.f2py",  # sklearn 会拖 matplotlib，显式排除
    # Qt 绑定冲突（环境里同时存在 PyQt5/PySide6 时 PyInstaller 会直接报错）
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    # 云端/密钥（providers 里均为惰性导入，排除后运行时自动降级 env）
    "boto3", "botocore", "awscli",
    "azure", "azure.identity", "azure.keyvault",
    "google.generativeai", "google.ai",
    "anthropic",
    "hvac",
    "mcp", "mcp_server",  # MCP 审计服务是独立进程，不进桌面 exe
    # 训练/评测脚本（非运行时）
    "pytest", "hypothesis",
]

# --------------------------------------------------------------- VC++ 运行库
# 目标机常缺 VCRUNTIME140.dll / MSVCP140.dll；从本机 System32 一并打入
sysroot = os.environ.get("SystemRoot", r"C:\Windows")
for _dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
    _p = os.path.join(sysroot, "System32", _dll)
    if os.path.exists(_p):
        binaries.append((_p, "."))

# --------------------------------------------------------------- 分析与产物
a = Analysis(
    [os.path.join(ROOT, "desktop_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AgentRL",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # 保留控制台便于查看服务日志；要隐藏可改 False
    disable_windowed_traceback=False,
)

upx_excludes = [
    "vcruntime140.dll", "msvcp140.dll", "python3*.dll",
    "cudart*.dll", "cublas*.dll", "cudnn*.dll", "cufft*.dll",
    "curand*.dll", "cusolver*.dll", "cusparse*.dll",
    "nvrtc*.dll", "c10.dll", "torch*.dll", "fbgemm*.dll",
    "libiomp*.dll", "openblas*.dll",
]

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=upx_excludes,
    name="AgentRL",
)
