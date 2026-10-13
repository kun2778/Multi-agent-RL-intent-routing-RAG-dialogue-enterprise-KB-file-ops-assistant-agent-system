"""
统一运行时路径配置模块。

所有运行时文件（缓存、数据库、FAISS索引、日志、会话、工作区产物）
都通过此模块的路径变量派生，不再硬编码散落在 ~/.agentrl/、%TEMP%/ 等位置。

可通过环境变量 AGENTRL_RUNTIME_DIR 覆盖运行时根目录。
"""

import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    """是否运行在 PyInstaller 打包环境。"""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """应用根目录：开发环境为项目根；打包后为 exe 同级目录（可写，便于模型/配置随安装分发与更新）。"""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _default_runtime_root() -> Path:
    """运行时根目录默认值：开发环境为项目根/.runtime；打包后为 %LOCALAPPDATA%\\AgentRL\\.runtime。"""
    if _is_frozen():
        base = os.getenv("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "AgentRL" / ".runtime"
    return app_dir() / ".runtime"


def models_dir() -> Path:
    """模型权重目录：可用环境变量 AGENTRL_MODELS_DIR 覆盖。
    打包后优先 exe 同级 models/（可写、可热更新）；缺失时回退到内置 _MEIPASS/models。"""
    env = os.getenv("AGENTRL_MODELS_DIR")
    if env:
        return Path(env)
    if _is_frozen():
        exe_models = Path(sys.executable).resolve().parent / "models"
        if exe_models.is_dir():
            return exe_models
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "models"
        if bundled.is_dir():
            return bundled
        return exe_models
    return app_dir() / "models"


# 项目/应用根目录（此文件位于项目根目录下，向上一级即项目根；打包后为 exe 同级目录）
PROJECT_ROOT = app_dir()

# 运行时根目录，可通过环境变量覆盖
RUNTIME_ROOT = Path(os.getenv("AGENTRL_RUNTIME_DIR", str(_default_runtime_root())))

# 子目录
TEMP_DIR = RUNTIME_ROOT / "temp"
CACHE_DIR = RUNTIME_ROOT / "cache"
LOG_DIR = RUNTIME_ROOT / "logs"
DB_DIR = RUNTIME_ROOT / "db"
FAISS_DIR = RUNTIME_ROOT / "faiss"
WORKSPACE_DIR = RUNTIME_ROOT / "workspace"
CONVERSATIONS_DIR = RUNTIME_ROOT / "conversations"
EXPORT_DIR = RUNTIME_ROOT / "exports"
BACKUP_DIR = RUNTIME_ROOT / "backups"


def ensure_runtime_dirs():
    """确保所有运行时子目录存在。"""
    for d in [TEMP_DIR, CACHE_DIR, LOG_DIR, DB_DIR, FAISS_DIR,
              WORKSPACE_DIR, CONVERSATIONS_DIR, EXPORT_DIR, BACKUP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_tempfile_dir():
    """将 Python tempfile 默认目录设为 TEMP_DIR。"""
    import tempfile
    tempfile.tempdir = str(TEMP_DIR)
