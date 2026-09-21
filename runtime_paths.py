"""
统一运行时路径配置模块。

所有运行时文件（缓存、数据库、FAISS索引、日志、会话、工作区产物）
都通过此模块的路径变量派生，不再硬编码散落在 ~/.agentrl/、%TEMP%/ 等位置。

可通过环境变量 AGENTRL_RUNTIME_DIR 覆盖运行时根目录。
"""

import os
from pathlib import Path

# 项目根目录（此文件位于项目根目录下，向上一级即项目根）
PROJECT_ROOT = Path(__file__).resolve().parent

# 运行时根目录，可通过环境变量覆盖
RUNTIME_ROOT = Path(os.getenv("AGENTRL_RUNTIME_DIR", str(PROJECT_ROOT / ".runtime")))

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
