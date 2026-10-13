"""AgentRL 桌面版入口：启动本地聊天服务并自动打开浏览器。

用法:
    python desktop_app.py                # 开发环境运行
    AgentRL.exe                          # 打包后双击运行
    AgentRL.exe --no-browser             # 只启服务不开浏览器
    AgentRL.exe --port 8080              # 指定端口（默认 5000，被占用自动顺延）

打包说明见 packaging/build_exe.ps1。
"""

import multiprocessing  # noqa: F401  # Windows 冻结进程必须先导入（freeze_support 依赖）
import os
import socket
import sys
import threading
import time
import webbrowser


def _app_dir() -> str:
    """应用根目录：打包后为 exe 同级目录，开发环境为项目根。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _load_env(app_dir: str) -> None:
    """加载 exe/项目同级 .env（API 密钥等，不打进 exe）。"""
    env_path = os.path.join(app_dir, ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass


def _setup_hf_home(app_dir: str) -> None:
    """若构建时内置了 HuggingFace 模型缓存（hf_cache/），指向它实现离线检索。"""
    hf = os.path.join(app_dir, "hf_cache")
    if os.path.isdir(hf) and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = hf


def _pick_port(host: str, start: int, tries: int = 10) -> int:
    """从 start 起找一个可绑定端口。"""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start


def _wait_and_open(host: str, port: int, timeout: float = 60.0) -> None:
    """轮询 /health，服务就绪后打开浏览器。"""
    import urllib.request
    health = f"http://{host}:{port}/health"
    page = f"http://{host}:{port}/chat.html"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1):
                break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(page)


def _parse_port(argv) -> int:
    if "--port" in argv:
        try:
            return int(argv[argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass
    return int(os.environ.get("AGENTRL_PORT", "5000"))


def main() -> int:
    multiprocessing.freeze_support()

    app_dir = _app_dir()
    _load_env(app_dir)
    _setup_hf_home(app_dir)
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, app_dir)

    host = os.environ.get("AGENTRL_HOST", "127.0.0.1")
    port = _pick_port(host, _parse_port(sys.argv))

    if "--no-browser" not in sys.argv:
        threading.Thread(target=_wait_and_open, args=(host, port), daemon=True).start()

    import uvicorn
    from chat_server import app

    print(f"[AgentRL] 服务启动: http://{host}:{port}/chat.html")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
