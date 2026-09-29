import json
import os
import re
import time
import subprocess
import winreg
from typing import Any, Dict, List, Optional, Tuple
from runtime_paths import TEMP_DIR, CACHE_DIR
from atomic_io import atomic_write_json, atomic_write_text
from write_queue import get_default_write_queue

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_HEX_DIGIT = re.compile(r"^[0-9]$")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _distinct_cjk(text: str, kw: str) -> float:
    t = {c for c in text if _is_cjk(c)}
    k = {c for c in kw if _is_cjk(c)}
    if not k:
        return 0.0
    return len(t & k) / len(k)


class SystemFileFinder:
    """Windows 本体文件/程序查找器。

    匹配原理：查询关键词 → 同时对比多字段（注册表名 / 快捷方式名 / exe文件名 / 目录名 / 路径），
    取最高匹配置信度作为综合置信度。即使 exe 文件名与描述不同（如 wemeetapp.exe vs 腾讯会议），
    也能通过快捷方式名或注册表 DisplayName 作为桥梁找到。

    置信度计算：每个字段独立打分 0.0～1.0，最终取 max。
      - 完全相等           → 1.0
      - 包含关键词子串     → 0.8
      - 多字符 token 命中  → 0.6
      - CJK 字符重合 >50%  → 0.4
      - CJK 字符重合 >25%  → 0.2
    """

    FIND_OPS = ("找到", "查找", "搜索", "找一下", "查一下", "帮找", "定位",
                "在哪", "哪里", "哪个盘", "安装位置", "安装目录", "快捷方式")
    OBJECT_WORDS = ("程序", "exe", "快捷方式", "安装", "应用", "软件", "文件")
    LEADING = ("为我", "请", "麻烦", "帮我", "帮忙", "给我", "立刻", "马上", "快点")
    EXE_RE = re.compile(r"[\u4e00-\u9fff\w][.\-]*\.(?:exe|lnk|msi|bat|cmd|com|scr|dll)\b", re.I)
    TRAILING = ("文件", "程序", "快捷方式", "应用", "软件", "的安装位置", "的安装目录",
                "安装位置", "安装目录", "位置", "路径", "目录", "在哪", "在哪儿",
                "哪里", "哪个盘", "启动程序", "根目录", "exe", ".exe", "lnk", ".lnk")
    PRUNE_DIRS = {"node_modules", ".git", ".svn", "__pycache__", ".cache", ".m2",
                  ".gradle", "Windows", "WinSxS", "Recovery", "$RECYCLE.BIN"}

    def __init__(self):
        self._lnk_cache: Optional[List[Dict]] = None
        self._lnk_ts = 0.0
        self._reg_cache: Optional[List[Dict]] = None
        self._reg_ts = 0.0
        self._pf_cache: Optional[List[Dict]] = None
        self._pf_ts = 0.0
        self._ttl = 300.0

    # ------------------------------------------------------------- 判定与提词
    def matches(self, query: str) -> bool:
        q = query or ""
        if self.EXE_RE.search(q):
            return True
        return any(w in q for w in self.FIND_OPS) and any(w in q for w in self.OBJECT_WORDS)

    def _extract_keyword(self, query: str) -> str:
        kw = re.sub(r"[\.,。，\s]+", "", query)
        for w in self.LEADING:
            kw = kw.replace(w, "")
        for w in ("找到", "查找", "搜索", "检索", "找出", "定位", "找一找",
                  "找一下", "查一下", "帮我找", "请帮我"):
            kw = kw.replace(w, "")
        for w in self.TRAILING:
            kw = kw.replace(w, "")
        for w in ("的", "和", "与", "还有", "什么", "那个"):
            kw = kw.replace(w, "")
        return kw.strip(" _:\u3000") or query

    # ------------------------------------------------------------- 置信度计算
    @staticmethod
    def _field_confidence(kw: str, field_value: str) -> float:
        if not kw or not field_value:
            return 0.0
        kw_low, fv_low = kw.lower(), field_value.lower()
        if kw_low == fv_low:
            return 1.0
        if kw_low in fv_low or fv_low in kw_low:
            return 0.8
        for tok in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", kw):
            if len(tok) >= 2 and tok.lower() in fv_low:
                return 0.6
        cjk_ratio = _distinct_cjk(field_value, kw)
        if cjk_ratio > 0.5:
            return 0.4
        if cjk_ratio > 0.25:
            return 0.2
        return 0.0

    def _compute_confidence(self, item: Dict, kw: str) -> float:
        fields = []
        for key in ("display_name", "exe_name", "exe_dir", "path", "target"):
            v = item.get(key, "")
            if v:
                fields.append((key, v))
        best = 0.0
        match_field, match_value = "", ""
        for field_key, field_val in fields:
            c = self._field_confidence(kw, field_val)
            if c > best:
                best, match_field, match_value = c, field_key, field_val
        item["_confidence"] = round(best, 2)
        item["_match_field"] = match_field
        item["_match_value"] = match_value
        return best

    # ------------------------------------------------------------- 数据源
    def _registry_entries(self) -> List[Dict]:
        now = time.time()
        if self._reg_cache is not None and now - self._reg_ts < self._ttl:
            return self._reg_cache
        entries = []
        roots = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        for root, sub in roots:
            try:
                with winreg.OpenKey(root, sub) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, name) as app:
                                display = self._reg_get(app, "DisplayName")
                                if not display or display.startswith("${{") or not any(c.isalnum() for c in display):
                                    continue
                                raw_path = (
                                    self._reg_get(app, "InstallLocation")
                                    or self._reg_get(app, "DisplayIcon")
                                    or ""
                                ).strip().rstrip(",").strip()
                                if raw_path and not os.path.exists(raw_path):
                                    raw_path = ""
                                base_entry = {
                                    "display_name": display, "path": raw_path,
                                    "source": "注册表", "lnk_name": "",
                                }
                                if raw_path and os.path.isdir(raw_path):
                                    exes = self._list_exes_in_dir(raw_path)
                                    for exe in exes:
                                        e = dict(base_entry)
                                        e["exe_name"] = os.path.basename(exe)
                                        e["exe_dir"] = os.path.basename(raw_path)
                                        e["path"] = exe
                                        e["target"] = ""
                                        entries.append(e)
                                    if not exes:
                                        e = dict(base_entry)
                                        e["exe_name"] = ""
                                        e["exe_dir"] = os.path.basename(raw_path)
                                        e["target"] = ""
                                        entries.append(e)
                                else:
                                    e = dict(base_entry)
                                    e["exe_name"] = ""
                                    e["exe_dir"] = ""
                                    e["target"] = ""
                                    entries.append(e)
                        except OSError:
                            continue
            except OSError:
                continue
        self._reg_cache = entries
        self._reg_ts = now
        return entries

    @staticmethod
    def _reg_get(key, value_name: str) -> str:
        try:
            v, _ = winreg.QueryValueEx(key, value_name)
            return str(v) if v else ""
        except OSError:
            return ""

    def _shortcuts(self) -> List[Dict]:
        now = time.time()
        if self._lnk_cache is not None and now - self._lnk_ts < self._ttl:
            return self._lnk_cache
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(TEMP_DIR / "_lnk_out.txt")
        ps_path = str(TEMP_DIR / "_lnk_scan.ps1")
        ps = (
            "$ErrorActionPreference='SilentlyContinue'\n"
            "$W = New-Object -ComObject WScript.Shell\n"
            "$dirs = @(\n"
            "  [Environment]::GetFolderPath('Desktop'),\n"
            "  [Environment]::GetFolderPath('CommonDesktopDirectory'),\n"
            "  (Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu'),\n"
            "  (Join-Path $env:ProgramData 'Microsoft\\Windows\\Start Menu')\n"
            ")\n"
            "$lines = New-Object System.Collections.Generic.List[string]\n"
            "foreach($d in $dirs){\n"
            "  if(Test-Path -LiteralPath $d){\n"
            "    Get-ChildItem -LiteralPath $d -Recurse -Filter *.lnk -Force | ForEach-Object {\n"
            "      $t=''\n"
            "      try { $t=$W.CreateShortcut($_.FullName).TargetPath } catch {}\n"
            "      $lines.Add('LNK|'+$_.FullName+'|'+$t)\n"
            "    }\n"
            "  }\n"
            "}\n"
            "[System.IO.File]::WriteAllLines('" + out_path + "', $lines, "
            "(New-Object System.Text.UTF8Encoding($false)))\n"
        )
        entries = []
        try:
            atomic_write_text(ps_path, ps)
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_path],
                timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.rstrip("\r\n").split("|")
                        if not parts or parts[0] != "LNK":
                            continue
                        link = parts[1] if len(parts) > 1 else ""
                        target = parts[2] if len(parts) > 2 else ""
                        lnk_name = os.path.splitext(os.path.basename(link))[0] if link else ""
                        exe_name = os.path.basename(target) if target else ""
                        exe_dir = os.path.basename(os.path.dirname(target)) if target else ""
                        entries.append({
                            "display_name": lnk_name,
                            "lnk_name": lnk_name,
                            "path": link,
                            "target": target,
                            "exe_name": exe_name,
                            "exe_dir": exe_dir,
                            "source": "快捷方式",
                        })
        except Exception:
            pass
        finally:
            for p in (ps_path, out_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        self._lnk_cache = entries
        self._lnk_ts = now
        return entries

    def _program_files_exes(self) -> List[Dict]:
        now = time.time()
        if self._pf_cache is not None and now - self._pf_ts < self._ttl:
            return self._pf_cache
        entries = self._pf_load_from_disk(now)
        if entries is None:
            entries = self._pf_scan()
            self._pf_save_to_disk(entries, now)
        self._pf_cache = entries
        self._pf_ts = now
        return entries

    @staticmethod
    def _pf_cache_path() -> str:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return str(CACHE_DIR / "agentrl_pf_cache.json")

    def _pf_load_from_disk(self, now: float) -> Optional[List[Dict]]:
        path = self._pf_cache_path()
        try:
            get_default_write_queue().flush()  # 先排空排队中的写，避免读到旧缓存
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if now - data.get("ts", 0) > 86400 or not data.get("files"):
                return None
            sample = data["files"][0]
            if not os.path.exists(sample):
                return None
            return [{"display_name": os.path.splitext(os.path.basename(p))[0],
                     "path": p, "target": "", "exe_name": os.path.basename(p),
                     "exe_dir": os.path.basename(os.path.dirname(p)),
                     "lnk_name": "", "source": "程序目录"} for p in data["files"]]
        except (OSError, ValueError, KeyError):
            return None

    def _pf_save_to_disk(self, entries: List[Dict], now: float):
        payload = {"ts": now, "files": [e["path"] for e in entries]}
        try:
            # 缓存写是纯 fire-and-forget：交给单写者队列，不阻塞扫描路径
            get_default_write_queue().submit_json(self._pf_cache_path(), payload)
        except RuntimeError:
            atomic_write_json(self._pf_cache_path(), payload)
        except OSError:
            pass

    def _pf_scan(self) -> List[Dict]:
        roots = []
        for var in ("ProgramFiles", "ProgramFiles(x86)"):
            p = os.environ.get(var, "")
            if p:
                roots.append(p)
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            roots.append(os.path.join(local, "Programs"))
        entries = []
        budget = 250000
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self.PRUNE_DIRS]
                for fn in filenames:
                    if budget <= 0:
                        break
                    budget -= 1
                    if fn.lower().endswith(".exe"):
                        full = os.path.join(dirpath, fn)
                        entries.append({
                            "display_name": fn, "path": full, "target": "",
                            "exe_name": fn, "exe_dir": os.path.basename(dirpath),
                            "lnk_name": "", "source": "程序目录",
                        })
                if budget <= 0:
                    break
            if budget <= 0:
                break
        return entries

    # ------------------------------------------------------------- 检索
    def search(self, query: str, top_k: int = 8) -> Tuple[List[Dict], str]:
        kw = self._extract_keyword(query)
        candidates = []
        for src in (self._registry_entries, self._shortcuts, self._program_files_exes):
            try:
                candidates.extend(src())
            except Exception:
                continue
        for item in candidates:
            self._compute_confidence(item, kw)
        candidates = [c for c in candidates if c.get("_confidence", 0) > 0]
        candidates.sort(key=lambda x: (-x["_confidence"], x.get("display_name", "")))
        hits = candidates[:top_k]
        return hits, kw

    # ------------------------------------------------------------- 结果文案
    def _format_hits(self, kw: str, hits: List[Dict]) -> str:
        lines = [f"### 系统查找结果: 「{kw}」\n"]
        for i, h in enumerate(hits, 1):
            conf = h.get("_confidence", 0)
            pct = f"{conf:.0%}"
            display = h.get("display_name", "")
            source = h.get("source", "")
            exe_path = h.get("target") or h.get("path") or ""
            lnk_path = h.get("path", "") if h.get("target") else ""
            exe_name = h.get("exe_name", "")
            lnk_name = h.get("lnk_name", "")
            match_field = h.get("_match_field", "")
            match_label = {"display_name": "名称匹配",
                           "exe_name": "文件名匹配",
                           "exe_dir": "目录名匹配",
                           "path": "路径匹配",
                           "target": "目标匹配"}.get(match_field, "模糊匹配")
            exe_basename = os.path.basename(exe_path) if exe_path else ""

            lines.append(f"{i}. **{display}**  {source}  置信度: **{pct}**  [{match_label}]")
            if exe_path:
                lines.append(f"   可执行文件: {exe_path}")
            if lnk_path and exe_name:
                lines.append(f"   快捷方式: {lnk_path}")
            lines.append("")
        lines.append("提示：置信度基于查询关键词与名称/文件名/目录名的综合相似度计算。"
                     "若结果不理想，可尝试更精确的名称（如英文名/版本号）再次查询。")
        return "\n".join(lines)

    def format_result(self, query: str, hits: List[Dict]) -> str:
        kw = self._extract_keyword(query)
        if not hits:
            return self.format_noresult(query)
        return self._format_hits(kw, hits)

    def format_noresult(self, query: str) -> str:
        kw = self._extract_keyword(query)
        return (
            f"### 系统查找结果: 「{kw}」\n\n"
            f"在 Windows 系统中未找到与「{kw}」匹配的程序或文件。\n\n"
            "建议：\n"
            "- 确认该软件已安装（可在「设置 → 应用」中查看）；\n"
            "- 绿色版/便携版/离线包 安装的程序请给出更具体的关键词（如英文名、版本号）；\n"
            "- 也可以让我「查看桌面 / 开始菜单」辅助定位。"
        )
