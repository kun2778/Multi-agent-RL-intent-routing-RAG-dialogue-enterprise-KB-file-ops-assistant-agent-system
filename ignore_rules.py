"""忽略规则引擎：读取 .gitignore、.git/info/exclude 等，合并规则、glob/否定/目录规则。

功能：
- 读取 .gitignore、.git/info/exclude、.ignore、.npmignore、.dockerignore
- 规则合并，冲突时更严格排除优先
- 处理 glob 模式、否定规则、目录规则
- 参考 .editorconfig、sonar-project.properties
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _translate_glob(pattern: str) -> str:
    """将 gitignore glob 转为正则。

    规则要点（与 git 一致的简化版）：
    - 无斜杠的模式匹配任意层级 basename
    - 以 / 开头锚定根目录
    - 以 / 结尾表示仅目录
    - ** 跨目录，* 不跨 /，? 单字符
    - [] 字符类，! 前缀为否定（调用方处理）
    """
    i = 0
    n = len(pattern)
    out = []
    while i < n:
        c = pattern[i]
        if c == '*':
            if i + 1 < n and pattern[i + 1] == '*':
                # ** 
                if i + 2 < n and pattern[i + 2] == '/':
                    out.append('(?:.*/)?')
                    i += 3
                    continue
                out.append('.*')
                i += 2
                continue
            out.append('[^/]*')
            i += 1
            continue
        if c == '?':
            out.append('[^/]')
            i += 1
            continue
        if c == '[':
            j = i + 1
            if j < n and pattern[j] in '!^':
                j += 1
            if j < n and pattern[j] == ']':
                j += 1
            while j < n and pattern[j] != ']':
                j += 1
            if j >= n:
                out.append('\\[')
                i += 1
                continue
            cls = pattern[i:j + 1]
            if cls.startswith('[!'):
                cls = '[^' + cls[2:]
            out.append(cls)
            i = j + 1
            continue
        if c == '\\' and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        out.append(re.escape(c))
        i += 1
    return ''.join(out)


class IgnoreRule:
    """单个忽略规则。"""

    def __init__(self, pattern: str, source: str = ""):
        raw = pattern.rstrip('\n\r')
        self.raw = raw
        self.source = source
        self.is_negation = False
        self.dir_only = False
        self.anchored = False
        self.pattern = raw

        if self.pattern.startswith('!'):
            self.is_negation = True
            self.pattern = self.pattern[1:]
        elif self.pattern.startswith('\\!'):
            self.pattern = self.pattern[1:]
        elif self.pattern.startswith('\\#'):
            self.pattern = self.pattern[1:]

        if self.pattern.endswith('/'):
            self.dir_only = True
            self.pattern = self.pattern.rstrip('/')

        if self.pattern.startswith('/'):
            self.anchored = True
            self.pattern = self.pattern[1:]

        self._compile()

    def _compile(self):
        if not self.pattern:
            self.compiled = None
            self.basename_compiled = None
            return
        body = _translate_glob(self.pattern)
        self.compiled = re.compile('^' + body + '(?:/.*)?$' if '/' in self.pattern
                                   else '^' + body + '$')
        if '/' in self.pattern:
            self.basename_compiled = re.compile('^' + _translate_glob(self.pattern.split('/')[-1]) + '$')
        else:
            self.basename_compiled = re.compile('^' + body + '$')

    @property
    def valid(self) -> bool:
        return self.compiled is not None

    def matches(self, rel_path: str, is_dir: bool = False) -> bool:
        if self.compiled is None:
            return False
        if self.dir_only and not is_dir:
            # 目录规则也匹配该目录下的内容（git 语义）
            pass
        norm = rel_path.replace('\\', '/').strip('/')
        if not norm:
            return False

        if self.dir_only:
            # 匹配目录本身或路径中任意一段目录
            parts = norm.split('/')
            for idx in range(len(parts)):
                sub = '/'.join(parts[: idx + 1])
                if self.compiled.match(sub) or (self.basename_compiled and self.basename_compiled.match(parts[idx])):
                    return True
            return False

        if '/' in self.pattern:
            if self.anchored:
                return bool(self.compiled.match(norm))
            # 非锚定但含斜杠：匹配任意后缀位置
            if self.compiled.match(norm):
                return True
            segments = norm.split('/')
            for idx in range(len(segments)):
                sub = '/'.join(segments[idx:])
                if self.compiled.match(sub):
                    return True
            return False

        # 无斜杠：匹配任意层级 basename
        basename = norm.split('/')[-1]
        return bool(self.basename_compiled.match(basename))


class IgnoreRulesEngine:
    """忽略规则引擎：加载多源规则，按 git 语义合并（最后匹配优先，否定可撤销）。"""

    # 冲突时更严格：排除类源优先于参考类源
    SOURCE_PRIORITY = {
        '.gitignore': 100,
        '.git/info/exclude': 90,
        '.ignore': 80,
        '.npmignore': 70,
        '.dockerignore': 60,
        'system': 50,
        '.editorconfig': 10,
        'sonar-project.properties': 10,
        'custom': 40,
    }

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.rules: List[IgnoreRule] = []
        self._loaded_sources: List[str] = []
        self.exclusion_hints: Set[str] = set()
        self.editorconfig_excludes: List[str] = []
        self.sonar_excludes: List[str] = []

    # ------------------------------------------------------------ loaders
    def load_all(self) -> List[str]:
        """加载全部已知规则源。"""
        self.load_gitignore()
        self.load_git_info_exclude()
        for name in ('.ignore', '.npmignore', '.dockerignore'):
            self.load_custom_ignore(name)
        self.load_editorconfig()
        self.load_sonar_project_properties()
        return list(self._loaded_sources)

    def load_gitignore(self, path: str = None):
        gitignore_path = path or os.path.join(self.root_path, ".gitignore")
        if os.path.exists(gitignore_path):
            self._parse_gitignore_file(gitignore_path, ".gitignore")

    def load_git_info_exclude(self):
        gitinfo_path = os.path.join(self.root_path, ".git", "info", "exclude")
        if os.path.exists(gitinfo_path):
            self._parse_gitignore_file(gitinfo_path, ".git/info/exclude")

    def load_custom_ignore(self, filename: str = ".ignore"):
        path = os.path.join(self.root_path, filename)
        if os.path.exists(path):
            self._parse_gitignore_file(path, filename)

    def load_npmignore(self):
        self.load_custom_ignore(".npmignore")

    def load_dockerignore(self):
        self.load_custom_ignore(".dockerignore")

    def load_editorconfig(self):
        path = os.path.join(self.root_path, ".editorconfig")
        if not os.path.exists(path):
            return
        self._loaded_sources.append(".editorconfig")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('#', ';', '[')):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if key in ('exclude', 'excludes'):
                            for part in re.split(r'[, ]+', value):
                                if part:
                                    self.editorconfig_excludes.append(part)
                                    self.exclusion_hints.add(part)
        except OSError:
            pass

    def load_sonar_project_properties(self):
        path = os.path.join(self.root_path, "sonar-project.properties")
        if not os.path.exists(path):
            return
        self._loaded_sources.append("sonar-project.properties")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    if 'exclusion' in key.lower() or key.endswith('.exclusions'):
                        for part in re.split(r'[, ]+', value):
                            if part:
                                self.sonar_excludes.append(part)
                                self.exclusion_hints.add(part)
        except OSError:
            pass

    def _parse_gitignore_file(self, filepath: str, source: str):
        if source not in self._loaded_sources:
            self._loaded_sources.append(source)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    self.add_rule(line.strip(), source=source)
        except OSError:
            pass

    def add_rule(self, pattern: str, source: str = "custom"):
        rule = IgnoreRule(pattern, source=source)
        if rule.valid:
            self.rules.append(rule)

    def add_system_defaults(self):
        """内置系统排除（.git、node_modules、历史审查产物等）。"""
        for name in sorted(SYSTEM_EXCLUDED_DIRS):
            self.add_rule(name + "/", source="system")
            self.add_rule(name, source="system")
        # 历史只读审查产物目录：review_YYYYMMDD_HHumss_hash
        # 仅排除目录，不误伤源码文件 review_collector.py / review_pipeline.py
        self.add_rule("review_20*/", source="system")

    def compile_all(self):
        for rule in self.rules:
            rule._compile()

    # ------------------------------------------------------------ matching
    def _match(self, rel_path: str, is_dir: bool) -> Tuple[bool, Optional[IgnoreRule]]:
        """按源优先级 + 文件内顺序求最终结果。

        更严格排除优先：高优先级源中的正向排除可覆盖低优先级源的否定。
        """
        norm = rel_path.replace('\\', '/').strip('/')
        decision: Optional[bool] = None
        winner: Optional[IgnoreRule] = None
        winner_pri = -1
        winner_idx = -1

        for idx, rule in enumerate(self.rules):
            if not rule.matches(norm, is_dir=is_dir):
                continue
            pri = self.SOURCE_PRIORITY.get(rule.source, 40)
            if winner is None or pri > winner_pri or (pri == winner_pri and idx > winner_idx):
                winner = rule
                winner_pri = pri
                winner_idx = idx
                decision = not rule.is_negation

        if decision is None:
            # 参考源 hints（editorconfig/sonar）作为弱排除
            segments = norm.split('/')
            for hint in self.exclusion_hints:
                hn = hint.strip('/').split('/')[-1] if hint else ''
                if hn and hn in segments:
                    return True, None
            return False, None
        return decision, winner

    def should_exclude(self, rel_path: str) -> bool:
        excluded, _ = self._match(rel_path, is_dir=False)
        if not excluded:
            excluded_dir, _ = self._match(rel_path, is_dir=True)
            return excluded_dir
        return excluded

    def should_include(self, rel_path: str) -> bool:
        return not self.should_exclude(rel_path)

    def should_prune_dir(self, rel_dir: str) -> bool:
        """目录是否应被剪枝（不再递归）。"""
        excluded, rule = self._match(rel_dir, is_dir=True)
        if excluded:
            return True
        # 即使规则是文件规则，若匹配目录名也可剪枝（无否定撤销时）
        for r in self.rules:
            if r.is_negation:
                continue
            if r.matches(rel_dir.replace('\\', '/').strip('/'), is_dir=True):
                # 检查是否有更高优先级否定
                neg_hit = any(
                    nr.is_negation and nr.matches(rel_dir.replace('\\', '/').strip('/'), is_dir=True)
                    for nr in self.rules
                )
                if not neg_hit:
                    return True
        return False

    def get_loaded_sources(self) -> List[str]:
        return list(self._loaded_sources)

    def get_excluded_paths(self) -> Set[str]:
        """扫描树收集被排除路径（调试用）。"""
        excluded: Set[str] = set()
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            keep_dirs = []
            for d in dirnames:
                rel = os.path.relpath(os.path.join(dirpath, d), self.root_path)
                if self.should_prune_dir(rel):
                    excluded.add(rel.replace('\\', '/'))
                else:
                    keep_dirs.append(d)
            dirnames[:] = keep_dirs
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root_path)
                if self.should_exclude(rel):
                    excluded.add(rel.replace('\\', '/'))
        return excluded


# 预定义的系统排除目录
SYSTEM_EXCLUDED_DIRS = {
    ".git", "node_modules", "dist", "build",
    "__pycache__", ".cache", ".venv", "venv",
    ".svn", ".hg", ".idea", ".vscode",
    ".runtime", ".eggs", "*.egg-info",
}


def create_default_engine(root_path: str, load_sources: bool = True) -> IgnoreRulesEngine:
    """创建默认的忽略规则引擎：系统默认 + 多源合并。"""
    engine = IgnoreRulesEngine(root_path)
    engine.add_system_defaults()
    if load_sources:
        engine.load_all()
    return engine


if __name__ == "__main__":
    engine = create_default_engine(os.path.dirname(os.path.abspath(__file__)))
    print("sources:", engine.get_loaded_sources())
    for p in [".git", "node_modules", "dist", "build", "README.md",
              "src/main.py", "__pycache__/x.pyc", ".env", "config/app.yaml"]:
        print(f"Path '{p}': excluded={engine.should_exclude(p)}")
