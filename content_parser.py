"""内容解析层：语言识别、注释剥离、未注释 TODO 检测、依赖抽取、API/配置/密钥提取。"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


LANGUAGE_BY_EXT = {
    '.py': 'python', '.pyi': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.java': 'java', '.kt': 'kotlin', '.kts': 'kotlin',
    '.go': 'go', '.rs': 'rust', '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.cc': 'cpp',
    '.hpp': 'cpp', '.cs': 'csharp', '.rb': 'ruby', '.php': 'php',
    '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell',
    '.ps1': 'powershell', '.bat': 'batch', '.cmd': 'batch',
    '.sql': 'sql', '.r': 'r', '.lua': 'lua', '.swift': 'swift',
    '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
    '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json', '.toml': 'toml',
    '.ini': 'ini', '.cfg': 'ini', '.conf': 'ini', '.env': 'dotenv',
    '.xml': 'xml', '.html': 'html', '.htm': 'html', '.css': 'css',
    '.csv': 'csv', '.tsv': 'csv',
    '.pdf': 'pdf',
}

TODO_PATTERN = re.compile(r'\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.*)$', re.IGNORECASE)
URL_PATTERN = re.compile(r'https?://[^\s\'")\]>]+')
SECRET_KEY_NAMES = (
    r'(?:api[_-]?key|secret(?:[_-]?key)?|token|password|passwd|'
    r'credential(?:[_-]?id)?|auth[_-]?token|private[_-]?key|'
    r'access[_-]?key(?:[_-]?id)?|client[_-]?secret|aws[_-]?secret)'
)
# 仅作用于「赋值右侧字面量」，不扫整行（避免 key 名/路径误伤）
SECRET_PLACEHOLDERS = re.compile(
    r'(?i)(your[_-]|your-|example|placeholder|changeme|change[_-]me|dummy|'
    r'fake[_-]|test[_-]|sample|xxx{2,}|<[^>]+>|\$\{|%s|redacted|'
    r'dummy[_-]value|not[_-]?a[_-]?real|insert[_-]|replace[_-]|'
    r'^(?:password|secret|token|apikey|api[_-]key|none|null|undefined|true|false)$)'
)
# 值形态：排除函数调用/属性链/环境变量读取/纯变量引用等非字面量
SECRET_VALUE_REJECT = re.compile(
    r'(?i)(?:\(|\)|\bself\.|\bsuper\(|::|=>|\bos\.|\benv\b|get_secret|'
    r'getenv|environ|process\.env|\.get\(|\.load|def |class |'
    r"^[A-Z][A-Za-z0-9_]*\()"  # DefaultAzureCredential() 等构造调用
)
# 纯标识符（变量/参数传递）不是硬编码字面量
SECRET_VALUE_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# kwargs 关键字位置：api_key=api_key / api_key=some_var
SECRET_KWARGS_REF = re.compile(
    rf'{SECRET_KEY_NAMES}\s*=\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][\w]*)*)\s*[,)\n]',
    re.I,
)
SECRET_PATTERNS = [
    # 带引号赋值（捕获引号内字面量，≥6 字符）
    re.compile(rf'{SECRET_KEY_NAMES}\s*[=:]\s*"([^"\']{{6,}})"', re.I),
    re.compile(rf"{SECRET_KEY_NAMES}\s*[=:]\s*'([^'\"]{{6,}})'", re.I),
    # 无引号 / YAML 冒号赋值（值 8+ 字符）
    re.compile(rf'{SECRET_KEY_NAMES}\s*[=:]\s*([^\s#\'\"{{}}]{{8,}})', re.I),
    # 行业固定前缀
    re.compile(r'(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|'
               r'sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
]
ENV_PATTERNS = [
    re.compile(r'os\.environ\[[\"\']([\w_]+)[\"\']\]'),
    re.compile(r'os\.environ\.get\([\"\']([\w_]+)[\"\']'),
    re.compile(r'process\.env(?:\.([\w]+)|\[[\"\']([\w]+)[\"\']\])'),
    re.compile(r'getenv\([\"\']([\w_]+)[\"\']'),
    re.compile(r'\$\{?([A-Z][A-Z0-9_]{2,})\}?'),
]
CONFIG_KEY_PATTERNS = {
    'yaml': re.compile(r'^[ \t]*([A-Za-z_][\w.-]*)\s*:', re.M),
    'json': re.compile(r'"([A-Za-z_][\w.-]*)"\s*:'),
    'ini': re.compile(r'^[ \t]*([A-Za-z_][\w.-]*)\s*=', re.M),
    'toml': re.compile(r'^[ \t]*([A-Za-z_][\w.-]*)\s*=', re.M),
    'dotenv': re.compile(r'^[ \t]*([A-Za-z_][\w]*)\s*=', re.M),
}

# 多语言 import/require 依赖抽取
IMPORT_RES = {
    'python': [
        re.compile(r'^\s*import\s+([A-Za-z_][\w.]*)', re.M),
        re.compile(r'^\s*from\s+([A-Za-z_][\w.]*)\s+import', re.M),
    ],
    'javascript': [
        re.compile(r'''require\(\s*['"]([^'"./][^'"]*)['"]\s*\)'''),
        re.compile(r'''from\s+['"]([^'"]+)['"]'''),
        re.compile(r'''import\s+['"]([^'"]+)['"]'''),
        re.compile(r'''import\s+[^'"]*?\s+from\s+['"]([^'"]+)['"]'''),
    ],
    'typescript': None,  # reuse javascript
    'java': [re.compile(r'^\s*import\s+([a-z][\w.]*)', re.M)],
    'go': [re.compile(r'^\s*import\s+(?:[\w]+\s+)?"([^"]+)"', re.M),
           re.compile(r'^\s*import\s+\(([^)]*)\)', re.M | re.S)],
    'rust': [re.compile(r'^\s*(?:pub\s+)?use\s+([a-z][\w:]*)', re.M)],
    'c': [re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M)],
    'cpp': [re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M)],
    'ruby': [re.compile(r"""^\s*require\s+['"]([^'"]+)['"]""", re.M)],
    'php': [re.compile(r"""^\s*(?:require|include)(?:_once)?\s*\(?\s*['"]([^'"]+)['"]""", re.M)],
    'shell': [re.compile(r'^\s*(?:source|\.)\s+([^\s;]+)', re.M)],
}

BLOCK_COMMENT_RES = {
    'python': [(r'"""', r'"""'), (r"'''", r"'''"), (r'#', None)],
    'javascript': [(r'/*', r'*/'), (r'//', None)],
    'typescript': [(r'/*', r'*/'), (r'//', None)],
    'java': [(r'/*', r'*/'), (r'//', None)],
    'go': [(r'/*', r'*/'), (r'//', None)],
    'rust': [(r'/*', r'*/'), (r'//', None)],
    'c': [(r'/*', r'*/'), (r'//', None)],
    'cpp': [(r'/*', r'*/'), (r'//', None)],
    'csharp': [(r'/*', r'*/'), (r'//', None)],
    'ruby': [(r'=begin', r'=end'), (r'#', None)],
    'php': [(r'/*', r'*/'), (r'//', None), (r'#', None)],
    'shell': [(r'#', None)],
    'powershell': [(r'<#', r'#>'), (r'#', None)],
    'sql': [(r'/*', r'*/'), (r'--', None)],
    'yaml': [(r'#', None)],
    'toml': [(r'#', None)],
    'ini': [(r'[;#]', None)],
    'dotenv': [(r'#', None)],
    'default': [(r'/*', r'*/'), (r'//', None), (r'#', None)],
}


def detect_language(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return LANGUAGE_BY_EXT.get(ext, 'unknown')


def _strip_line_comment(line: str, marker: str) -> str:
    idx = line.find(marker)
    if idx < 0:
        return line
    # 避免误伤字符串中的 # 简化处理：直接截断
    return line[:idx]


def strip_comments(content: str, language: str) -> Tuple[str, List[str]]:
    """剥离注释，返回 (剥离后代码, 注释文本列表)。

    注释文本用于「未被注释掉的 TODO」判断——即注释本身是活注释，
    而被多重注释/字符串包裹的死代码 TODO 不计入。
    """
    rules = BLOCK_COMMENT_RES.get(language, BLOCK_COMMENT_RES['default'])
    comments: List[str] = []
    out_lines: List[str] = []
    i = 0
    n = len(content)
    line_start = 0

    def flush_line(buf: str):
        out_lines.append(buf)

    active_block: Optional[Tuple[str, str]] = None  # (start, end) end=None -> line comment
    block_buf = ""

    while i < n:
        if active_block is not None:
            start, end = active_block
            if end is None:
                # line comment until newline
                j = content.find('\n', i)
                if j < 0:
                    j = n
                comments.append(content[i:j].strip())
                i = j
                active_block = None
                continue
            j = content.find(end, i)
            if j < 0:
                comments.append(content[i:].strip())
                i = n
                continue
            comments.append(content[i:j].strip())
            i = j + len(end)
            active_block = None
            continue

        matched = False
        for start, end in rules:
            if content.startswith(start, i):
                # 行首缓冲区推进
                active_block = (start, end)
                i += len(start)
                matched = True
                break
        if matched:
            continue

        ch = content[i]
        if ch == '\n':
            flush_line('\n')
            i += 1
            continue
        flush_line(ch)
        i += 1

    stripped = ''.join(out_lines)
    # 清理多余空行
    stripped = re.sub(r'\n{3,}', '\n\n', stripped)
    return stripped, [c for c in comments if c]


def find_active_todos(content: str, language: str) -> List[Dict]:
    """检测未被注释掉（活注释或裸代码）的 TODO/FIXME/XXX/HACK。

    - 数据文件（json/csv 等）中作为字符串内容出现的标记不计
    - 被再次注释的死代码（## TODO / // // TODO）不计
    - 标记必须处于注释上下文或独立代码标记位置
    """
    if language in ('json', 'csv', 'markdown', 'rst', 'text'):
        # 文档/数据中提及 TODO 关键词不算技术债标记
        return []

    results: List[Dict] = []
    lines = content.splitlines()
    comment_markers = {
        'python': '#', 'shell': '#', 'yaml': '#', 'ruby': '#', 'toml': '#',
        'dotenv': '#', 'ini': ';', 'powershell': '#',
    }
    block_line = {
        'javascript': '//', 'typescript': '//', 'java': '//', 'go': '//',
        'rust': '//', 'c': '//', 'cpp': '//', 'csharp': '//', 'php': '//',
        'sql': '--',
    }
    lm = comment_markers.get(language) or block_line.get(language) or '//'

    for lineno, line in enumerate(lines, 1):
        m = TODO_PATTERN.search(line)
        if not m:
            continue
        stripped = line.strip()
        marker = m.group(1).upper()
        # 排除字典键/常量映射：'FIXME': 3 / "TODO": 0
        if re.search(rf'''['"]{marker}['"]\s*:''', line):
            continue
        # 排除整行都是字符串字面量且无注释符的情况（纯数据行）
        if lm not in stripped and not re.search(rf'\b{marker}\s*:', stripped):
            # 非注释行且无 TODO: 说明形态 → 多半是字符串内容
            if stripped.startswith(('"', "'", '`')):
                continue

        # 死代码：注释符重复
        dead = False
        if stripped.startswith(lm + lm) or stripped.startswith(lm + ' ' + lm):
            dead = True
        if language == 'python' and stripped.startswith('##'):
            dead = True

        in_comment = (
            lm in stripped
            or stripped.startswith(('/*', '*', '#', '//', '--', ';'))
        )
        # 仅保留：活注释中的标记，或代码中的 TODO: 形态
        if not in_comment and not re.search(rf'\b{marker}\s*:', stripped):
            # 裸出现在代码中但无冒号说明 → 疑似字符串，跳过
            if not re.search(rf'#.*{marker}|//.*{marker}|--.*{marker}|;.*{marker}', stripped, re.I):
                continue

        results.append({
            'line': lineno,
            'marker': marker,
            'text': stripped[:300],
            'in_comment': in_comment,
            'active': not dead,
            'priority': {'FIXME': 3, 'HACK': 2, 'XXX': 1, 'TODO': 0}.get(marker, 0),
        })
    return [r for r in results if r['active']]


def extract_imports(content: str, language: str) -> List[str]:
    if language == 'typescript':
        language = 'javascript'
    res = IMPORT_RES.get(language) or IMPORT_RES.get('javascript' if language == 'javascript' else None)
    if not res:
        return []
    mods: List[str] = []
    for rx in res:
        for m in rx.finditer(content):
            raw = m.group(1).strip()
            if language == 'go' and '\n' in raw:
                for im in re.finditer(r'"([^"]+)"', raw):
                    mods.append(im.group(1))
                continue
            if language in ('c', 'cpp'):
                mods.append(raw)
                continue
            mods.append(raw.split('.')[0].split('/')[0].split(':')[0])
    # 去重保序
    seen: Set[str] = set()
    out = []
    for m in mods:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def extract_public_api(content: str, language: str, path: str) -> List[str]:
    apis: List[str] = []
    if language == 'python':
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return apis
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                apis.append(node.name)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                apis.append(node.name)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith('_'):
                        apis.append(f"{node.name}.{sub.name}")
    elif language in ('javascript', 'typescript'):
        for m in re.finditer(r'export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)', content):
            apis.append(m.group(1))
        for m in re.finditer(r'module\.exports\.([A-Za-z_$][\w$]*)', content):
            apis.append(m.group(1))
    elif language == 'go':
        for m in re.finditer(r'^func\s+([A-Z][\w]*)', content, re.M):
            apis.append(m.group(1))
        for m in re.finditer(r'^type\s+([A-Z][\w]*)', content, re.M):
            apis.append(m.group(1))
    elif language == 'rust':
        for m in re.finditer(r'^pub\s+(?:async\s+)?fn\s+([A-Za-z_][\w]*)', content, re.M):
            apis.append(m.group(1))
        for m in re.finditer(r'^pub\s+struct\s+([A-Za-z_][\w]*)', content, re.M):
            apis.append(m.group(1))
    elif language == 'java':
        for m in re.finditer(r'public\s+(?:static\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)', content):
            apis.append(m.group(1))
        for m in re.finditer(r'public\s+(?:static\s+)?[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\(', content):
            apis.append(m.group(1))
    return apis


def extract_env_refs(content: str) -> List[str]:
    refs: List[str] = []
    for rx in ENV_PATTERNS:
        for m in rx.finditer(content):
            for g in m.groups():
                if g:
                    refs.append(g)
    return sorted(set(refs))


def extract_urls(content: str) -> List[str]:
    return sorted(set(URL_PATTERN.findall(content)))


def _is_variable_ref(key: str, val: str) -> bool:
    """api_key=api_key / token=some_var 等纯变量传递（非硬编码字面量）。"""
    val = val.strip().rstrip(',);]}')
    if not SECRET_VALUE_IDENT.match(val):
        return False
    key_l = (key or '').lower().replace('-', '_')
    val_l = val.lower().replace('-', '_')
    # 与 key 同名或互为子串（api_key=api_key, token=token）
    if val_l == key_l or (key_l and (val_l in key_l or key_l in val_l)):
        return True
    # 全小写蛇形/纯字母且无数字 → 变量名风格（短密码带数字则放过）
    if val.isalpha() and val.islower() and len(val) <= 20:
        return True
    if '_' in val and val.replace('_', '').islower() and not any(c.isdigit() for c in val):
        return True
    return False


def extract_secrets(content: str, lines: Optional[List[str]] = None) -> List[Dict]:
    """疑似硬编码密钥：字面量赋值 + 行业前缀；过滤占位符与代码表达式。"""
    findings: List[Dict] = []
    lines = lines if lines is not None else content.splitlines()
    for i, line in enumerate(lines, 1):
        if re.search(r'os\.environ|process\.env|getenv\s*\(', line, re.I):
            continue
        for rx in SECRET_PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            if m.lastindex:
                raw_val = m.group(1)
                val = raw_val.strip().strip('\'"').rstrip(',);]}')
                if len(val) < 6:
                    continue
                if SECRET_PLACEHOLDERS.search(val):
                    continue
                if SECRET_VALUE_REJECT.search(val):
                    continue
                # 键名提取（用于变量同名判断）
                head = line[:m.start(1)] if m.start(1) else line
                km2 = re.search(r'([A-Za-z_][\w-]*)\s*[=:]', head)
                key_name = km2.group(1) if km2 else ''
                if _is_variable_ref(key_name, val):
                    continue
                # kwargs: key=val 且 val 为无数字标识符 → 变量
                if re.search(rf'{SECRET_KEY_NAMES}\s*=\s*{re.escape(val)}\s*[,)]', line, re.I):
                    if SECRET_VALUE_IDENT.match(val) and not any(c.isdigit() for c in val):
                        continue
            elif SECRET_PLACEHOLDERS.search(line) and not re.search(
                    r'-----BEGIN|AKIA[0-9A-Z]{16}', line):
                continue
            findings.append({'line': i, 'text': line.strip()[:200]})
            break
    return findings


def extract_config_keys(content: str, language: str) -> List[str]:
    rx = CONFIG_KEY_PATTERNS.get(language)
    if not rx:
        if language == 'yaml':
            rx = CONFIG_KEY_PATTERNS['yaml']
        else:
            return []
    return sorted(set(rx.findall(content)))


def extract_test_refs(content: str) -> List[str]:
    """测试覆盖引用：assert/test 名与被测符号。"""
    refs: List[str] = []
    for m in re.finditer(r'\b(?:assert|expect)\w*\s*\(', content):
        refs.append(m.group(0))
    for m in re.finditer(r'\b(?:test_|it\(|describe\(|def test_)', content):
        refs.append(m.group(0).strip())
    return refs


def extract_doc_refs(content: str, language: str) -> List[str]:
    """文档引用：markdown/rst 中出现的代码符号与文件名。"""
    refs: List[str] = []
    refs += re.findall(r'`([A-Za-z_][\w.]{2,})`', content)
    refs += re.findall(r'\b([\w/]+\.(?:py|js|ts|java|go|rs|md))\b', content)
    return sorted(set(refs))


@dataclass
class ParsedFile:
    path: str
    language: str
    size: int
    encoding: str = 'utf-8'
    line_count: int = 0
    imports: List[str] = field(default_factory=list)
    todos: List[Dict] = field(default_factory=list)
    public_apis: List[str] = field(default_factory=list)
    env_refs: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    suspected_secrets: List[Dict] = field(default_factory=list)
    config_keys: List[str] = field(default_factory=list)
    test_refs: List[str] = field(default_factory=list)
    doc_refs: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    stripped_code: str = ''
    error: str = ''


def parse_file(path: str, content: str, encoding: str = 'utf-8') -> ParsedFile:
    language = detect_language(path)
    pf = ParsedFile(
        path=path.replace('\\', '/'),
        language=language,
        size=len(content.encode('utf-8', errors='replace')),
        encoding=encoding,
        line_count=content.count('\n') + 1,
    )
    try:
        stripped, comments = strip_comments(content, language)
        pf.comments = comments[:200]
        pf.stripped_code = stripped
        pf.todos = find_active_todos(content, language)
        pf.imports = extract_imports(content, language)
        pf.public_apis = extract_public_api(content, language, path)
        pf.env_refs = extract_env_refs(content)
        pf.urls = extract_urls(content)
        pf.suspected_secrets = extract_secrets(content)
        pf.config_keys = extract_config_keys(content, language)
        pf.test_refs = extract_test_refs(content) if 'test' in os.path.basename(path).lower() else []
        pf.doc_refs = extract_doc_refs(content, language) if language in ('markdown', 'rst', 'text') else []
    except Exception as e:
        pf.error = str(e)
    return pf
