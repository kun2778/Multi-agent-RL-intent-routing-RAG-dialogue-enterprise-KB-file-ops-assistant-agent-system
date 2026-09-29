from typing import List, Tuple, Dict, Optional
import re
from agent_state import QuerySubIntent, SlotType


class QueryRewriter:
    """查询改写：提取操作意图、目标文件、核心实体与关系。

    这是状态空间的"特征前端" —— 把对话框输入转换成 RL 可用的结构化信号：
      - score_operations(): 对 ADD/DELETE/MODIFY/REFACTOR/QUERY/DEEP_SEARCH 打操作分
      - extract_target():   提取被操作的目标文件（路径/文件名）
      - rewrite():          生成重写后的查询 + 实体 + 关系
    """

    OP_KEYWORDS = {
        "add_file": ["添加", "新增", "创建", "新建", "建立", "写入", "保存", "加上", "增加", "建立", "加入",
                     "add", "create", "new", "write", "save", "mkdir"],
        "delete_file": ["删除", "移除", "清理", "删掉", "干掉", "去掉", "移除掉", "清除", "删除文件",
                        "remove", "delete", "rm", "clean"],
        "modify_file": ["修改", "更新", "编辑", "改成", "改为", "替换", "覆盖", "补充", "改一下", "改配置",
                        "改代码", "改掉", "modify", "edit", "update", "change", "replace", "rewrite"],
        "refactor": ["整理", "重构", "重命名", "移动", "分类", "归档", "归类", "重组", "组织", "排序",
                     "分一下类", "分类归类",
                     "整理文件", "重构文件夹", "rerange", "refactor", "rename", "move", "organize",
                     "sort", "restructure", "structure"],
        "query": ["查", "找", "搜索", "查询", "检索", "什么", "怎么", "多少", "哪里", "谁", "何时",
                  "是否存在", "存在吗", "有哪些", "文件名", "search", "find", "query",
                  "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取",
                  "显示", "列出", "查看", "打印", "返回", "最近", "最近保存", "最近修改",
                  "最近创建", "最近添加", "最近操作", "最近使用", "最近编辑", "最近打开",
                  "最近访问", "文件列表", "文件信息", "文件内容", "文件路径"],
        "deep_search": ["对比", "比较", "分析", "差异", "区别", "综合", "总结", "报告", "趋势",
                        "深入", "研究", "评估", "原因", "影响", "方案", "推荐",
                        "compare", "analyze", "analysis", "summary", "deep", "trend"],
        # 新增意图
        "file_analysis": [
            "分析", "审查", "检查", "诊断", "评估", "扫描分析",
            "analyze", "review", "inspect", "diagnose", "evaluate",
            "只读分析", "深度分析", "代码分析", "静态分析",
        ],
        "file_scan": [
            "扫描", "遍历", "列举", "列出", "统计", "清点",
            "scan", "traverse", "list", "enumerate", "count",
            "文件扫描", "目录扫描", "项目扫描",
        ],
        "report_generate": [
            "报告", "生成报告", "输出报告", "汇总", "总结报告",
            "report", "generate report", "summary report",
            "markdown报告", "json报告", "csv报告",
        ],
        "filename_query": [
            "最近添加的文件", "最近修改的文件", "最近创建的文件",
            "最近编辑的文件", "最近使用的文件", "最近访问的文件",
            "最近保存的文件", "最近操作的文件", "最近的文件",
            "文件名搜索", "搜索文件名", "查找文件名",
            "最新文件", "最新添加", "最新修改", "最新创建",
            "最近添加的文件名", "最近修改的文件名", "最近创建的文件名",
            "recent files", "find files", "search files",
            "newly added", "recently modified", "recently created",
        ],
    }

    FILE_RE = re.compile(r"[A-Za-z]:\\.+?\.[A-Za-z0-9]{1,6}|[\w.\-\\/]+\.[A-Za-z0-9]{1,6}", re.IGNORECASE)
    STRONG_FILE_RE = re.compile(r"[A-Za-z]:\\.+?\.[A-Za-z0-9]{1,6}|[\w\-]+\.[a-z]{1,6}", re.IGNORECASE)
    
    ACTION_WORDS = ["删除", "移除", "清理", "删掉", "干掉", "添加", "新增", "创建", "新建", 
                    "修改", "更新", "编辑", "remove", "delete", "rm", "add", "create", 
                    "new", "write", "save", "modify", "edit", "update"]

    def __init__(self):
        self.entity_patterns = [
            r'\b(文件|文档|配置|代码|脚本|项目|文件夹|目录|目录结构|模块|文件列表)\b',
        ]
        self.relation_patterns = [
            r'(对比|比较|差异|区别|vs|versus|与|和)',
            r'(是多少|为什么|怎么样|如何|怎样|什么|哪个|哪些|谁|何时|哪里|多少|存在吗)',
            r'(趋势|变化|增长|下降|上升|减少|影响|因素|原因|结果|效果)',
        ]

    def score_operations(self, query: str) -> Dict[str, float]:
        """对 6 类操作意图分别打分（关键词命中数 / 查询长度归一化）。"""
        q = query.lower()
        scores: Dict[str, float] = {}
        for intent, kws in self.OP_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw.lower() in q)
            scores[intent] = hits
        return scores

    def top_operation(self, query: str) -> str:
        """返回得分最高的意图键（含 'vague'）。"""
        scores = self.score_operations(query)
        max_key = max(scores, key=scores.get)
        if scores[max_key] == 0:
            return "vague"
        return max_key

    def extract_target(self, query: str) -> Optional[str]:
        """提取被操作的目标文件路径（第一个）。"""
        targets = self.extract_targets(query)
        return targets[0] if targets else None

    def extract_targets(self, query: str) -> List[str]:
        """提取查询中所有的目标文件路径（对比/批量操作需要）。"""
        matches = self.FILE_RE.findall(query)
        if not matches:
            matches = self.STRONG_FILE_RE.findall(query)

        cleaned = []
        for m in matches:
            t = m
            for word in self.ACTION_WORDS:
                if t.lower().startswith(word.lower()):
                    t = t[len(word):].strip()
                    break
            if t:
                refined = self._refine_path(t)
                cleaned.append(refined or t)

        return list(dict.fromkeys(cleaned))

    @staticmethod
    def _refine_path(path: str) -> Optional[str]:
        """文件系统感知精炼：正则可能把中文指令吞进路径，此处截到真实存在的文件。

        例: "C:\\...\\resume&essay路径下帮我输出resume .pdf"
          → 尾部 ASCII 文件名 "resume .pdf"
          → 目录段允许前缀匹配 → "C:\\...\\resume&essay"
          → "C:\\...\\resume&essay\\resume .pdf"
        """
        import os
        if not path:
            return None
        norm = path.replace("/", "\\")
        if not os.path.isabs(norm):
            return None
        if os.path.isfile(norm):
            return norm

        # 1) 尾部提取 filename.ext —— 优先 ASCII，避免中文指令被吞进文件名
        fm = re.search(r"([A-Za-z0-9_\- ]+\.[A-Za-z0-9]{1,6})$", norm)
        if fm:
            filename = fm.group(1)
            dir_text = norm[:fm.start()].rstrip("\\/")
        else:
            fm2 = re.search(r"([^\\\\:*?\"<>|]+\.[A-Za-z0-9]{1,6})$", norm)
            if not fm2:
                return None
            filename = fm2.group(1)
            dir_text = norm[:fm2.start()].rstrip("\\/")

        base = QueryRewriter._longest_existing_dir(dir_text)
        if not base:
            return None

        cand = base + "\\" + filename
        if os.path.isfile(cand):
            return cand

        # 4) 文件名仍被中文指令污染：按扩展名在 base 内消歧
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            return None
        try:
            names = [n for n in os.listdir(base)
                     if os.path.isfile(os.path.join(base, n)) and n.lower().endswith(ext)]
        except OSError:
            return None
        if filename in names:
            return os.path.join(base, filename)
        # 文件名是真实名的后缀，或真实名是提取名的后缀
        for n in names:
            if n.endswith(filename) or filename.endswith(n):
                return os.path.join(base, n)
        if len(names) == 1:
            return os.path.join(base, names[0])
        # 最长公共后缀启发
        best, best_len = None, 0
        for n in names:
            k = 0
            while k < min(len(n), len(filename)) and n[-(k + 1)].lower() == filename[-(k + 1)].lower():
                k += 1
            if k > best_len and k >= len(os.path.splitext(filename)[1]):
                best, best_len = n, k
        return os.path.join(base, best) if best else None

    @staticmethod
    def _longest_existing_dir(dir_text: str) -> Optional[str]:
        """在 dir_text 中找最长已存在的目录前缀；最后一段允许与中文指令粘连（前缀匹配）。"""
        import os
        if not dir_text:
            return None
        parts = dir_text.split("\\")
        acc: List[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                if os.path.isdir(part):
                    acc.append(part)
                else:
                    return "\\".join(acc) or None
                continue
            if not part:
                continue
            trial = "\\".join(acc + [part])
            if os.path.isdir(trial):
                acc.append(part)
                continue
            # 粘连段：从长到短试前缀
            found = None
            for j in range(len(part) - 1, 0, -1):
                trial = "\\".join(acc + [part[:j]])
                if os.path.isdir(trial):
                    found = part[:j]
                    break
            if found:
                acc.append(found)
            break
        return "\\".join(acc) if acc else None

    def rewrite(self, query: str) -> Tuple[str, List[str], List[str]]:
        """重写查询，提取目标文件/实体/关系。返回 (rewritten, entities, relations)。"""
        targets = self.extract_targets(query)
        entities = self._extract_entities(query)
        relations = self._extract_relations(query)
        top_op = self.top_operation(query)

        if targets:
            entities = list(dict.fromkeys(targets + entities))

        if targets and top_op != "vague":
            rewritten = f"{top_op}: {'、'.join(targets)}"
        elif entities and top_op != "vague":
            rewritten = f"{top_op}: {'、'.join(entities)}"
        else:
            rewritten = query
        return rewritten, entities, relations

    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        target = self.extract_target(query)
        if target:
            entities.append(target)
        for pattern in self.entity_patterns:
            entities.extend(re.findall(pattern, query))
        return list(dict.fromkeys(entities))

    def _extract_relations(self, query: str) -> List[str]:
        relations = []
        for pattern in self.relation_patterns:
            relations.extend(re.findall(pattern, query))
        return list(dict.fromkeys(relations)) if relations else []

    def has_action_word(self, query: str) -> bool:
        """对话框输入中是否已包含明确的操作动词。"""
        return self.top_operation(query) != "vague"

    def generate_clarification(self, query: str) -> List[str]:
        """为不完整/模糊输入生成澄清问题（闭环返回入口）。"""
        top_op = self.top_operation(query)
        target = self.extract_target(query)
        questions: List[str] = []
        op_names = {
            "add_file": "添加/新建文件",
            "delete_file": "删除文件",
            "modify_file": "修改文件",
            "refactor": "整理/重构",
            "query": "查询/检索",
            "deep_search": "深度分析/对比",
        }

        if top_op == "vague":
            questions.append("您的指令缺少明确操作。请从以下操作中选择：添加/删除/修改文件、整理重构、查询或深度分析。")
            questions.append("例如：「添加文件 X 内容为...」或「删除文件 X」或「整理文件夹」。")
            return questions

        if not target and top_op in ("add_file", "delete_file", "modify_file"):
            questions.append(f"您想{op_names[top_op]}哪个文件？请提供文件名或路径（如 notes.txt、config.yaml）。")
        if top_op == "add_file" and "内容" not in query and "content" not in query.lower():
            questions.append("新文件的内容是什么？请在指令中补充，例如：「内容为...」")
        if top_op == "modify_file" and "内容" not in query and "content" not in query.lower():
            questions.append("要修改成什么内容？请说明修改后的内容。")
        if top_op == "refactor" and "文件夹" not in query and "目录" not in query:
            questions.append("要整理/重构的对象是哪个文件或文件夹？")
        if top_op == "deep_search" and "对比" not in query and "比较" not in query:
            questions.append("深度分析的主题/对象是什么？需要对比或分析的几个维度是？")

        if not questions:
            questions.append("您的指令信息不足，请补充目标文件或操作内容。")
        return questions

    # --------------------------------------------------------- 子意图识别
    QUERY_SUBINTENT_KEYWORDS = {
        QuerySubIntent.QUERY_METADATA: [
            "信息", "大小", "修改时间", "创建时间", "属性", "元数据", "metadata",
            "文件大小", "文件信息", "属性是什么", "什么时候修改",
        ],
        QuerySubIntent.QUERY_CONTENT: [
            "内容", "写了什么", "里面", "读取", "查看内容", "打开看看",
            "显示内容", "输出内容", "查看一下", "读一下",
        ],
        QuerySubIntent.QUERY_OCR: [
            "识别", "ocr", "文字识别", "提取文字", "图片文字", "截图文字",
            "截屏", "屏幕文字", "扫描", "识别文字", "读取图片",
        ],
        QuerySubIntent.QUERY_SUMMARY: [
            "总结", "摘要", "概括", "归纳", "提炼", "总结一下",
            "简述", "概述", "大意", "概要",
        ],
        QuerySubIntent.QUERY_TRANSLATE: [
            "翻译", "translate", "译成", "转换成", "翻成",
            "英文", "中文", "日文", "韩文", "法文", "德文",
        ],
        QuerySubIntent.QUERY_EXTRACT: [
            "提取", "抽出", "摘取", "抽取", "找出",
            "关键", "重点", "核心", "字段", "信息提取",
        ],
        QuerySubIntent.QUERY_SEARCH: [
            "搜索", "查找", "找", "查", "检索", "查询",
            "在哪里", "有没有", "哪些", "哪些文件",
        ],
        QuerySubIntent.QUERY_SYSTEM: [
            "系统", "程序", "软件", "安装", "启动",
            "红警", "游戏", "应用", "exe",
        ],
        # 新增子意图
        QuerySubIntent.QUERY_FILE_SCAN: [
            "扫描", "遍历", "列举", "统计", "文件扫描", "目录扫描",
        ],
        QuerySubIntent.QUERY_FILE_ANALYZE: [
            "分析", "审查", "检查", "诊断", "评估", "代码分析",
        ],
        QuerySubIntent.QUERY_READONLY_REPORT: [
            "报告", "生成报告", "审查报告", "输出报告",
        ],
    }

    QUERY_SUBINTENT_PRIORITY = {
        QuerySubIntent.QUERY_SUMMARY: ["总结", "摘要", "概括", "简述", "概述", "大意"],
        QuerySubIntent.QUERY_OCR: ["识别", "ocr", "文字识别", "截屏"],
        QuerySubIntent.QUERY_TRANSLATE: ["翻译", "translate", "译成"],
        QuerySubIntent.QUERY_EXTRACT: ["提取", "抽出", "抽取"],
        QuerySubIntent.QUERY_METADATA: ["信息", "大小", "修改时间"],
        QuerySubIntent.QUERY_CONTENT: ["内容", "读取", "查看内容"],
        QuerySubIntent.QUERY_SEARCH: ["搜索", "查找", "检索"],
        QuerySubIntent.QUERY_SYSTEM: ["系统", "程序", "软件"],
        # 新增子意图优先级
        QuerySubIntent.QUERY_FILE_SCAN: ["扫描", "遍历", "列举"],
        QuerySubIntent.QUERY_FILE_ANALYZE: ["分析", "审查", "检查"],
        QuerySubIntent.QUERY_READONLY_REPORT: ["报告", "审查报告", "生成报告"],
    }

    IMAGE_KEYWORDS = [
        "截屏", "截图", "屏幕截图", "屏幕", "screen", "screenshot",
        "图片", "图像", "照片", "photo", "image", "png", "jpg", "jpeg",
        "最近的截屏", "最新的截图", "刚才的截屏", "桌面截屏",
    ]

    def classify_query_subintent(self, query: str) -> QuerySubIntent:
        """对 QUERY 意图进行子意图分类（优先级匹配 + 得分匹配）。"""
        q = query.lower().strip()

        for sub_intent, priority_keywords in self.QUERY_SUBINTENT_PRIORITY.items():
            for kw in priority_keywords:
                if kw.lower() in q:
                    return sub_intent

        scores: Dict[QuerySubIntent, int] = {}
        for sub_intent, keywords in self.QUERY_SUBINTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in q)
            if score > 0:
                scores[sub_intent] = score

        if not scores:
            return QuerySubIntent.QUERY_OTHER

        return max(scores, key=scores.get)

    def extract_slots(self, query: str) -> Dict[str, any]:
        """从查询中提取槽位信息。"""
        slots: Dict[str, any] = {}

        targets = self.extract_targets(query)
        if targets:
            slots[SlotType.FILE_PATH.value] = targets[0]
            if len(targets) > 1:
                slots["file_paths"] = targets

        image_match = self._extract_image_path(query)
        if image_match:
            slots[SlotType.IMAGE_PATH.value] = image_match

        q = query.lower()
        for lang in ["英文", "中文", "日文", "韩文", "法文", "德文", "english", "chinese",
                      "japanese", "korean", "french", "german"]:
            if lang in q:
                slots[SlotType.LANGUAGE.value] = lang
                break

        time_patterns = [
            (r"最近(\d+)(分钟|小时|天|周|月)", "recent"),
            (r"刚才", "recent"),
            (r"今天", "today"),
            (r"昨天", "yesterday"),
            (r"本周", "this_week"),
            (r"本月", "this_month"),
        ]
        for pattern, value in time_patterns:
            if re.search(pattern, q):
                slots[SlotType.TIME_RANGE.value] = value
                break

        format_patterns = [
            (r"json格式|json", "json"),
            (r"csv格式|csv", "csv"),
            (r"markdown|md格式", "markdown"),
            (r"表格", "table"),
            (r"列表", "list"),
        ]
        for pattern, value in format_patterns:
            if re.search(pattern, q):
                slots[SlotType.FORMAT.value] = value
                break

        return slots

    def _extract_image_path(self, query: str) -> Optional[str]:
        """提取图片文件路径。"""
        image_exts = r"\.(png|jpg|jpeg|bmp|tiff|gif|webp)"
        pattern = re.compile(r"[\w.\-\\/]+" + image_exts, re.IGNORECASE)
        matches = pattern.findall(query)
        if matches:
            return matches[0] if isinstance(matches[0], str) else matches[0][0]

        path_pattern = re.compile(r"[\w.\-\\/]+\.[A-Za-z0-9]{1,6}", re.IGNORECASE)
        for match in path_pattern.finditer(query):
            path = match.group()
            if any(ext in path.lower() for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]):
                return path

        return None

    def detect_ocr_need(self, query: str) -> bool:
        """检测查询是否需要OCR功能。"""
        q = query.lower()
        return any(kw in q for kw in self.IMAGE_KEYWORDS)

    def detect_recent_screenshot(self, query: str) -> bool:
        """检测是否提到最近的截屏。"""
        q = query.lower()
        recent_keywords = ["最近", "刚才", "最新", "刚截", "recent", "latest", "last"]
        screenshot_keywords = ["截屏", "截图", "屏幕", "screen", "screenshot"]
        return any(rk in q for rk in recent_keywords) and any(sk in q for sk in screenshot_keywords)

    def extract_readonly_constraints(self, query: str) -> Dict[str, any]:
        """提取只读约束和排除条件。"""
        constraints = {
            "readonly": False,
            "exclude_dirs": [],
            "ignore_files": [],
            "exclude_patterns": [],
        }
        
        q = query.lower()
        
        # 只读检测
        readonly_kws = ["只读", "readonly", "read-only", "不修改", "不移动", 
                        "不删除", "不重命名", "不覆盖"]
        constraints["readonly"] = any(kw in q for kw in readonly_kws)
        
        # 排除目录检测
        exclude_dir_patterns = [
            r"排除目录[：:]?\s*(.+?)(?:[,，;；]|$)",
            r"忽略目录[：:]?\s*(.+?)(?:[,，;；]|$)",
            r"不包含目录[：:]?\s*(.+?)(?:[,，;；]|$)",
        ]
        for pattern in exclude_dir_patterns:
            matches = re.findall(pattern, q)
            constraints["exclude_dirs"].extend(matches)
        
        # 忽略文件检测
        ignore_file_patterns = [
            r"忽略文件[：:]?\s*(.+?)(?:[,，;；]|$)",
            r"排除文件[：:]?\s*(.+?)(?:[,，;；]|$)",
        ]
        for pattern in ignore_file_patterns:
            matches = re.findall(pattern, q)
            constraints["ignore_files"].extend(matches)
        
        # 排除模式检测（如 >5MB、二进制、符号链接）
        exclude_pattern_kws = [
            "符号链接", "只读文件", "权限不足", "二进制", 
            "编码无法识别", ">5MB", "大于5MB",
            "symlink", "readonly", "permission", "binary",
        ]
        constraints["exclude_patterns"] = [
            kw for kw in exclude_pattern_kws if kw in q
        ]
        
        return constraints

    def extract_analysis_params(self, query: str) -> Dict[str, any]:
        """提取分析参数（相似度阈值、函数复杂度等）。"""
        params = {}
        
        q = query.lower()
        
        # 相似度阈值
        similarity_match = re.search(r"相似度\s*(\d+\.?\d*)", q)
        if similarity_match:
            params["similarity_threshold"] = float(similarity_match.group(1))
        
        # 函数复杂度阈值
        complexity_match = re.search(r"复杂度\s*(\d+)", q)
        if complexity_match:
            params["complexity_threshold"] = int(complexity_match.group(1))
        
        # 特殊标记检测
        marker_kws = ["TODO", "FIXME", "XXX", "HACK", "BUG", "WARNING"]
        params["markers"] = [kw for kw in marker_kws if kw.lower() in q]
        
        # git log检测
        if "git log" in q or "git提交" in q or "git提交记录" in q:
            params["include_git_log"] = True
        
        return params