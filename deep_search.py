import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from agent_state import AgentState, IntentType, RetrievalResult
from query_rewriter import QueryRewriter
from file_corpus import FileCorpus


REACT_SYSTEM_PROMPT = """\
你是 AgentRL 深度搜索引擎。你正在执行一个 ReAct（思考-行动-观察）循环来回答用户问题。

## 已收集的证据
{evidence_context}

## 当前状态
- 已完成检索轮次: {round_num}/{max_rounds}
- 已收集文档数: {doc_count}

## 输出格式
每次输出严格使用以下 JSON 格式之一：

### 如果需要继续搜索：
```json
{{
  "thought": "分析当前证据缺口，说明为什么要搜这个方向",
  "action": "search",
  "search_query": "具体的搜索关键词（中文）"
}}
```

### 如果证据充分可以回答：
```json
{{
  "thought": "综合所有证据，可以给出完整回答",
  "action": "answer",
  "answer": "基于证据的完整回答"
}}
```

## 判断标准
1. 如果已有足够的多角度证据支撑回答 → 选 "answer"
2. 如果证据有明显缺口（如缺少某个维度的信息）→ 选 "search"
3. 搜索查询应该针对缺口，不要重复已搜过的内容
4. 如果已到最大轮次，必须选 "answer" 并基于现有证据回答
"""


class ReActLLMClient:
    """用于 ReAct 循环的轻量 LLM 客户端（复用 config 配置）。"""

    def __init__(self):
        self.base_url = ""
        self.api_key = ""
        self.model = ""
        self.timeout = 30
        self._load_config()

    def _load_config(self):
        try:
            from providers.config_loader import get_config_loader
            loader = get_config_loader()
            app = loader.load_app_config() or {}
            models_cfg = loader.load_models_config() or {}

            gen_cfg = app.get("generation", {})
            intent_cfg = app.get("intent_llm", {})
            provider = intent_cfg.get("provider", "deepseek")
            model = intent_cfg.get("model", "deepseek-chat")

            prov_cfg = (models_cfg.get("providers") or {}).get(provider, {}) or {}
            self.base_url = (prov_cfg.get("base_url") or "").rstrip("/")
            self.model = model
            auth_env = prov_cfg.get("auth_env", "")
            self.api_key = loader.get_secret(auth_env) if auth_env else ""
            if not self.api_key:
                self.api_key = prov_cfg.get("api_key", "")
            self.timeout = int(intent_cfg.get("timeout", 30))
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.available:
            return None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [ReAct LLM] 调用失败: {e}")
            return None


def _extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


class DeepSearchHandler:
    """
    深度检索分支（ReAct 模式）：LLM 驱动的思考-行动-观察循环。

    每一轮：
      1. Thought  —— LLM 分析当前证据，判断是否充分，决定下一步
      2. Action   —— 如果不足：search（指定搜索关键词）；如果充分：answer（生成回答）
      3. Observation —— FAISS 检索结果（或最终回答）
    直到 LLM 判断证据充分 或 达到最大轮次。
    """

    def __init__(self, corpus: FileCorpus = None, max_rounds: int = 5, top_k: int = 5):
        self.corpus = corpus or FileCorpus()
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.query_rewriter = QueryRewriter()
        self.llm = ReActLLMClient()
        self.react_steps: List[Dict[str, str]] = []

    def _build_evidence_context(self, state: AgentState) -> str:
        if not state.evidence:
            return "（暂无检索证据）"
        lines = []
        for i, ev in enumerate(state.evidence, 1):
            files = [d.get("path", "?") for d in ev.documents]
            contents = []
            for d in ev.documents[:3]:
                c = d.get("content", "")[:150]
                if c:
                    contents.append(f"    - {d.get('path', '?')}: {c}")
            lines.append(f"第{i}轮 [查询: {ev.query_used}]")
            lines.append(f"  命中文件: {', '.join(files) if files else '无'}")
            if contents:
                lines.extend(contents)
        return "\n".join(lines)

    def _build_user_prompt(self, query: str, state: AgentState) -> str:
        evidence_ctx = self._build_evidence_context(state)
        return (
            f"用户原始问题: {query}\n\n"
            f"=== 已收集的证据 ===\n{evidence_ctx}\n\n"
            f"当前是第 {state.current_round}/{self.max_rounds} 轮。\n"
            f"请分析证据是否充分，输出下一步行动的 JSON。"
        )

    def _parse_react_action(self, llm_output: str) -> Tuple[str, str, str]:
        """解析 LLM 输出，返回 (action, thought, search_query_or_answer)。"""
        data = _extract_json(llm_output)
        if data and isinstance(data, dict):
            action = data.get("action", "answer").lower()
            thought = data.get("thought", "")
            if action == "search":
                return "search", thought, data.get("search_query", "")
            return "answer", thought, data.get("answer", "")
        if "answer" in llm_output.lower():
            return "answer", llm_output[:200], llm_output
        return "search", "尝试基于关键词搜索", ""

    def _rule_based_fallback(self, query: str, state: AgentState) -> Tuple[str, str]:
        """LLM 不可用时的规则化 ReAct 替代：基于意图分析生成子查询。"""
        sub_queries = self._decompose_query(query, state)
        
        # 收集已检索到的内容关键词
        retrieved_keywords = set()
        for ev in state.evidence:
            for d in ev.documents:
                content = d.get("content", "")
                if content:
                    # 提取有意义的关键词
                    words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', content[:200])
                    retrieved_keywords.update(w.lower() for w in words if len(w) > 1)
        
        # 计算子查询的覆盖度
        covered_count = 0
        for sq in sub_queries:
            sq_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', sq))
            sq_words = {w.lower() for w in sq_words if len(w) > 1}
            # 检查是否有关键词被覆盖
            if sq_words & retrieved_keywords:
                covered_count += 1
        
        # 判断证据是否充分
        coverage_ratio = covered_count / len(sub_queries) if sub_queries else 0
        evidence_count = len(state.evidence)
        
        # 证据充分的条件：
        # 1. 覆盖率超过 60% 且至少 2 轮检索
        # 2. 或者已经检索了所有子查询
        if (coverage_ratio >= 0.6 and evidence_count >= 2) or evidence_count >= len(sub_queries):
            # 生成回答
            all_contents = []
            for ev in state.evidence:
                for d in ev.documents:
                    c = d.get("content", "")
                    if c:
                        all_contents.append(c[:150])
            summary = "\n".join(all_contents[:10])
            return "answer", f"基于 {evidence_count} 轮检索的综合分析：\n{summary}"
        
        # 选择下一个要检索的子查询（跳过已覆盖的）
        for i, sq in enumerate(sub_queries):
            sq_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', sq))
            sq_words = {w.lower() for w in sq_words if len(w) > 1}
            if not (sq_words & retrieved_keywords):
                return "search", sq
        
        # 如果所有子查询都已覆盖，选择当前轮次对应的子查询
        idx = min(state.current_round - 1, len(sub_queries) - 1)
        return "search", sub_queries[idx]

    def _decompose_query(self, query: str, state: AgentState) -> List[str]:
        """分解查询为多个子查询，生成正确的检索主题（不把参数当主题）。"""
        # 先分析意图，获取正确的子查询
        intent_slots = self._analyze_user_intent(query)
        
        # 根据意图类型生成正确的子查询
        task_type = intent_slots.get("task_type", "")
        content_conditions = intent_slots.get("content_conditions", [])
        
        sub_queries = []
        
        # 根据内容条件生成查询
        if "未注释 TODO/FIXME/XXX/HACK" in content_conditions:
            sub_queries.append("查找包含 TODO FIXME XXX HACK 标记的代码文件")
        
        if "import/require/from 依赖" in content_conditions:
            sub_queries.append("分析同目录文件之间的 import require from 依赖关系")
        
        if "相似度 > 0.82" in content_conditions:
            sub_queries.append("检测重复代码片段和相似度超过阈值的文件")
        
        if "复杂度 > 15" in content_conditions:
            sub_queries.append("检测函数复杂度过高的文件和代码质量")
        
        # 根据分析目标生成查询
        analysis_targets = intent_slots.get("analysis_targets", [])
        if "缺失文档" in analysis_targets or "公开 API" in analysis_targets:
            sub_queries.append("分析公开 API 配置键和文档覆盖情况")
        if "孤儿模块" in analysis_targets:
            sub_queries.append("查找未被导入的孤立模块文件")
        if "未使用导出" in analysis_targets:
            sub_queries.append("分析未使用的导出函数和变量")
        
        # Git 过滤相关查询
        git_filters = intent_slots.get("git_filters", [])
        if git_filters:
            sub_queries.append("结合 git log 分析文件修改历史和作者信息")
        
        # 如果没有生成任何查询，使用默认查询
        if not sub_queries:
            sub_queries = [
                "查找包含 TODO FIXME XXX HACK 标记的代码文件",
                "分析文件之间的导入依赖关系",
                "检测函数复杂度过高的文件",
                "分析代码质量和文档覆盖情况"
            ]
        
        return sub_queries
    
    def _extract_core_query(self, query: str) -> str:
        """从查询中提取核心主题，去除参数值和过滤条件。"""
        # 移除阈值参数模式（如"相似度超过0.82"、"复杂度超过15"）
        query = re.sub(r'相似度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'复杂度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'\d+\.?\d*\s*[或和]\s*\d+\.?\d*', '', query)
        
        # 移除过滤扩展名模式（如".tmp/.log/.lock/.min.js/.map"）
        query = re.sub(r'\.[a-z]+/[.\w/]+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\.\w+(?=[\s,，]|$)', '', query)
        
        # 移除目录排除条件
        query = re.sub(r'排除?\s*[\w./]+目录', '', query)
        query = re.sub(r'除\s*[\w./]+\s*之外', '', query)
        
        # 清理多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query if query else "项目分析"

    def _iterative_retrieve(self, query: str, round_num: int) -> RetrievalResult:
        from fact_query import build_retrieval_result
        docs = self.corpus.search(query, self.top_k)
        result = build_retrieval_result(docs, query, round_num)
        print(f"    [FAISS检索] 查询「{query}」返回 {len(result.documents)} 个文档")
        for doc in result.documents[:2]:
            print(f"      - [得分:{doc['score']:.4f}] {doc['path']}")
        return result

    def execute(self, state: AgentState, evaluator, router) -> AgentState:
        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        self.react_steps = []

        print(f"\n  [Deep Search ReAct] 开始 agentic 检索: {query}")
        print(f"  [LLM] {'可用' if self.llm.available else '不可用，使用规则化回退'}")
        
        # 分析用户意图，提取关键槽位
        intent_slots = self._analyze_user_intent(query)
        print(f"  [意图分析] 核心任务: {intent_slots['task_type']}")
        print(f"  [意图分析] 关键主题: {intent_slots['core_topics']}")
        print(f"  [意图分析] 过滤条件: {intent_slots['filters']}")
        print(f"  [意图分析] 输出要求: {intent_slots['output_requirements']}")

        for step_i in range(self.max_rounds):
            if state.elapsed_time > state.retrieval_budget:
                print(f"\n  [终止] 超过时间预算 ({state.retrieval_budget:.1f}s)")
                break

            state.increment_round()
            print(f"\n  === ReAct 第 {state.current_round} 轮 ===")

            if self.llm.available:
                action, thought, payload = self._llm_react_step(query, state)
            else:
                action, payload = self._rule_based_fallback(query, state)
                thought = f"[规则回退] {'搜索: ' + payload if action == 'search' else '生成回答'}"

            self.react_steps.append({
                "step": state.current_round,
                "thought": thought,
                "action": action,
                "detail": payload[:200],
            })

            if action == "answer":
                print(f"  [Thought] {thought[:100]}...")
                print(f"  [Action] → answer")
                if payload:
                    state.terminate(payload)
                break

            search_query = payload if payload else query
            print(f"  [Thought] {thought[:100]}...")
            print(f"  [Action] → search: {search_query}")

            result = self._iterative_retrieve(search_query, state.current_round)
            
            # 过滤噪声文件
            filtered_docs = self._filter_noise_documents(result.documents)
            result.documents = filtered_docs
            
            state.add_evidence(result)

            if evaluator:
                done, reason = evaluator.evaluate(state)
                print(f"  [观察] {reason}")
                if done:
                    break
            else:
                print(f"  [观察] 已收集 {len(state.evidence)} 组证据")

        if not state.final_answer:
            done, reason = evaluator.evaluate(state) if evaluator else (True, "已完成检索")
            if not done:
                print(f"\n  [补充检索] 证据仍不足，执行补充...")
                state.increment_round()
                gap_query = " ".join(
                    doc["content"][:40]
                    for ev in state.evidence for doc in ev.documents[:1]
                ) or query
                result = self._iterative_retrieve(gap_query, state.current_round)
                
                # 过滤噪声文件
                filtered_docs = self._filter_noise_documents(result.documents)
                result.documents = filtered_docs
                
                state.add_evidence(result)

        return state
    
    def _analyze_user_intent(self, query: str) -> dict:
        """分析用户意图，提取关键槽位。结构化输出符合审查要求的意图。"""
        intent_slots = {
            "intent": "",
            "task_type": "",
            "core_topics": [],
            "forbidden_operations": [],
            "allowed_operations": [],
            "scope": "",
            "exclude_dirs": [],
            "exclude_exts": [],
            "exclude_prefixes": [],
            "git_filters": [],
            "content_conditions": [],
            "analysis_targets": [],
            "outputs": [],
            "skip_conditions": [],
            "need_verification": False
        }
        
        q = query.lower()
        
        # ========== 1. 核心任务识别（最高优先级） ==========
        # 识别"只读扫描 + 深度分析 + 新增报告"的核心任务
        if ("只读" in q or "不修改" in q or "禁止修改" in q) and \
           ("扫描" in q or "分析" in q) and \
           ("新增" in q or "创建" in q or "生成" in q or "review_" in q):
            intent_slots["intent"] = "只读扫描并深度分析文件，新增报告目录"
            intent_slots["task_type"] = "只读代码审查"
        elif "扫描" in q and ("分析" in q or "审计" in q):
            intent_slots["intent"] = "扫描并分析项目文件"
            intent_slots["task_type"] = "文件扫描分析"
        elif any(w in q for w in ["对比", "比较", "差异", "vs"]):
            intent_slots["intent"] = "对比分析多个主题"
            intent_slots["task_type"] = "对比分析"
        elif any(w in q for w in ["总结", "报告", "汇总"]):
            intent_slots["intent"] = "生成总结报告"
            intent_slots["task_type"] = "总结报告"
        elif any(w in q for w in ["查找", "搜索", "找"]):
            intent_slots["intent"] = "查找文件或内容"
            intent_slots["task_type"] = "文件查找"
        else:
            intent_slots["intent"] = "深度检索分析"
            intent_slots["task_type"] = "深度检索"
        
        # ========== 2. 禁止操作（最高优先级约束） ==========
        forbidden_keywords = {
            "修改": ["修改", "改", "编辑", "更新", "覆盖", "overwrite"],
            "移动": ["移动", "move"],
            "删除": ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"],
            "重命名": ["重命名", "rename"],
            "覆盖": ["覆盖", "覆写", "overwrite"]
        }
        for op, keywords in forbidden_keywords.items():
            if any(kw in q for kw in keywords):
                intent_slots["forbidden_operations"].append(op)
        
        # 如果明确说"不修改"或"只读"，添加所有禁止操作
        if "只读" in q or "不修改" in q or "禁止修改" in q or "不要修改" in q:
            intent_slots["forbidden_operations"] = ["修改", "移动", "删除", "重命名", "覆盖"]
        
        # ========== 3. 允许操作 ==========
        intent_slots["allowed_operations"] = ["读取", "查询", "分析"]
        if "新增" in q or "创建" in q or "生成" in q or "review_" in q:
            intent_slots["allowed_operations"].append("新增 review_* 目录和报告")
        
        # ========== 4. 作用域 ==========
        if "当前项目" in q or "整个项目" in q:
            intent_slots["scope"] = "当前项目及一级子项目中的文本文件"
        else:
            intent_slots["scope"] = "当前项目中的文件"
        
        # ========== 5. 排除目录 ==========
        exclude_dir_patterns = [
            (r'\.git\b', '.git'),
            (r'node_modules', 'node_modules'),
            (r'\bdist\b', 'dist'),
            (r'\bbuild\b', 'build'),
            (r'\bbackup\b', 'backup'),
            (r'\barchive\b', 'archive'),
            (r'\bvendor\b', 'vendor'),
            (r'__pycache__', '__pycache__'),
            (r'\.venv\b', '.venv'),
        ]
        for pattern, dirname in exclude_dir_patterns:
            if re.search(pattern, q):
                intent_slots["exclude_dirs"].append(dirname)
        
        # ========== 6. 排除扩展名 ==========
        ext_pattern = r'\.(tmp|log|lock|min\.js|map)\b'
        ext_matches = re.findall(ext_pattern, q)
        for ext in ext_matches:
            intent_slots["exclude_exts"].append(f".{ext}")
        
        # ========== 7. 排除前缀 ==========
        if "_draft" in q:
            intent_slots["exclude_prefixes"].append("_draft")
        
        # ========== 8. Git 过滤 ==========
        if ".gitignore" in q:
            intent_slots["git_filters"].append(".gitignore")
        if ".git/info/exclude" in q:
            intent_slots["git_filters"].append(".git/info/exclude")
        if "git log" in q or "git" in q:
            intent_slots["git_filters"].append("最近30天 git log")
        if "ci-bot" in q or "dependabot" in q:
            intent_slots["git_filters"].append("作者不是 ci-bot/dependabot")
        
        # ========== 9. 内容条件 ==========
        content_conditions = []
        if re.search(r'TODO|FIXME|XXX|HACK', q, re.IGNORECASE):
            content_conditions.append("未注释 TODO/FIXME/XXX/HACK")
        if "import" in q or "require" in q or "from" in q or "依赖" in q:
            content_conditions.append("import/require/from 依赖")
        if re.search(r'相似度.*0\.82|0\.82.*相似度', q):
            content_conditions.append("相似度 > 0.82")
        if re.search(r'复杂度.*15|15.*复杂度', q):
            content_conditions.append("复杂度 > 15")
        intent_slots["content_conditions"] = content_conditions
        
        # ========== 10. 分析目标 ==========
        analysis_targets = []
        if "缺失文档" in q or "文档" in q:
            analysis_targets.append("缺失文档")
        if "循环依赖" in q:
            analysis_targets.append("循环依赖")
        if "孤儿" in q:
            analysis_targets.append("孤儿模块")
        if "未使用" in q or "导出" in q:
            analysis_targets.append("未使用导出")
        if "异常" in q:
            analysis_targets.append("未处理异常")
        if "TODO" in q:
            analysis_targets.append("TODO 优先级")
        if "公开 API" in q or "API" in q:
            analysis_targets.append("公开 API")
        if "配置键" in q:
            analysis_targets.append("配置键")
        if "环境变量" in q:
            analysis_targets.append("环境变量")
        if "URL" in q or "url" in q:
            analysis_targets.append("硬编码 URL")
        if "密钥" in q or "key" in q.lower():
            analysis_targets.append("密钥疑似项")
        if "测试" in q:
            analysis_targets.append("测试覆盖引用")
        if not analysis_targets:
            analysis_targets = ["缺失文档", "循环依赖", "孤儿模块", "未使用导出", "未处理异常"]
        intent_slots["analysis_targets"] = analysis_targets
        
        # ========== 11. 输出要求 ==========
        outputs = []
        if "review_" in q:
            outputs.append("review_YYYYMMDD_HHmmss 目录")
        if "Markdown" in q or ".md" in q or "总报告" in q:
            outputs.append("Markdown 总报告")
        if "JSON" in q or ".json" in q or "索引" in q:
            outputs.append("JSON 机器可读索引")
        if "CSV" in q or ".csv" in q or "明细" in q:
            outputs.append("CSV 明细表")
        if "摘要" in q or "分析摘要" in q:
            outputs.append("按原相对路径命名的只读分析摘要文件")
        if "跳过" in q and "原因" in q:
            outputs.append("跳过原因记录")
        if "校验" in q or "验证" in q:
            outputs.append("原文件未改动校验")
        if not outputs:
            outputs = ["review_YYYYMMDD_HHmmss 目录", "Markdown 总报告", "JSON 索引", "CSV 明细"]
        intent_slots["outputs"] = outputs
        
        # ========== 12. 跳过条件 ==========
        skip_conditions = []
        skip_keywords = ["符号链接", "只读文件", "权限不足", "二进制", "编码无法识别", ">5MB", "超过 5MB"]
        for kw in skip_keywords:
            if kw in q:
                skip_conditions.append(kw)
        if not skip_conditions:
            skip_conditions = ["符号链接", "只读文件", "权限不足", "二进制", "编码无法识别", ">5MB"]
        intent_slots["skip_conditions"] = skip_conditions
        
        # ========== 13. 验证要求 ==========
        if "校验" in q or "验证" in q or "不修改原文件" in q:
            intent_slots["need_verification"] = True
        
        return intent_slots
    
    def _extract_core_query(self, query: str) -> str:
        """从查询中提取核心主题。"""
        # 移除阈值参数
        query = re.sub(r'相似度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'复杂度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'\d+\.?\d*\s*[或和]\s*\d+\.?\d*', '', query)
        
        # 移除过滤扩展名
        query = re.sub(r'\.[a-z]+/[.\w/]+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\.\w+(?=[\s,，]|$)', '', query)
        
        # 移除目录排除条件
        query = re.sub(r'排除?\s*[\w./]+目录', '', query)
        query = re.sub(r'除\s*[\w./]+\s*之外', '', query)
        
        # 清理多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query if query else "项目分析"
    
    def _filter_noise_documents(self, documents: List[Dict]) -> List[Dict]:
        """过滤噪声文件，保留高质量文档。"""
        noise_patterns = ['.operation_log.json', '.log', '.tmp', '.lock', '.min.js', '.map']
        
        filtered = []
        for doc in documents:
            path = doc.get("path", "").lower()
            
            # 检查是否是噪声文件
            is_noise = any(noise in path for noise in noise_patterns)
            
            # 检查得分是否足够高
            score = doc.get("score", 0.0)
            
            # 如果是噪声文件但得分很高，仍然保留但降低优先级
            if is_noise and score > 0.5:
                doc["score"] = score * 0.3  # 降低噪声文件的得分
                filtered.append(doc)
            elif not is_noise:
                filtered.append(doc)
        
        # 按得分排序，保留前5个
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return filtered[:5]

    def _llm_react_step(self, query: str, state: AgentState) -> Tuple[str, str, str]:
        """执行一步 LLM ReAct 决策。"""
        system = REACT_SYSTEM_PROMPT.format(
            evidence_context=self._build_evidence_context(state),
            round_num=state.current_round,
            max_rounds=self.max_rounds,
            doc_count=sum(len(ev.documents) for ev in state.evidence),
        )
        user = self._build_user_prompt(query, state)

        llm_output = self.llm.chat(system, user)
        if llm_output:
            print(f"  [LLM输出] {llm_output[:150]}...")
            return self._parse_react_action(llm_output)

        print("  [LLM] 无输出，回退到规则决策")
        action, payload = self._rule_based_fallback(query, state)
        return action, f"[规则回退] {payload}", payload

    def get_react_trace(self) -> List[Dict[str, str]]:
        return self.react_steps

    def get_corpus_stats(self) -> Dict:
        return self.corpus.stats()
