from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time
import numpy as np

from rl_config import get_max_rounds, get_retrieval_budget


class IntentType(Enum):
    """更大的操作意图标签集合 —— RL 动作空间。"""
    ADD_FILE = "add_file"          # 添加/新建/写入文件
    DELETE_FILE = "delete_file"    # 删除/移除文件
    MODIFY_FILE = "modify_file"    # 修改/更新/编辑文件
    REFACTOR = "refactor"          # 整理/重构/重命名/移动/分类
    QUERY = "query"                # 事实查询（单轮向量检索）
    DEEP_SEARCH = "deep_search"    # 分析/对比/综合（多轮深度检索）
    VAGUE = "vague"                # 模糊/不完整（主动澄清闭环）
    FILE_ANALYSIS = "file_analysis"  # 文件深度分析（只读）
    FILE_SCAN = "file_scan"          # 文件扫描（只读）
    REPORT_GENERATE = "report_generate"  # 生成审查报告
    FILENAME_QUERY = "filename_query"    # 按名称搜索文件（时间筛选）
    FILELIST_SEARCH = "filelist_search"  # 按路径+条件列出/找出文件（ReAct 扫描）


class QuerySubIntent(Enum):
    """QUERY 一级意图下的子意图分类。"""
    QUERY_METADATA = "query_metadata"      # 查文件信息（大小、修改时间、类型等）
    QUERY_CONTENT = "query_content"        # 查文件内容（读取文本内容）
    QUERY_OCR = "query_ocr"               # 识别图片文字（OCR）
    QUERY_SUMMARY = "query_summary"        # 总结内容
    QUERY_TRANSLATE = "query_translate"    # 翻译内容
    QUERY_EXTRACT = "query_extract"        # 提取字段/文字
    QUERY_SEARCH = "query_search"          # 搜索文件/信息
    QUERY_SYSTEM = "query_system"          # 查找系统文件/程序
    QUERY_OTHER = "query_other"            # 其他查询
    QUERY_FILE_SCAN = "query_file_scan"          # 文件扫描
    QUERY_FILE_ANALYZE = "query_file_analyze"    # 文件分析
    QUERY_READONLY_REPORT = "query_readonly_report"  # 只读报告


class SlotType(Enum):
    """槽位类型定义。"""
    FILE_PATH = "file_path"        # 文件路径
    FILE_NAME = "file_name"        # 文件名
    CONTENT = "content"            # 内容
    KEYWORD = "keyword"            # 关键词
    LANGUAGE = "language"          # 目标语言（翻译用）
    FORMAT = "format"              # 输出格式
    TIME_RANGE = "time_range"      # 时间范围
    IMAGE_PATH = "image_path"      # 图片路径
    READONLY_MODE = "readonly_mode"    # 只读模式标记
    EXCLUDE_DIRS = "exclude_dirs"      # 排除目录
    IGNORE_FILES = "ignore_files"      # 忽略文件
    OUTPUT_FORMAT = "output_format"    # 输出格式（Markdown/JSON/CSV）


class RoutingPath(Enum):
    """意图到执行路径的映射。"""
    FILE_ADD = "file_add"
    FILE_DELETE = "file_delete"
    FILE_MODIFY = "file_modify"
    FILE_REFACTOR = "file_refactor"
    SINGLE_VECTOR = "single_vector"
    DEEP_SEARCH = "deep_search"
    CLARIFICATION = "clarification"
    # QUERY 子意图路由路径
    QUERY_METADATA = "query_metadata"
    QUERY_CONTENT = "query_content"
    QUERY_OCR = "query_ocr"
    QUERY_SUMMARY = "query_summary"
    QUERY_TRANSLATE = "query_translate"
    QUERY_EXTRACT = "query_extract"
    QUERY_SEARCH = "query_search"
    QUERY_SYSTEM = "query_system"
    # 新增路由路径
    FILE_SCAN = "file_scan"
    FILE_ANALYSIS = "file_analysis"
    REPORT_GENERATE = "report_generate"
    QUERY_FILE_SCAN = "query_file_scan"
    QUERY_FILE_ANALYZE = "query_file_analyze"
    QUERY_READONLY_REPORT = "query_readonly_report"
    FILENAME_QUERY = "filename_query"
    FILELIST_SEARCH = "filelist_search"


SHORT_LABELS = {
    IntentType.ADD_FILE: "添加",
    IntentType.DELETE_FILE: "删除",
    IntentType.MODIFY_FILE: "修改",
    IntentType.REFACTOR: "整理/重构",
    IntentType.QUERY: "事实查询",
    IntentType.DEEP_SEARCH: "深度检索",
    IntentType.VAGUE: "澄清",
    IntentType.FILE_ANALYSIS: "文件分析",
    IntentType.FILE_SCAN: "文件扫描",
    IntentType.REPORT_GENERATE: "报告生成",
    IntentType.FILENAME_QUERY: "文件名搜索",
    IntentType.FILELIST_SEARCH: "文件列表搜索",
}

QUERY_SUBINTENT_LABELS = {
    QuerySubIntent.QUERY_METADATA: "查文件信息",
    QuerySubIntent.QUERY_CONTENT: "查文件内容",
    QuerySubIntent.QUERY_OCR: "OCR识别",
    QuerySubIntent.QUERY_SUMMARY: "内容总结",
    QuerySubIntent.QUERY_TRANSLATE: "翻译内容",
    QuerySubIntent.QUERY_EXTRACT: "提取字段",
    QuerySubIntent.QUERY_SEARCH: "搜索信息",
    QuerySubIntent.QUERY_SYSTEM: "系统查找",
    QuerySubIntent.QUERY_OTHER: "其他查询",
    QuerySubIntent.QUERY_FILE_SCAN: "文件扫描",
    QuerySubIntent.QUERY_FILE_ANALYZE: "文件分析",
    QuerySubIntent.QUERY_READONLY_REPORT: "只读报告",
}


@dataclass
class DialogHistory:
    user_query: str = ""
    assistant_response: str = ""
    timestamp: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    query_used: str = ""
    round_num: int = 0


@dataclass
class RLTransition:
    state: np.ndarray = field(default_factory=lambda: np.zeros(268))
    action: int = 0
    reward: float = 0.0
    next_state: np.ndarray = field(default_factory=lambda: np.zeros(268))
    done: bool = False


@dataclass
class OperationEvent:
    """一次真实文件操作事件（状态转移的触发记录）。"""
    kind: str            # add / delete / modify / refactor / query / deep_search
    target: str          # 目标文件
    success: bool
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Agent 状态，承载 状态转移 逻辑。

    状态转移规则：
      - 每次对话框回合: increment_round()
      - 真实操作发生: record_operation(op_kind, target, success, message)
        -> 更新成功/失败统计、操作历史、last_outcome（奖励键）
      - 操作成功或资源耗尽 -> is_terminated=True
    get_state_vector() 将当前状态编码为 RL 观测（12 维）。
    """

    dialog_history: List[DialogHistory] = field(default_factory=list)
    current_round: int = 0
    # config/app.yaml rl: 段覆盖（缺失时回退代码默认值）
    max_rounds: int = field(default_factory=get_max_rounds)
    retrieval_budget: float = field(default_factory=get_retrieval_budget)
    time_started: float = field(default_factory=time.time)
    current_intent: Optional[IntentType] = None
    current_sub_intent: Optional[QuerySubIntent] = None
    routing_path: Optional[RoutingPath] = None
    evidence: List[RetrievalResult] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)
    # 已解决的澄清轮数：奖励衰减与期望决策的深度 k（k = clarification_rounds + 1）
    clarification_rounds: int = 0
    query_rewrite_history: List[str] = field(default_factory=list)
    is_terminated: bool = False
    final_answer: Optional[str] = None
    rl_rewards: List[float] = field(default_factory=list)
    training_data: List[RLTransition] = field(default_factory=list)

    # ---- 槽位系统 ----
    slots: Dict[str, Any] = field(default_factory=dict)

    # ---- 文件操作相关的实际状态 ----
    corpus_size: int = 0
    operation_history: List[OperationEvent] = field(default_factory=list)
    successful_ops: int = 0
    failed_ops: int = 0
    added_count: int = 0
    deleted_count: int = 0
    modified_count: int = 0
    refactored_count: int = 0
    last_outcome: str = ""

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.time_started

    @property
    def budget_remaining(self) -> float:
        return self.retrieval_budget - self.current_round * (self.retrieval_budget / self.max_rounds)

    def add_dialog(self, user_query: str, assistant_response: str = "", context: Dict[str, Any] = None):
        self.dialog_history.append(DialogHistory(
            user_query=user_query,
            assistant_response=assistant_response,
            timestamp=time.time(),
            context=context or {}
        ))

    def add_evidence(self, result: RetrievalResult):
        self.evidence.append(result)

    def increment_round(self):
        self.current_round += 1

    def add_training_data(self, transition: RLTransition):
        self.training_data.append(transition)

    def add_reward(self, reward: float):
        self.rl_rewards.append(reward)

    def record_operation(self, kind: str, target: str, success: bool, message: str):
        """状态转移：记录一次操作事件并更新统计。"""
        event = OperationEvent(kind=kind, target=target, success=success, message=message)
        self.operation_history.append(event)
        self.increment_round()
        if success:
            self.successful_ops += 1
            if kind == "add":
                self.added_count += 1
            elif kind == "delete":
                self.deleted_count += 1
            elif kind == "modify":
                self.modified_count += 1
            elif kind == "refactor":
                self.refactored_count += 1
        else:
            self.failed_ops += 1
        self.last_outcome = f"{kind}_{'success' if success else 'failure'}"

    def last_operation(self) -> Optional[OperationEvent]:
        return self.operation_history[-1] if self.operation_history else None

    def terminate(self, answer: str = ""):
        self.is_terminated = True
        self.final_answer = answer

    def get_state_vector(self) -> np.ndarray:
        """把当前状态转成 RL 观测向量（12 维，全部为实际状态信号）。"""
        avg_score = np.mean([ev.score for ev in self.evidence]) if self.evidence else 0.0
        query_len = len(self.dialog_history[-1].user_query) if self.dialog_history else 0
        has_clarification = 1.0 if self.clarifications else 0.0
        # successful_ops 已在 record_operation 中累加 add/modify/refactor/delete，
        # 这里再加一遍会把第 2 维高估约一倍（DQN 与 MAPPO 共用本向量，会同时中毒）。
        success_total = self.successful_ops

        return np.array([
            self.corpus_size / 40.0,
            success_total / 20.0,
            self.deleted_count / 10.0,
            self.refactored_count / 10.0,
            self.failed_ops / 10.0,
            len(self.dialog_history) / 8.0,
            len(self.clarifications) / 3.0,
            avg_score,
            query_len / 50.0,
            self.current_round / self.max_rounds,
            self.elapsed_time / self.retrieval_budget,
            has_clarification,
        ], dtype=np.float32)

    def clone(self) -> "AgentState":
        return AgentState(
            dialog_history=self.dialog_history.copy(),
            current_round=self.current_round,
            max_rounds=self.max_rounds,
            retrieval_budget=self.retrieval_budget,
            time_started=self.time_started,
            current_intent=self.current_intent,
            routing_path=self.routing_path,
            evidence=[e for e in self.evidence],
            clarifications=[c for c in self.clarifications],
            clarification_rounds=self.clarification_rounds,
            query_rewrite_history=[q for q in self.query_rewrite_history],
            is_terminated=self.is_terminated,
            final_answer=self.final_answer,
            corpus_size=self.corpus_size,
            operation_history=self.operation_history.copy(),
            successful_ops=self.successful_ops,
            failed_ops=self.failed_ops,
            added_count=self.added_count,
            deleted_count=self.deleted_count,
            modified_count=self.modified_count,
            refactored_count=self.refactored_count,
            last_outcome=self.last_outcome,
        )

    def summary(self) -> str:
        intent = self.current_intent.value if self.current_intent else 'None'
        path = self.routing_path.value if self.routing_path else 'None'
        op_his = ", ".join(f"{e.kind}[{'✓' if e.success else '✗'}]" for e in self.operation_history) or "无"
        return (
            f"=== Agent State Summary ===\n"
            f"回合: {self.current_round}/{self.max_rounds}  时间: {self.elapsed_time:.1f}s\n"
            f"意图: {intent}  路径: {path}\n"
            f"语料库文件: {self.corpus_size}  (添加{self.added_count}/"
            f"删除{self.deleted_count}/修改{self.modified_count}/整理{self.refactored_count}, 失败{self.failed_ops})\n"
            f"证据: {len(self.evidence)} 组, 澄清: {len(self.clarifications)}\n"
            f"操作历史: {op_his}\n"
            f"RL 奖励数: {len(self.rl_rewards)}  最后结果: {self.last_outcome or '无'}\n"
            f"终止: {self.is_terminated}\n"
            f"=========================="
        )