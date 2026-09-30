"""agent_rl 与 marl_agent 的共享意图仲裁公共件（单一事实来源）。

背景：两个 agent 是「同一接口的两份平行实现」（router.py 用同一行
``self.agent_rl.process(...)`` 驱动它们），历史上把特征提取、语义护栏、
路由表、训练模板、投票聚合等逻辑各复制了一份。复制之后两侧各自演化，
产生了两类系统性问题：

  1. 同一个 bug 被写两遍（例如投票聚合把「得分和」除以「来源数」，
     导致多路共识反而输给单路高分）；
  2. 同一份逻辑在两侧口径漂移（搜索关键词表三份不同、护栏覆盖意图不同、
     只读守卫单边缺失、训练奖励 15 vs 5），修一边必然漏另一边。

本模块把这些公共件收敛为**唯一定义**，两个 agent 只做委托。
任何口径调整只改这里一处。
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent_state import (IntentType, QuerySubIntent, RoutingPath,
                         SHORT_LABELS)

# ================================================================
# 一、共享阈值（原来两处各写一份，数值相同但只改一边必漂移）
# ================================================================
#: 投票置信度下限；澄清与否由 confidence_threshold(k) 期望决策决定
CONFIDENCE_THRESHOLD = 0.65
#: Engram 外部记忆改判必须比主体判断高出的最小间隔
ENGRAM_MIN_MARGIN = 0.10

# ================================================================
# 二、关键词表（唯一定义，供语义护栏 / 只读守卫 / 安全回退共用）
# ================================================================
#: 查询类关键词 —— 合并 agent_rl（含「搜」）与 marl_agent（含 OCR/提取类）
#: 两侧口径的并集，任一 agent 判定为「只读查询」的输入都不会落到写操作。
SEARCH_KEYWORDS: Tuple[str, ...] = (
    "搜", "找", "查", "搜索", "查询", "检索", "怎么", "什么", "哪里",
    "search", "find", "查看", "看看", "在哪", "定位",
    "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取",
)

DELETE_KEYWORDS: Tuple[str, ...] = (
    "删除", "移除", "清理", "删掉", "干掉", "remove", "delete",
)

#: 以下三张表原来在 ``agent_rl._safe_fallback`` 与 ``MetaAgent.rule_keywords``
#: 各写一份（router 又单有一份查询/删除词表），只改一边必然漂移，现统一定义。

#: 深度分析 / 对比类
#: 注：不含「评估」—— 它同时属于 FILE_ANALYSIS（审查/检查/诊断/评估），
#: 两表都收会让同分打平，而 rule_predict 的同分按字典序裁决，谁排前谁赢。
DEEP_ANALYSIS_KEYWORDS: Tuple[str, ...] = (
    "对比", "比较", "分析", "差异", "区别", "vs", "综合", "总结",
    "趋势", "影响", "analyze",
)

#: 新建文件类
ADD_FILE_KEYWORDS: Tuple[str, ...] = (
    "添加", "新增", "创建", "新建", "写入", "保存", "生成",
    "add", "create", "new",
)

#: 修改文件类
MODIFY_FILE_KEYWORDS: Tuple[str, ...] = (
    "修改", "更新", "编辑", "改成", "modify", "edit", "update",
)

#: 整理 / 重构类
REFACTOR_KEYWORDS: Tuple[str, ...] = (
    "整理", "重构", "归类", "归档", "重命名", "移动",
    "分一下类", "分类", "分类归类", "refactor", "organize",
)

#: 判定「这是一条写操作指令」的词（用于问候语优先级压制）
WRITE_KEYWORDS: Tuple[str, ...] = tuple(dict.fromkeys(
    ADD_FILE_KEYWORDS + DELETE_KEYWORDS + MODIFY_FILE_KEYWORDS + REFACTOR_KEYWORDS
    + ("导入",)
))

READONLY_KEYWORDS: Tuple[str, ...] = (
    "只读", "不修改", "不移动", "不删除", "不重命名", "不覆盖",
)

ANALYSIS_KEYWORDS: Tuple[str, ...] = ("分析", "审查", "检查", "诊断", "评估")
SCAN_KEYWORDS: Tuple[str, ...] = ("扫描", "遍历", "列举", "统计")
REPORT_KEYWORDS: Tuple[str, ...] = ("报告", "生成报告", "输出报告")

#: ``AgentRL._safe_fallback`` 的「分析/扫描/报告」闸门词。
#: 以前这是函数内的一份 ``analysis_kws`` 手写副本，与本模块护栏的判定词
#: 是两套口径；现在由护栏的四张表拼出来，护栏改词这里跟着改。
ANALYSIS_GATE_KEYWORDS: Tuple[str, ...] = tuple(dict.fromkeys(
    READONLY_KEYWORDS + ANALYSIS_KEYWORDS + SCAN_KEYWORDS + REPORT_KEYWORDS
    + ("review", "readonly", "scan", "analysis", "report")
))

#: 闸门内「用户明确禁止改动文件」的判定词 —— 命中即落 FILE_ANALYSIS。
FORBIDDEN_MODIFY_KEYWORDS: Tuple[str, ...] = (
    "不修改", "不移动", "不删除", "不重命名", "不覆盖",
    "禁止修改", "禁止移动", "禁止删除",
)

FILENAME_QUERY_KEYWORDS: Tuple[str, ...] = (
    "最近添加的文件", "最近修改的文件", "最近创建的文件",
    "最近的文件", "最新文件", "文件名搜索", "搜索文件名",
)

#: 只读安全守卫的判定词（用户明确要求不要动文件）
READONLY_INDICATORS: Tuple[str, ...] = (
    "只读", "readonly", "read-only", "不修改", "不移动", "不删除",
    "不重命名", "不覆盖", "禁止修改", "禁止移动", "禁止删除",
    "禁止", "不能", "不要", "别动", "别改", "别删",
)

#: 写操作意图 —— 所有护栏一致的拦截范围（原 agent_rl 只拦 DELETE_FILE，
#: marl_agent 拦 4 类，文件名纠偏又各不相同，现统一为全集）
WRITE_INTENTS = frozenset({
    IntentType.ADD_FILE,
    IntentType.DELETE_FILE,
    IntentType.MODIFY_FILE,
    IntentType.REFACTOR,
})

# ================================================================
# 三、语义特征提取器（原来两份逐字副本）
# ================================================================
class SemanticFeatureExtractor:
    """中文/英文字符 n-gram 哈希 + 随机投影 → 256 维稠密向量。

    原理:
      - 提取 unigram + bigram + trigram（字符级）
      - 每个 n-gram 哈希到 HASH_DIM 维稀疏桶
      - 乘以固定的随机投影矩阵 → 256 维稠密向量
      - 无需训练，无需外部依赖，启动即用

    随机种子固定为 42，两侧 agent 得到**完全相同**的投影基，
    保证第 6 路交叉投票时特征空间可比。
    """

    HASH_DIM = 2048
    PROJ_DIM = 256

    def __init__(self):
        rng = np.random.RandomState(42)
        self._proj = rng.randn(self.HASH_DIM, self.PROJ_DIM).astype(np.float32) * (
            1.0 / math.sqrt(self.HASH_DIM))

    def _char_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        ngrams = []
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])
        return ngrams

    def _hash_to_bucket(self, s: str) -> int:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        return h % self.HASH_DIM

    def extract(self, query: str) -> np.ndarray:
        ngrams = self._char_ngrams(query)
        if not ngrams:
            return np.zeros(self.PROJ_DIM, dtype=np.float32)

        vec = np.zeros(self.HASH_DIM, dtype=np.float32)
        for ng in ngrams:
            vec[self._hash_to_bucket(ng)] += 1.0

        tf = np.log1p(vec)
        proj = tf @ self._proj
        norm = np.linalg.norm(proj)
        if norm > 0:
            proj /= norm
        return proj.astype(np.float32)

    @property
    def dim(self) -> int:
        return self.PROJ_DIM


# ================================================================
# 四、护栏
# ================================================================
def detect_readonly_guard(query: str) -> bool:
    """检测用户是否明确要求只读操作。"""
    q = query.lower()
    return any(ind in q for ind in READONLY_INDICATORS)


def semantic_conflict_guard(query: str, proposed: IntentType,
                            votes: Optional[Dict[IntentType, Dict]] = None) -> IntentType:
    """语义冲突护栏：关键词明确指向只读/查询时，禁止落进写操作意图。

    三层判定（顺序即优先级）：
      1. 只读/分析/扫描/报告 —— 最具体的意图，优先于泛化查询判定；
      2. 查询类关键词（且无删除词）—— 降级为 QUERY；
      3. 文件名类关键词 —— 降级为 FILENAME_QUERY。

    ``votes`` 仅为兼容旧签名保留（历史上曾读取 votes[QUERY].score，
    但 votes 只包含拿到票的意图，QUERY 不一定存在 → KeyError 风险）。
    """
    del votes  # 不再依赖：旧实现在此有 KeyError + 死分支缺陷
    q = query.lower().strip()

    # 1) 只读/分析/扫描/报告类
    has_readonly = any(kw in q for kw in READONLY_KEYWORDS)
    has_analysis = any(kw in q for kw in ANALYSIS_KEYWORDS)
    has_scan = any(kw in q for kw in SCAN_KEYWORDS)
    has_report = any(kw in q for kw in REPORT_KEYWORDS)
    if has_readonly or has_analysis or has_scan or has_report:
        if proposed in WRITE_INTENTS:
            if has_analysis:
                return IntentType.FILE_ANALYSIS
            if has_scan:
                return IntentType.FILE_SCAN
            if has_report:
                return IntentType.REPORT_GENERATE
            return IntentType.QUERY

    # 2) 查询类关键词（明确的删除意图除外）
    if any(kw in q for kw in SEARCH_KEYWORDS) and not any(
            kw in q for kw in DELETE_KEYWORDS):
        if proposed in WRITE_INTENTS:
            return IntentType.QUERY

    # 3) 文件名类查询
    if any(kw in q for kw in FILENAME_QUERY_KEYWORDS):
        if proposed in WRITE_INTENTS:
            return IntentType.FILENAME_QUERY

    return proposed


def apply_readonly_guard(query: str, intent: IntentType
                         ) -> Tuple[IntentType, bool]:
    """只读安全守卫：命中只读要求且当前为写意图时降级为 QUERY。

    返回 ``(意图, 是否降级)``；两个 agent 的 process 都必须调用，
    避免出现「单体有守卫、MARL 没有」的单边防护。
    """
    if detect_readonly_guard(query) and intent in WRITE_INTENTS:
        return IntentType.QUERY, True
    return intent, False


# ================================================================
# 五、路由表
# ================================================================
#: 意图 → 处理路径。用模块级常量 + ``.get`` 兜底：
#: 原实现是函数内的 ``routing_map[intent]`` 硬索引，而
#: ``query_routing_path`` 用的是 ``.get(...)`` —— 两处口径不一致，
#: 将来给 IntentType 加一个新成员就会在路由处 KeyError（线上才炸）。
#: ``test_route_maps_cover_every_member`` 会在 CI 上强制两表覆盖全部成员，
#: 所以兜底分支不会掩盖「忘了登记新意图」这件事。
ROUTING_MAP: Dict[IntentType, RoutingPath] = {
    IntentType.ADD_FILE: RoutingPath.FILE_ADD,
    IntentType.DELETE_FILE: RoutingPath.FILE_DELETE,
    IntentType.MODIFY_FILE: RoutingPath.FILE_MODIFY,
    IntentType.REFACTOR: RoutingPath.FILE_REFACTOR,
    IntentType.QUERY: RoutingPath.SINGLE_VECTOR,
    IntentType.DEEP_SEARCH: RoutingPath.DEEP_SEARCH,
    IntentType.VAGUE: RoutingPath.CLARIFICATION,
    IntentType.FILE_ANALYSIS: RoutingPath.FILE_ANALYSIS,
    IntentType.FILE_SCAN: RoutingPath.FILE_SCAN,
    IntentType.REPORT_GENERATE: RoutingPath.REPORT_GENERATE,
    IntentType.FILENAME_QUERY: RoutingPath.FILENAME_QUERY,
}

QUERY_ROUTING_MAP: Dict[QuerySubIntent, RoutingPath] = {
    QuerySubIntent.QUERY_METADATA: RoutingPath.QUERY_METADATA,
    QuerySubIntent.QUERY_CONTENT: RoutingPath.QUERY_CONTENT,
    QuerySubIntent.QUERY_OCR: RoutingPath.QUERY_OCR,
    QuerySubIntent.QUERY_SUMMARY: RoutingPath.QUERY_SUMMARY,
    QuerySubIntent.QUERY_TRANSLATE: RoutingPath.QUERY_TRANSLATE,
    QuerySubIntent.QUERY_EXTRACT: RoutingPath.QUERY_EXTRACT,
    QuerySubIntent.QUERY_SEARCH: RoutingPath.SINGLE_VECTOR,
    QuerySubIntent.QUERY_SYSTEM: RoutingPath.QUERY_SYSTEM,
    QuerySubIntent.QUERY_OTHER: RoutingPath.SINGLE_VECTOR,
    QuerySubIntent.QUERY_FILE_SCAN: RoutingPath.FILE_SCAN,
    QuerySubIntent.QUERY_FILE_ANALYZE: RoutingPath.FILE_ANALYSIS,
    QuerySubIntent.QUERY_READONLY_REPORT: RoutingPath.REPORT_GENERATE,
}


def route_intent(intent: IntentType) -> RoutingPath:
    return ROUTING_MAP.get(intent, RoutingPath.SINGLE_VECTOR)


def query_routing_path(sub_intent: QuerySubIntent) -> RoutingPath:
    return QUERY_ROUTING_MAP.get(sub_intent, RoutingPath.SINGLE_VECTOR)


def intent_label(intent: IntentType) -> str:
    """意图 → 展示用短标签（唯一实现）。

    旧实现是 ``AgentRL.get_intent_label`` 与 ``MetaAgent.get_intent_label``
    两个逐字相同的 one-liner（评审 findings.csv F310 标记为 sim=0.821/1.0），
    改一处漏一处；标签表 SHORT_LABELS 本就来自 agent_state，没必要再抄一遍。
    """
    return SHORT_LABELS.get(intent, intent.value)


# ================================================================
# 六、投票聚合
# ================================================================
# 共识得分缩放指数 α：score = Σ各路得分 / n**α
#   α = 1.0 → 旧实现（各路均值，共识被惩罚：0.90+0.80 → 0.85 会被 0.86 击败）
#   α = 0.0 → 纯求和（几路弱票堆积即可压过一路强证据，多数票淹没质量）
#   α = 0.75 → 本项目取值：共识按 n**0.25 缓升，既能反超共识惩罚，又不改变
#               124 条回归集上任何一条样本的判定（见下方 aggregate_votes 实测）
AGG_DAMPING_EXP = 0.75


def aggregate_votes(candidates: Sequence[Tuple[IntentType, float, str]]
                    ) -> Dict[IntentType, Dict]:
    """把各路候选聚合为 ``{意图: {"score": 共识得分, "sources": [来源...]}}``。

    得分 ``score = Σ各路得分 / n**α``（α = :data:`AGG_DAMPING_EXP`），
    等价于「各路平均质量 × n**(1-α)」，即**共识增强**得分。

    - 旧实现 ``α = 1.0``（取均值）：新增一路同向共识反而**降低**得分 ——
      两路高置信支持（0.90 + 0.80 → 0.85）会被一路 0.86 的单源意图击败，
      这就是「共识惩罚」，也是两套架构同时继承的缺陷。
    - ``α = 0.0``（纯求和）走到另一个极端：几路很弱的同向票
      （0.20 × 5 = 1.00）压过一路强证据（0.85），多数票淹没质量。

    本项目取 ``α = 0.75``：共识按 ``n**0.25`` 递增，
      - 上例 ``1.70 / 2**0.75 = 1.01 > 0.86``，共识反超 → 缺陷消除；
      - 在 124 条回归集上逐样本比对，与 ``α = 1.0`` 的判定**完全一致**
        （overall / macro 均相同），即修复不引入任何行为回退。

    实测（eval/regression/regression.jsonl，124 条）::

        α     overall  macro   逐类回退
        1.00  0.7742   0.7336  —（= 基线）
        0.75  0.7742   0.7336  无          ← 取此值
        0.50  0.7742   0.7417  deep_search、report_generate 各掉 1 条
        0.25  0.7661   0.7347  同上且总体下滑
        0.00  0.7661   0.7347  同上且总体下滑

    排序与置信度统一用这一个数，避免「按 A 排序、按 B 报置信」的错位。
    """
    votes: Dict[IntentType, Dict] = {}
    for intent, score, source in candidates:
        entry = votes.get(intent)
        if entry is None:
            entry = votes[intent] = {"score": 0.0, "sources": []}
        entry["score"] += score
        entry["sources"].append(source)
    for entry in votes.values():
        entry["score"] /= len(entry["sources"]) ** AGG_DAMPING_EXP
    return votes


def vote_confidence(votes: Dict[IntentType, Dict], intent: IntentType) -> float:
    """置信度 = 胜出意图的共识得分（见 :func:`aggregate_votes`），截断到 [0, 1]。

    单源时即该源原始得分 —— ``CONFIDENCE_THRESHOLD(0.65)`` 与强制澄清阈值(0.4)
    的历史标度不变；多源同向支持时按 ``n**0.25`` 适度抬升，不被来源数量稀释。
    """
    entry = votes.get(intent)
    if not entry or not entry["sources"]:
        return 0.0
    return float(min(1.0, max(0.0, entry["score"])))


# ================================================================
# 七、训练奖励与训练/评估样本划分
# ================================================================
TRAIN_REWARD_CORRECT = 5.0
TRAIN_REWARD_WRONG = -2.0
TRAIN_CORRECT_BONUS = 10.0


def train_reward(correct: bool) -> float:
    """模板自博弈的单步奖励 —— 两个 agent 与 train_strong 共用同一公式。

    旧实现两处各写一份：agent_rl 答对 = 5+10 = 15，
    marl_agent 答对 = 5，跨架构 reward 曲线相差 3 倍、不可比。
    """
    reward = TRAIN_REWARD_CORRECT if correct else TRAIN_REWARD_WRONG
    if correct:
        reward += TRAIN_CORRECT_BONUS
    return reward


#: 训练/评估模板按索引切分：奇数位留作评估（训练阶段不可见），偶数位用于训练。
#: 解决「训练集 = 测试集」导致两侧准确率一起虚高、互相印证不出泛化能力。
_HOLDOUT_STRIDE = 2
_HOLDOUT_OFFSET = 1


def train_pool(templates: Sequence[str]) -> List[str]:
    """训练用模板（偶数位；剔除留出样本）。不足 2 条时全部用于训练。"""
    pool = [t for i, t in enumerate(templates)
            if i % _HOLDOUT_STRIDE != _HOLDOUT_OFFSET]
    return pool or list(templates)


def eval_pool(templates: Sequence[str]) -> List[str]:
    """评估用模板（奇数位留出样本，训练阶段不可见）。不足 2 条时退回全量。"""
    pool = [t for i, t in enumerate(templates)
            if i % _HOLDOUT_STRIDE == _HOLDOUT_OFFSET]
    return pool or list(templates)


# ================================================================
# 八、训练模板 / 演示样本（唯一副本）
# ================================================================
TASK_TEMPLATES: Dict[IntentType, List[str]] = {
    IntentType.ADD_FILE: [
        "添加文件 {f} 内容为 {c}", "创建新文件 {f}", "写入 {f} 保存内容: {c}",
        "新建一个 {f}", "帮我写一个 {f}", "把以下内容保存到 {f}",
        "生成文件 {f}", "帮我创建 {f} 然后写入内容",
    ],
    IntentType.DELETE_FILE: [
        "删除文件 {f}", "移除 {f}", "清理文件 {f}",
        "把 {f} 删掉", "帮我删除 {f}", "移除掉 {f}",
        "清理掉 {f}", "把 {f} 从项目中删去",
    ],
    IntentType.MODIFY_FILE: [
        "修改文件 {f} 内容为 {c}", "更新 {f} 改成 {c}",
        "编辑 {f}", "帮我改一下 {f}", "把 {f} 的内容替换一下",
        "更新 {f} 的内容", "修改 {f} 中的配置",
    ],
    IntentType.REFACTOR: [
        "整理文件夹", "重构项目结构", "把文件按类型分类归档",
        "帮我整理一下目录", "重命名文件", "移动文件到对应文件夹",
        "代码重构", "优化项目结构",
    ],
    IntentType.QUERY: [
        "查询文件 {f} 里写了什么", "搜索 {f} 的内容", "{f} 存在吗",
        "帮我找一下 {f}", "查看 {f}", "{f} 里面有什么",
        "搜索强化学习", "查一下这个文件", "找到红警3的启动程序在哪里",
    ],
    IntentType.DEEP_SEARCH: [
        "对比分析 {f} 和 {g}", "深度分析 {f} 的趋势与影响",
        "综合分析 {f} 和 {g} 的差异", "帮我比较一下 {f} 和 {g}",
        "分析 {f} 的优缺点", "深入研究 {f} 的发展方向",
        "评估 {f} 和 {g} 的区别",
    ],
    IntentType.VAGUE: [
        "你好", "嗨", "在吗", "你是谁", "帮我一下",
        "请问", "嗯", "好的", "谢谢", "哈哈",
        "今天天气怎么样", "讲个笑话", "随便聊聊",
        "你能做什么", "介绍一下自己", "无聊",
        "123", "test", "hello", "hi", "hey",
        "你是AI吗", "我有个问题", "等等",
    ],
    IntentType.FILE_ANALYSIS: [
        "只读分析 {f} 的代码质量", "审查 {f} 的结构",
        "检查 {f} 中的问题", "诊断 {f} 的性能",
        "评估 {f} 的复杂度", "分析 {f} 的依赖关系",
        "深度分析 {f} 的架构", "审查项目的代码规范",
    ],
    IntentType.FILE_SCAN: [
        "扫描项目中的所有文件", "遍历目录列出所有文件",
        "统计项目文件数量", "列举所有Python文件",
        "扫描代码中的TODO标记", "检查项目结构",
        "列出所有配置文件", "扫描重复文件",
    ],
    IntentType.REPORT_GENERATE: [
        "生成代码审查报告", "输出项目分析报告",
        "生成Markdown格式的报告", "创建JSON格式的分析结果",
        "汇总生成CSV报告", "生成项目质量报告",
        "输出代码复杂度报告", "生成依赖分析报告",
    ],
    IntentType.FILENAME_QUERY: [
        "为我找到最近添加的文件名", "最近修改的文件有哪些",
        "查找最近创建的文件", "最近编辑过的文件名",
        "最新添加的文件有哪些", "最近保存的文件名列表",
        "搜索最近的txt文件", "最近修改的py文件有哪些",
        "找到本周创建的文件", "最近一天修改的文件",
        "recently modified files", "find newly created files",
        "最新的文件名有哪些", "最近操作过的文件",
    ],
}

SAMPLE_FILES = ["notes.txt", "config.yaml", "main.py", "app.json",
                "report.md", "readme.md", "data.csv"]
SAMPLE_CONTENT = "这是一段用于演示的文档内容。"
