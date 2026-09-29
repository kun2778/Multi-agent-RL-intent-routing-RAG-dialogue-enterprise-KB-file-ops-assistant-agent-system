# AgentRL + Deep Search 对话框文件管理系统

基于 **对话框输入** 的智能文件/语料管理 Agent：用户输入自然语言，AgentRL 识别操作意图
并路由到真实模块执行 —— 可 **添加/删除/修改** 任意文件（`.txt/.yaml/.py/.slnx/.docx/.doc/.xml/.mui/.pdf` 等）、
**整理重构** 文件夹、对语料库做 **单轮事实查询** 或 **多轮深度检索**。

## 架构概览

```
对话框输入(自然语言)
   ↓
AgentRL 意图识别（7 类动作, Q-Learning）
   ├── 添加文件   → FileOperationHandler  真实写文件到 workspace/
   ├── 删除文件   → FileOperationHandler  移入 workspace/.trash
   ├── 修改文件   → FileOperationHandler  备份后改写
   ├── 整理/重构  → FileOperationHandler  按类型归类到 code/config/docs/data
   ├── 事实查询   → FactQueryHandler     单轮向量检索(FAISS/TF-IDF)
   ├── 深度分析   → DeepSearchHandler    多轮迭代检索 → 证据充分性终止
   └── 模糊/不完整 → ClarificationHandler 主动澄清 → 用户补充 → 闭环重新路由
  每步按真实操作/检索结果计算奖励 → 写入经验池 → 终止条件判定
```

## 技术栈

| 组件 | 实现 |
|------|------|
| **RL 策略** | Q-Learning 神经网络 (PyTorch) + Experience Replay + ε-greedy |
| **意图仲裁** | LLM + Q网络 + 分类头 + 关键词规则 + **检索式投票**（标注语料 FAISS 召回）五方加权 |
| **动作空间** | 7 类意图（添加/删除/修改/整理/查询/深度检索/澄清） |
| **状态观测** | 22 维：12 维状态向量 + 7 维操作分 + 目标/实体/长度特征 |
| **向量检索** | TF-IDF 嵌入 + FAISS（缺失时 numpy 回退），支持中文（单字+二元组） |
| **语料库** | workspace/ 真实文件 + corpus_index.json 注册表（可增删改的持久化模块） |
| **LLM** | OpenAI 可选；无 key 时规则合成答案 |

## 模块职责

| 模块 | 功能 |
|------|------|
| `main.py` | 命令行对话框交互循环 + 单条指令/演示模式 |
| `agent_state.py` | **状态转移模块**：7 类意图枚举、操作事件记录、12 维状态向量 |
| `agent_rl.py` | **奖励模块**：18 种奖励事件、Q-Network、自博弈训练、在线学习 |
| `query_rewriter.py` | 操作关键词打分、目标文件(多目标)提取、澄清问题生成 |
| `file_corpus.py` | **语料库模块**：workspace 文件增删改、注册表持久化、向量索引 |
| `file_operators.py` | add/delete/modify/refactor 真实文件操作执行器 |
| `router.py` | 调度中心（菱形节点）：意图→处理器→奖励→终止 |
| `evidence_evaluator.py` | **终止条件模块**：操作提交成功/证据充分/资源预算 双条件终止 |
| `fact_query.py` | 单轮向量检索分支 |
| `intent_retriever.py` | **检索式意图投票**：标注语料稠密向量召回，classify_intent 第四路信号 |
| `deep_search.py` | 多轮迭代检索：子查询拆解→逐轮检索→证据评估→补充检索 |
| `answer_generator.py` | 操作结果总结 / 深度答案+引用 |

## 状态转移（State Transition）

每次对话框回合 → `record_operation(kind, target, success, message)` 记录一次状态转移：

- 操作成功 → 相应计数+1（added/deleted/modified/refactored），`last_outcome=xxx_success`
- 操作失败 → `failed_ops+1`，`last_outcome=xxx_failure`
- 回合数+1；成功或资源耗尽 → `is_terminated=True`

**12 维状态观测**：
```
[0] 语料库文件数/40     [6] 澄清问题数/3
[1] 成功操作数/20       [7] 平均检索得分
[2] 删除数/10           [8] 查询长度/50
[3] 整理数/10           [9] 回合数/最大回合
[4] 失败操作数/10       [10] 已用时间/预算
[5] 对话框轮数/8       [11] 是否有澄清 (0/1)
```

## 奖励信号（真实结果驱动）

| 事件 | 奖励 | | 事件 | 奖励 |
|------|------|---|---|------|
| 添加成功 | +5.0 | | 添加/删除/修改/整理失败 | -4.0 |
| 删除成功 | +5.0 | | 事实查询成功 | +5.0 |
| 修改成功 | +5.0 | | 深度检索成功 | +8.0 |
| 整理成功 | +7.0 | | 证据充分 | +8.0 |
| 正确路由 | +10.0 | | 证据不足 | -2.0 |
| 错误路由 | -5.0 | | 澄清解决 | +7.0 |
| 超时/预算耗尽 | -3.0 | | 无证据 | -4.0 |

## 终止条件（双条件）

1. **任务完成**：文件操作已提交成功，或检索证据充分（且无矛盾）
2. **资源预算**：达到最大回合数 / 超过时间预算 → 强制终止
3. **澄清闭环**：VAGUE 意图不终止，等待用户补充后重新路由

## 使用方式

```bash
pip install -r requirements.txt        # 安装依赖
python check_deps2.py                  # 检查依赖
python main.py                          # 进入交互对话框（启动自动预热 RL 训练）
python demo.bat / run_demo.bat          # 一键演示
python main.py "添加文件 notes.txt 内容为 你好"     # 单条指令
python main.py "对比分析 强化学习.txt 和 python简介.txt"
```

交互对话框示例：

```
你 > 添加文件 report.txt 内容为 深度检索的年度总结报告
你 > 修改文件 config.yaml 内容为 server:\n  port: 9000
你 > 整理文件夹
你 > 搜索 强化学习
你 > 对比分析 强化学习.txt 和 python简介.txt
你 > 列出 / 查看 report.txt / 训练 / 演示 / 退出
```

语料库位于 `workspace/`，删除的文件进入 `.trash`，修改前自动备份到 `.backup`。

## 验证与 CI

```bash
python test_infra.py              # 基础设施：原子写 / 读写锁 / 写队列 / 脱敏 / 回归 gate 判定
python test_intent_retriever.py   # 检索式意图识别 9 项
python test_marl.py               # MARL 交叉验证 11 项
python regression_gate.py         # 回归 gate：0=通过 1=回退 2=缺基线/集合被改
python regression_gate.py --json  # 同上，输出机读报告
```

gate 与**本地缓存状态无关**：运行时会丢弃已构造的检索索引、从语料强制重建，
因此本地（命中过旧缓存）与 CI（冷启动）给出的逐条预测一致；嵌入器的构造逻辑
一旦变化需同步递增 `embedding.py` 的 `schema_version`，否则旧向量会被误认为有效缓存。

基线在 `eval/regression/baseline.json`，回归集被有意修改时用
`python regression_gate.py --update-baseline` 重新固化。
CI 见 `.github/workflows/regression-gate.yml`：先跑单元测试，再跑 gate，任一失败即拦截。

## 路线图（Roadmap）

### 一、增量学习

按依赖关系排序，先建安全网、再接数据回路、最后做稳定性与外部记忆：

| 顺序 | 事项 | 落点 | 状态 |
|------|------|------|------|
| 1 | **补回归 gate**：固定回归集 + 基线 + 阈值 + CI 自动拦截 | `regression_gate.py`（确定性回归集 `eval/regression/regression.jsonl`、基线 `baseline.json`、容忍阈值 + 集合哈希防篡改），`.github/workflows/regression-gate.yml` 自动拦截 | ✅ |
| 2 | **失败数据回流 replay**：线上 bad case 脱敏入 buffer | `redaction.py`（`redact()` 脱敏）+ `badcase_replay.py`（`FAILURE_DIR` → `replay_buffer.push()`，sha1 去重持久化，`main.py` 启动时自动回流，失败不阻塞启动） | ✅ |
| 3 | **持久化 replay buffer 和模型版本** | `agent_rl.py` `save()/load()`：checkpoint 旁挂 `<path>.replay.npz`（上限 20000 条），`save_version/list_model_versions/latest_model_version/load_version` 落 `models/versions/v_*`，原子写 | ✅ |
| 4 | **软更新 + PER 稳定在线策略** | `agent_rl.py`：τ 退火（`tau_init→tau_base`，随 checkpoint 保存）、TD-error 裁剪 + NaN 清洗、新样本优先级下限防饿死、β 退火、loss 非有限值跳过这一步、`SumTree._propagate` 越界保护 | ✅ |
| 5 | **Engram 外部记忆作为补充** | `agent_rl.py` `classify_intent()`：仅在投票置信度低于 `CONFIDENCE_THRESHOLD` 时读取；一致则按 `ENGRAM_WEIGHT` 抬升置信度，不一致需高出 `ENGRAM_MIN_MARGIN` 才改判，高置信度绝不被外部记忆覆盖 | ✅ |

**原则**：gate 未通过不放行新版本；失败样本先脱敏再进 buffer；模型与 buffer 同版本落盘，可回滚。

### 二、并发与文件

按"先保证不损坏、再提升吞吐、最后优化内存"排序：

| 顺序 | 事项 | 落点 | 状态 |
|------|------|------|------|
| 1 | **所有写文件改成 `os.replace` 原子写**（临时文件 + 同目录替换） | `atomic_io.py`（`atomic_write_text/json/bytes/torch_save/savez/append_line`），已覆盖 `chat_server.py`、`continuous_learning.py`、`safety.py`、`faiss_store.py`（含 `index.faiss`）、`intent_retriever.py`、`system_file_finder.py`、`main.py`、`file_corpus.py`、`analyze.py`、`reverse/reverse_moves.py` | ✅ |
| 2 | **加 fsync 和目录 fsync** | `atomic_io.py` `_commit()`：写临时文件 → `fsync` → `os.replace` → `fsync_dir`（NTFS 上目录 fsync 不可用，已按平台降级）；开关 `AGENTRL_FSYNC=0` | ✅ |
| 3 | **全局锁升级为读写锁** | `rwlock.py`（写者优先 `RWLock`）→ `intent_retriever.py`（检索读共享 / 重建与 `add_example` 写独占）、`chat_server.py` `get_system()`（查表读共享 / 首次构造写独占，消除双初始化） | ✅ |
| 4 | **写操作队列化，单写者消费** | `write_queue.py`（单消费者线程 + 未完成计数的 `flush()`），已接入 `continuous_learning` 的读-改-写统计、`safety` 操作日志、`system_file_finder` 缓存；读路径先 `flush()` 再读；`atexit` 收尾 | ✅ |
| 5 | **大文件读用 mmap** | `atomic_io.open_read()`（≥8MiB 走只读 mmap）→ `registry_db.compute_file_hash()`（零拷贝喂哈希器）、`file_corpus/parallel_scanner/faiss_store` 文本读取（`decode_bytes()` 增量解码，mmap 不复制成 bytes）、`mcp_server/ocr_module.py` 图片读取 | ✅ |
| 6 | **PDF 改流式逐页处理** | `pdf_parser.py`：四个后端（pypdf/pdfplumber/fitz/builtin）均逐页累计，达到 `max_chars` 即停（`metadata['truncated']`），不为截断而解压整份 PDF | ✅ |
| 7 | **小文件继续 BytesIO，大文件避免全量入内存** | `atomic_io.open_blob()` 分级（≤8MiB 用 `BytesIO`，更大落临时文件）→ `compression_handler.py` zip/tar 解析、`mcp_server/ocr_module.py` 图片加载 | ✅ |

**原则**：写路径先原子化再谈并发；读路径先流式化再谈缓存。