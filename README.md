# AgentRL 意图路由 + RAG 对话 Agent 系统

基于 **多 Agent 强化学习** 的意图路由与 RAG 对话 Agent：用户输入自然语言，系统识别意图并路由到真实模块执行 —— 可 **添加/删除/修改** 任意文件（`.txt/.yaml/.py/.slnx/.docx/.doc/.xml/.mui/.pdf` 等）、**整理重构** 文件夹、对知识库做 **单轮事实查询** 或 **多轮 ReAct 深度检索**，并支持主动澄清闭环与 confirm/rollback 事务语义。

- **意图空间**：12 类主意图 + 12 类子意图 + 12 类槽位 + 23 条执行路径
- **状态观测**：268 维 = 256 维语义特征 + 12 维状态特征
- **双训练架构**：单 Agent（Dueling Double DQN）/ 多 Agent（Meta 仲裁 + 三专家 MAPPO）
- **样本与训练**：10,816 条意图标注样本，默认 20 万步强化训练 + 在线学习

## 系统架构

```
对话框输入(自然语言)
   ↓
查询改写 query_rewriter（目标/槽位提取、澄清问题生成）
   ↓
意图识别 —— 六方置信度仲裁
   ├── LLM 意图分类（DeepSeek 主 → 本地 Ollama 备，端点自动容灾）
   ├── Q 网络（Dueling Double DQN / MARL 策略）
   ├── 分类头（交叉熵有监督）
   ├── 关键词规则
   ├── 检索式投票（标注语料 FAISS 召回，低相似度自动弃权）
   └── MARL Meta 仲裁器（EXP3 在线学习源权重）
   ↓
安全守卫（语义冲突守卫 / 只读降级守卫）
   ↓
路由调度 router.py —— 12 类意图 / 23 条执行路径
   ├── ADD_FILE      → FileOperationHandler  真实写文件到 workspace/
   ├── DELETE_FILE   → FileOperationHandler  移入 workspace/.trash
   ├── MODIFY_FILE   → FileOperationHandler  备份后改写
   ├── REFACTOR      → FileOperationHandler  按类型归类到 code/config/docs/data
   ├── QUERY         → 子意图分发（12 类）→ FactQueryHandler / OCR / 系统文件查找 / 文件扫描…
   ├── DEEP_SEARCH   → DeepSearchHandler     ReAct 思考-行动-观察多轮检索
   ├── FILE_ANALYSIS / FILE_SCAN / REPORT_GENERATE → 只读分析、扫描、审查报告
   ├── FILENAME_QUERY→ SystemFileFinder      按名称/时间筛选文件
   ├── FILELIST_SEARCH→ FileListSearchHandler 指定路径下按条件列文件（云 LLM ReAct 扫描）
   └── VAGUE         → ClarificationHandler  主动澄清 → 用户补充 → 闭环重新路由
   ↓
槽位提取 → 证据充分性评估 → 双条件终止判定
   ↓
每步按真实操作/检索结果计算奖励 → 优先经验回放 → 模型版本化落盘
```

## 技术栈

| 组件 | 实现 |
|------|------|
| **RL 策略（单）** | Dueling Double DQN + 优先经验回放 PER（SumTree，容量 50000）+ LLM/CSV 分类蒸馏 |
| **RL 策略（多）** | Meta 仲裁器 + 意图/子意图/槽位三专家子网络，MAPPO（PPO-Clip）联盟训练 + 联赛对手 + EXP3 在线源权重 |
| **Q 网络** | Looped-Transformer（单 Agent 参数共享 T=6，多 Agent 专家 T=4）+ 8 专家 top-2 Expert-Choice MoE + NSP 潜空间辅助（EMA 目标 + InfoNCE + H 步想象，双架构共用 `nsp_latent.py`）+ Engram 条件记忆 |
| **意图仲裁** | LLM / Q 网络 / 分类头 / 规则 / 检索投票 / MARL **六方加权**，Engram 仅低置信度补充 |
| **动作空间** | 12 类意图、12 类子意图、12 类槽位、23 条执行路径 |
| **状态观测** | 268 维 = 256 维语义特征（字符 n-gram 随机投影）+ 12 维状态向量（MARL 侧为 272 维，含 4 维角色） |
| **向量检索** | `BAAI/bge-small-zh-v1.5` + FAISS（噪声得分降权、top-5 召回），TF-IDF 128 维兜底 |
| **深度检索** | ReAct：子查询拆解 → Thought/Action/Observation 循环 → 证据充分性评估终止（默认 5 轮） |
| **LLM 容灾** | DeepSeek `deepseek-chat` 主端点 → 本地 Ollama `qwen2.5:3b` 备端点，密钥缺失/调用失败自动切换 |
| **服务与部署** | FastAPI + Uvicorn、MCP 审计服务、Docker Compose 6 容器 |
| **语料库** | workspace/ 真实文件 + `corpus_index.json` 注册表（可增删改的持久化模块） |

## 意图与状态空间

`agent_state.py` 定义全部枚举（单副本，两个架构共用）：

| 枚举 | 数量 | 内容 |
|------|------|------|
| `IntentType` | 12 | add_file / delete_file / modify_file / refactor / query / deep_search / vague / file_analysis / file_scan / report_generate / filename_query / filelist_search |
| `QuerySubIntent` | 12 | query_metadata / content / ocr / summary / translate / extract / search / system / other / file_scan / file_analyze / readonly_report |
| `SlotType` | 12 | file_path / file_name / content / keyword / language / format / time_range / image_path / readonly_mode / exclude_dirs / ignore_files / output_format |
| `RoutingPath` | 23 | 7 条主路径 + 8 条 QUERY 子意图路径 + 5 条只读/扫描路径 + filename_query + filelist_search |

**12 维状态观测**（追加 256 维语义特征得 268 维）：

```
[0] 语料库文件数/40     [6] 澄清问题数/3
[1] 成功操作数/20       [7] 平均检索得分
[2] 删除数/10           [8] 查询长度/50
[3] 整理数/10           [9] 回合数/最大回合
[4] 失败操作数/10       [10] 已用时间/预算
[5] 对话框轮数/8        [11] 是否有澄清 (0/1)
```

每次对话框回合通过 `record_operation(kind, target, success, message)` 记录一次状态转移：
操作成功 → 对应计数 +1 且 `last_outcome=xxx_success`；失败 → `failed_ops+1`；回合数 +1，成功或资源耗尽 → `is_terminated=True`。

## 双架构强化学习

### 单 Agent（`agent_rl.py`）

- `TransformerDuelingQNetworkWithNSP`：V/A dueling 头 + 旧版 NSP 头（保留仅为 checkpoint 兼容），输入 268 维，输出 12 动作
- Double DQN 目标（在线网选动作、目标网取值）+ 软更新 τ 退火；NSP 想象 Q 按 α 混入 TD bootstrap
- `PrioritizedReplayBuffer`：SumTree 采样、TD-error 优先级、β 退火、新样本优先级下限防饿死；`sample_segments()` 供 H 步 rollout 取真实轨迹
- 分类头交叉熵有监督训练；`distill_from_llm()` LLM 蒸馏、`train_from_csv()` 标注语料训练

### 多 Agent（`marl_agent.py`）

- `MetaAgent`：6 路仲裁 + MAPPO 联盟训练 + 跨架构交叉投票
- 三专家子网络：`IntentPolicy`（主意图）/ `SubIntentPolicy`（子意图）/ `SlotPolicy`（槽位抽取），观测 272 维
- 训练算法 MAPPO（PPO-Clip，on-policy，rollout=64）+ 联赛（`League`）对手池
- 协调源权重用 EXP3/Hedge 在线学习（`EXP3_LR=0.08`，权重下限 0.25）

### NSP 潜空间（`nsp_latent.py`，双架构共用单副本）

单 Agent 与三专家（`IntentPolicy` / `SubIntentPolicy` / `SlotPolicy`）各挂**一份实例**，
实现在 `nsp_latent.py` 里**只写一份**（`duplicate_gate.py` 按函数相似度 ≥0.82 拦截跨文件重复）。
`build_nsp()` 在 `fork_rng` 里构造，新增参数的初始化不扰动宿主的全局随机流，
因此 `regression_gate` / `test_marl` 的确定性预测不受影响。

| 缺口 | 落点 |
|------|------|
| 目标是「自己预测自己」 | `EMATargetEncoder`：在线编码器的 EMA 冻结副本做 stop-gradient 目标；`update_ema()` 每步之后调用，`ema_drift()` 观测收敛 |
| 容易塌缩 | `InfoNCE`（batch 内负样本，`temperature=0.1`），附带 top-1 命中率 `nsp_infonce_acc` |
| 只预测一步、只当正则项 | `imagine_q_target()`：预测的下一状态潜变量过 `LatentValueBridge` 得到想象 Q，按 α 混进 TD bootstrap（α 在 warmup 内恒 0）；`segment_losses()` 在真实 (H+1) 步轨迹上做自由运行 H 步 rollout（默认 H=3） |
| 确定性点估计 | `StochasticTransitionHead` 输出 (μ, logσ²)，用高斯 NLL + KL(0,1) 训练，采样而非取点 |
| 没有评估指标 | `probe()`：固定文本池 + 钉零状态 + 固定 RNG → 固定输出；`NSPMetrics` 记录训练期滚动均值 |

#### 逐项消融

全有/全无的开关没法归因到"到底是哪个缺口在起作用"，所以档位是**逐项**的，
实现在 `nsp_latent.apply_nsp_ablation(host, spec)` —— 宿主无关，
`AgentRL` 与 `MetaAgent`（三专家同时改）走同一份代码，档位幂等可逆
（每次先 `reset_ablation()` 复位再施加）。

| 档位 | 关掉什么 | 用来验证 |
|------|---------|---------|
| `none` | （无） | 对照组 |
| `off` | 整块 NSP | "NSP 整体有没有用" |
| `no_ema` | `ema_tau=1.0`（目标副本每步跟在线走平） | EMA 目标 vs 自己预测自己 |
| `no_infonce` | `infonce_weight=0` | InfoNCE 是否真的防塌缩（看 `nsp_infonce_acc`） |
| `no_bridge` | `bridge_weight=0` | 价值桥标定是否有贡献 |
| `no_hstep` | `horizon=1` | H 步 rollout vs 一步预测 |
| `no_imagine` | `nsp_imagine_alpha=0` | 想象 Q 混 TD vs 只当正则 |
| `deterministic` | NLL→MSE、KL 恒 0、不采样 | 随机 (μ,σ²) vs 确定点估计 |

```bash
# 单 Agent（steps / log_every 两侧必须完全一致，只改这一个开关）
python train_strong.py 200000 5000 --csv                    # 对照组
python train_strong.py 200000 5000 --csv --nsp-ablate off   # 实验组
python train_strong.py 200000 5000 --csv --nsp-ablate no_hstep

# MARL（steps 解释成自博弈回合数，默认 20）
python train_strong.py --agent marl 20
python train_strong.py --agent marl 20 --nsp-ablate no_imagine

python train_strong.py --eval-only --nsp-ablate none        # 只读对照读数
python train_strong.py --help                               # 档位全表
```

`--ablate-nsp` 是 `--nsp-ablate off` 的别名（保留兼容）。档位**写进 checkpoint**
（`nsp_ablate` 键），续训时会打印"checkpoint 自带档位"提示；旧 checkpoint 没有
该键 → 按 `nsp_enabled` 回退成 `none` / `off`。判据看每类 eval accuracy 与
`nsp_infonce_acc` / `nsp_rollout_err`，探针本身只说明潜空间质量。

#### 读数出口

```bash
python regression_gate.py                 # 报告里多一段 NSP 确定性探针
python train_strong.py --eval-only        # print_nsp：探针 + 训练滚动指标
```

- 旧 checkpoint 没有 `nsp` key → `load()` 容错并 `refresh_target()`；`q_network` 参数结构未变，strict 加载不受影响
- MCP：`get_nsp_metrics`（`agent=single|marl|both`，含历史趋势 `.runtime/nsp_history.jsonl`）

### 共享与隔离

`intent_shared.py` 是两个架构的**单副本上游**（打分聚合、阈值、模板、训练奖励、守卫规则），
`test_cross_arch_isolation.py` + `test_duplicate_gate.py` 保证跨架构引用成环、模板漂移、逐字复制三类问题被 CI 拦截。

## 奖励信号（真实结果驱动，14 类奖励键）

默认表在 `agent_rl.AgentRL.REWARD_MAP`，`config/app.yaml` 的 `intent_llm.reward` 可按键覆盖：

| 事件 | 奖励 | | 事件 | 奖励 |
|------|------|---|---|------|
| 添加成功 `add_success` | +5.0 | | 添加/删除/修改/整理失败 `*_failure` | -4.0 |
| 删除成功 `delete_success` | +5.0 | | 事实查询成功 `query_success` | +5.0 |
| 修改成功 `modify_success` | +5.0 | | 深度检索成功 `deep_search_success` | +8.0 |
| 整理成功 `refactor_success` | +7.0 | | 证据充分 `evidence_sufficient` | +8.0 |
| 澄清解决 `clarification_resolved` | +7.0 | | 证据不足 `insufficient_evidence` | -2.0 |
| 超时/预算耗尽 `timeout` | -3.0 | | | |

**澄清轮次衰减**：带 state 调用时答对键按 `a_k = a_{k-1} - q_k - eps_k` 折算，
`clarification_resolved` 按 `q_k = 1.5 * 0.8^(k-1)` 折算，失败/超时键不衰减，
路径总量满足 `P1 < P2 = P3 > P4 > ... > 答错(-4.0)`，避免策略靠反复澄清刷分。

## 鲁棒性设计

| 机制 | 实现 |
|------|------|
| **六方置信度仲裁** | LLM → 规则 + Q 网络 + 分类头 + 检索投票 + MARL → 保底（`agent_rl.py` `classify_intent`） |
| **语义冲突安全守卫** | `intent_shared.semantic_conflict_guard()`：含只读/查询/文件名关键词的输入禁止路由到写意图 |
| **只读降级守卫** | `detect_readonly_guard()` / `apply_readonly_guard()`：搜索词被误判为删除时强制降级为查询 |
| **LLM 主备容灾** | `llm_intent.py` 逐端点降级：DeepSeek 失败 → 本地 Ollama，全部失败 → 规则兜底 |
| **高危操作事务** | `safety.py`：`needs_confirmation` → `confirm_operation` → `execute_operation`，`backup_before_operation` / `rollback_operation` |
| **澄清闭环** | `clarification.py` + `router.resolve_clarification()`：VAGUE 不终止，补充后回到意图识别重新路由 |
| **双条件终止** | `evidence_evaluator.py`：① 任务完成（操作提交成功 / 证据充分且无矛盾）② 资源预算（回合数或时间超限强制终止）。VAGUE 不触发终止 |
| **运行时预算** | `config/app.yaml` `rl.max_rounds=5`、`rl.retrieval_budget=300s`（代码默认 6/120，配置可覆盖，`rl_config.py` 统一读取） |

## 知识库增强（RAG）

- **切分与入库**：500 字符 / 15% 重叠（75 字符）→ `BAAI/bge-small-zh-v1.5` 向量化 → FAISS `IndexFlatIP`，索引持久化到 `.runtime/faiss`
- **噪声处理**：相似度得分 > 0.7 的噪声模式降权（×0.3）或丢弃，噪声模式表见 `faiss_store.py`
- **召回**：top-5，低相似度弃权避免 OOD 乱投
- **ReAct 深度检索**（`deep_search.py`）：子查询拆解 → Thought / Action / Observation 逐轮循环 → `evidence_evaluator` 判断证据是否充分 → 继续检索或作答
- **意图侧检索**：`intent_retriever.py` 标注语料稠密向量召回，作为第六方之外的独立投票信号（相似度 < 0.30 弃权）

## 落地交付

### FastAPI 服务与 Web Demo

`chat_server.py`（`python -m uvicorn chat_server:app --port 5000` 或双击 `start_chat.bat`）：

- 对话：`POST /api/chat`、`GET|POST /api/conversations`、`DELETE /api/conversations/{id}`、`POST /api/feedback`、`GET /health`
- 持续学习：`POST /api/learning/label|eval-set|dedup|version`、`GET /api/learning/stats|failures|traces|clusters|versions|export`
- Web 对话页：`file_control.html`（前端直连 `127.0.0.1:5000/api/chat`，含思考过程与路由结果展示）

### MCP 审计量化服务（28 个工具）

`mcp_server/` 支持 stdio / SSE 两种传输：

| 分组 | 数量 | 代表工具 |
|------|------|---------|
| 审计 `audit_tools` | 11 | `get_routing_statistics`、`get_routing_accuracy_by_intent`、`analyze_routing_errors`、`detect_anomalies`、`export_audit_trail`、`ocr_recognize_image` |
| 量化 `metrics_tools` | 9 | `get_intent_accuracy`、`get_latency_distribution`（mean/p50/p99）、`get_reward_statistics`、`get_training_progress`、`get_nsp_metrics`、`compare_model_versions`、`benchmark_routing` |
| RL 洞察 `rl_insights` | 8 | `analyze_convergence`、`explain_routing_decision`、`visualize_q_values`、`get_feature_importance`、`analyze_exploration_exploitation`、`analyze_error_patterns` |

### 闭环持续学习（`continuous_learning.py`）

`Trace 记录 → 失败归因（8 类 FailureType，rule/llm/human 三路）→ 四级评估集（TRAIN / VALIDATION / TEST / CHALLENGE）→ 版本化回滚（models/versions/v_*）`

配套 `badcase_replay.py`（线上 bad case 脱敏回流 buffer）与 `regression_gate.py`（回归 gate：0=通过 1=回退 2=缺基线）。

### Docker Compose 部署（6 容器）

`deploy/docker-compose.yml`：

| 服务 | 作用 |
|------|------|
| `agentrl` | 主应用（FastAPI + Agent） |
| `mcp-server` | MCP 审计量化服务 |
| `gateway` | API 网关 |
| `vault` | 密钥管理（OpenAI key 动态下发/轮换） |
| `redis` | 限流 |
| `postgres` | 用量追踪 |

### 代码审查子系统

`review_pipeline.py` + `parallel_scanner.py` + `relation_analyzer.py` + `report_writer.py`：
实测 **5.58s 扫描 139 个文件，产出 385 条分级发现（高 1 / 中 294 / 低 90），循环依赖 0 组**。

## 模块职责

| 模块 | 功能 |
|------|------|
| `main.py` | 命令行对话框交互循环 + 单条指令/演示模式 + 架构切换（single/marl） |
| `agent_state.py` | **状态空间**：11/12/12/22 四类枚举、操作事件记录、12 维状态向量 |
| `agent_rl.py` | **单 Agent**：Dueling Double DQN + PER + 分类头 + 蒸馏 + 14 类奖励 + 六方仲裁 |
| `marl_agent.py` | **多 Agent**：MetaAgent 仲裁 + 三专家策略 + MAPPO + 联赛 + EXP3 |
| `transformer_arch.py` | Looped-Transformer、Expert-Choice MoE、Engram 条件记忆（NSP 头仅为 checkpoint 兼容保留） |
| `nsp_latent.py` | **NSP 潜空间单副本**：EMA 目标 / InfoNCE / 随机转移头 / H 步 rollout / 想象 Q / 探针 / `apply_nsp_ablation` 逐项消融 |
| `intent_shared.py` | 两架构单副本上游：打分聚合、阈值、模板、守卫规则、训练奖励 |
| `llm_intent.py` | LLM 意图分类端点管理（DeepSeek 主 / Ollama 备） |
| `intent_retriever.py` | 检索式意图投票：标注语料稠密向量召回，独立一路信号 |
| `query_rewriter.py` | 操作关键词打分、目标文件(多目标)提取、澄清问题生成 |
| `router.py` | 调度中心：改写 → 分类 → 路由 → 守卫 → 槽位 → 澄清的端到端管线 |
| `safety.py` | 高危操作确认与 confirm/rollback 事务语义 |
| `file_corpus.py` / `file_operators.py` | 语料库增删改、注册表持久化、真实文件操作执行器 |
| `faiss_store.py` / `embedding.py` | RAG 分块、向量化、FAISS 索引、噪声降权 |
| `deep_search.py` | ReAct 多轮深度检索 |
| `filelist_search.py` | 指定路径下的文件列表检索：只读工具（列目录/扫描/过滤/注册表）+ 云 LLM ReAct 调度，无密钥时降级为规则扫描计划 |
| `fact_query.py` / `answer_generator.py` | 单轮向量检索 / 操作结果总结与引用答案 |
| `evidence_evaluator.py` / `clarification.py` | 双条件终止 / 澄清闭环 |
| `continuous_learning.py` | Trace、失败归因、四级评估集、版本化回滚 |
| `regression_gate.py` / `duplicate_gate.py` | 回归 gate / 跨架构重复代码 gate |
| `chat_server.py` | FastAPI 服务与 Web Demo 后端 |
| `train_strong.py` | 训练与评估入口（默认 20 万步、蒸馏、CSV 训练、纯评估） |
| `mcp_server/` | 28 工具 MCP 审计量化服务 |

## 使用方式

```bash
pip install -r requirements.txt        # 安装依赖
python check_deps2.py                  # 检查依赖
python main.py                          # 进入交互对话框（启动自动预热 RL 训练）
python main.py "添加文件 notes.txt 内容为 你好"     # 单条指令
python main.py "对比分析 强化学习.txt 和 python简介.txt"

# Web 对话 Demo（http://127.0.0.1:5000）
start_chat.bat
# 或手动：python -m uvicorn chat_server:app --host 127.0.0.1 --port 5000

# 训练与评估
python train_strong.py                  # 默认 200000 步
python train_strong.py 50000 5000       # 自定义步数与日志间隔
python train_strong.py --distill        # LLM 蒸馏
python train_strong.py --csv            # 标注语料训练（10,816 条）
python train_strong.py --eval-only      # 模板集 + 独立真实样本双评估
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

### 部署

```bash
cd deploy
docker compose up -d          # 6 容器：主应用 / MCP / 网关 / Vault / Redis / PostgreSQL
```

## 验证与 CI

```bash
python test_infra.py              # 基础设施 35 项：原子写 / 读写锁 / 写队列 / 脱敏 / 回归 gate / PER / Engram
python test_intent_shared.py      # 单副本共享层 19 项：打分聚合 / 阈值 / 训练奖励 / 训练-评估隔离
python test_intent_retriever.py   # 检索式意图识别 9 项
python test_marl.py               # 双架构交叉验证（11 条查询用例，逐条断言两架构预测一致）
python test_cross_arch_isolation.py  # 跨架构隔离 5 项：仲裁器无环单例 / 上游拓扑钉死 / 交叉票不回灌
python test_duplicate_gate.py     # 重复代码 gate 10 项
python test_nsp_latent.py         # NSP 潜空间 28 项：确定性探针 / EMA 冻结 / InfoNCE / H 步 rollout / 逐项消融 / checkpoint 兼容
python regression_gate.py         # 回归 gate：0=通过 1=回退 2=缺基线/集合被改
python regression_gate.py --json  # 同上，输出机读报告
```

gate 与**本地缓存状态无关**：运行时会丢弃已构造的检索索引、从语料强制重建，
因此本地（命中过旧缓存）与 CI（冷启动）给出的逐条预测一致；嵌入器的构造逻辑
一旦变化需同步递增 `embedding.py` 的 `schema_version`，否则旧向量会被误认为有效缓存。

基线在 `eval/regression/baseline.json`（MARL 侧 `baseline_marl.json`，重复代码 `duplicate_baseline.json`），
回归集被有意修改时用 `python regression_gate.py --update-baseline` 重新固化。
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
