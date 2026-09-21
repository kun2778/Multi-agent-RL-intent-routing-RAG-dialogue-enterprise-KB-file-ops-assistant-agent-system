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