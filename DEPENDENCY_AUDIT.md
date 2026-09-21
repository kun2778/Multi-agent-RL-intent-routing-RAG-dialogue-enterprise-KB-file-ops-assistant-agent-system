# AgentRL_IntentRouting 运行时本地文件依赖审计报告

**审计时间**: 2026-09-20  
**目标入口**: `C:\Users\84950\Desktop\AgentRL_IntentRouting`

---

## 一、本地 Python 模块依赖 (23个)

### 1.1 核心执行链路

| 模块文件 | 引用源 | 运行时必需 |
|---|---|---|
| `agent_state.py` | main, router, agent_rl, marl_agent, deep_search, file_operators, fact_query, evidence_evaluator, answer_generator, ocr_handler, query_rewriter | ✅ |
| `agent_rl.py` | main, router | ✅ |
| `marl_agent.py` | main, router | ✅ |
| `router.py` | main | ✅ |
| `answer_generator.py` | main | ✅ |
| `file_corpus.py` | main, router, deep_search, fact_query | ✅ |
| `query_rewriter.py` | agent_rl, marl_agent, deep_search | ✅ |
| `llm_intent.py` | agent_rl, marl_agent | ✅ |
| `file_operators.py` | router | ✅ |
| `fact_query.py` | router, deep_search | ✅ |
| `deep_search.py` | router | ✅ |
| `clarification.py` | router | ✅ |
| `evidence_evaluator.py` | router | ✅ |
| `system_file_finder.py` | router | ✅ |
| `ocr_handler.py` | router | ✅ |
| `safety.py` | file_operators | ✅ |
| `registry_db.py` | file_corpus, parallel_scanner | ✅ |
| `parallel_scanner.py` | file_corpus | ✅ |
| `faiss_store.py` | file_corpus (延迟导入) | ✅ |

### 1.2 Providers 子系统

| 模块文件 | 引用源 | 运行时必需 |
|---|---|---|
| `providers/__init__.py` | - | ✅ |
| `providers/config_loader.py` | llm_intent, deep_search | ✅ |
| `providers/base.py` | 所有 adapter | ✅ |
| `providers/registry.py` | - | ✅ |
| `providers/schemas.py` | - | ✅ |
| `providers/adapters/__init__.py` | providers | ✅ |
| `providers/adapters/openai_adapter.py` | providers | ✅ |
| `providers/adapters/anthropic_adapter.py` | providers | ✅ |
| `providers/adapters/deepseek_adapter.py` | providers | ✅ |
| `providers/adapters/gemini_adapter.py` | providers | ✅ |
| `providers/adapters/local_adapter.py` | providers | ✅ |
| `providers/secrets/__init__.py` | - | ❌ |
| `providers/secrets/base.py` | - | ❌ |

### 1.3 辅助/可选模块

| 模块文件 | 引用源 | 运行时必需 |
|---|---|---|
| `chat_server.py` | - (独立启动) | ❌ |
| `train_strong.py` | - (独立脚本) | ❌ |
| `test_marl.py` | - (测试) | ❌ |
| `rebuild_index.py` | - (工具脚本) | ❌ |
| `check_deps2.py` | - (检查脚本) | ❌ |
| `crawler.py` | - (爬虫) | ❌ |
| `mcp_client_example.py` | - (示例) | ❌ |
| `gateway/main.py` | - (独立服务) | ❌ |
| `gateway/*.py` | - | ❌ |
| `mcp_server/*.py` | ocr_handler (延迟) | ❌ |
| `datasets/generate_corpus*.py` | - (数据生成) | ❌ |
| `reverse/reverse_moves.py` | - (工具) | ❌ |

---

## 二、配置文件依赖

| 文件路径 | 引用源 | 读取方式 | 必需 |
|---|---|---|---|
| `config/app.yaml` | providers/config_loader.py:28 | `yaml.safe_load()` | ✅ |
| `config/models.yaml` | providers/config_loader.py:26 | `yaml.safe_load()` | ✅ |
| `.env` | providers/config_loader.py:48 | `dotenv.load_dotenv()` | ✅ |
| `config/secrets.yaml` | providers/config_loader.py:32 | `yaml.safe_load()` | ❌ |
| `config/gateway.yaml` | providers/config_loader.py:34 | `yaml.safe_load()` | ❌ |
| `mcp_server/config.yaml` | mcp_server/ocr_module.py:256 | `yaml.safe_load()` | ❌ |

---

## 三、数据文件/模型依赖

| 文件路径 | 引用源 | 操作 | 必需 |
|---|---|---|---|
| `models/marl_agents.pth` | marl_agent.py:44 | `torch.save()/load()` | ✅ |
| `models/agent_rl.pth` | agent_rl.py:29 | `torch.save()/load()` | ✅ |
| `~/.agentrl/corpus_*.db` | file_corpus.py:196-199 | `sqlite3.connect()` | ✅ |
| `~/.agentrl/.faiss_store/default/index.faiss` | faiss_store.py:532 | `faiss.write_index()` | ✅ |
| `~/.agentrl/.faiss_store/default/chunks.json` | faiss_store.py:536 | `json.dump()` | ✅ |
| `workspace/` | file_corpus.py:173 | `os.scandir`/文件IO | ✅ |
| `workspace/.operation_log.json` | safety.py:49 | `json.load()/dump()` | ✅ |
| `workspace/.trash/` | file_corpus.py:187 | `shutil.move()` | ✅ |
| `workspace/.backup/` | file_corpus.py:188 | `shutil.copy2()` | ✅ |
| `%TEMP%/agentrl_pf_cache.json` | system_file_finder.py:271 | `json.load()/dump()` | ✅ |
| `intent_corpus.csv` | agent_rl.py:706 | `csv.DictReader` | ❌ (训练用) |
| `datasets/intent_corpus.csv` | agent_rl.py:709 | 同上 | ❌ |
| `corpus_index.json` | file_corpus.py | `json` | ❌ |

---

## 四、subprocess 调用

| 可执行文件 | 调用源 | 方式 | 必需 |
|---|---|---|---|
| `powershell` | system_file_finder.py:219-222 | `subprocess.run()` | ✅ (快捷方式扫描) |

**临时文件生成**:
- `%TEMP%/_lnk_out.txt` - PowerShell 输出
- `%TEMP%/_lnk_scan.ps1` - PowerShell 脚本

---

## 五、动态库加载

**无 ctypes/cffi 直接调用**。所有动态库依赖通过第三方包间接引入：
- `faiss` (faiss-cpu/faiss-gpu)
- `torch` (PyTorch)
- `numpy`

---

## 六、环境变量依赖

| 环境变量 | 引用源 | 必需 |
|---|---|---|
| `DEEPSEEK_API_KEY` | adapters/deepseek | ✅ (默认 provider) |
| `OPENAI_API_KEY` | adapters/openai | ❌ (按需) |
| `ANTHROPIC_API_KEY` | adapters/anthropic | ❌ (按需) |
| `GEMINI_API_KEY` | adapters/gemini | ❌ (按需) |
| `LOCAL_LLM_URL` | adapters/local | ❌ (备选) |
| `AGENT_TYPE` | main.py:37 | ❌ (默认 "marl") |

---

## 七、执行依赖图

```
main.py ─────────────────────────────────────────────────────
│
├─ agent_state.py ──────────────── 数据结构定义
│
├─ agent_rl.py ─────────────────── 单Agent RL
│   ├─ query_rewriter.py ───────── 查询改写
│   ├─ llm_intent.py ───────────── LLM意图分类
│   │   └─ providers/config_loader.py
│   │       ├─ config/app.yaml ──── 运行时配置
│   │       ├─ config/models.yaml ─ 模型配置
│   │       └─ .env ─────────────── API密钥
│   └─ models/agent_rl.pth ──────── 模型权重
│
├─ marl_agent.py ───────────────── 多Agent RL
│   ├─ query_rewriter.py
│   ├─ llm_intent.py
│   └─ models/marl_agents.pth ───── 模型权重
│
├─ router.py ───────────────────── 路由调度
│   ├─ file_operators.py ────────── 文件操作
│   │   └─ safety.py ───────────── 安全控制
│   │       └─ workspace/.operation_log.json
│   ├─ fact_query.py ────────────── 事实查询
│   ├─ deep_search.py ───────────── 深度检索
│   ├─ clarification.py ─────────── 澄清处理
│   ├─ evidence_evaluator.py ────── 证据评估
│   ├─ system_file_finder.py ────── 系统查找
│   │   ├─ subprocess(powershell)
│   │   └─ %TEMP%/agentrl_pf_cache.json
│   └─ ocr_handler.py ───────────── OCR处理
│
├─ answer_generator.py ─────────── 答案生成
│
└─ file_corpus.py ──────────────── 文件语料库
    ├─ registry_db.py ───────────── SQLite注册表
    │   └─ ~/.agentrl/corpus_*.db
    ├─ parallel_scanner.py ──────── 并行扫描
    └─ faiss_store.py ───────────── FAISS向量存储
        ├─ ~/.agentrl/.faiss_store/default/index.faiss
        └─ ~/.agentrl/.faiss_store/default/chunks.json
```

---

## 八、风险点

| 风险 | 严重度 | 说明 |
|---|---|---|
| `models/*.pth` 缺失 | 中 | 首次运行自动训练，无预训练模型则准确率低 |
| `~/.agentrl/` 目录权限 | 中 | SQLite/FAISS 存储在用户目录，权限问题导致启动失败 |
| `.env` 密钥为空 | 低 | LLM 意图分类不可用，回退到规则/Q网络 |
| `powershell` 不可用 | 低 | 非 Windows 系统上 system_file_finder 失效 |
| `faiss-cpu` 未安装 | 低 | 回退到 numpy 纯检索 |
| `torch` 反序列化 | 中 | `weights_only=False` 存在安全风险 |
| `workspace/.operation_log.json` | 低 | 可能被检索命中为噪声，已内置过滤 |

---

## 九、总结

**运行时必需文件清单** (最小依赖集):

```
本地模块 (19个):
  agent_state.py, agent_rl.py, marl_agent.py, router.py, answer_generator.py,
  file_corpus.py, query_rewriter.py, llm_intent.py, file_operators.py,
  fact_query.py, deep_search.py, clarification.py, evidence_evaluator.py,
  system_file_finder.py, ocr_handler.py, safety.py, registry_db.py,
  parallel_scanner.py, faiss_store.py

配置文件 (3个):
  config/app.yaml, config/models.yaml, .env

模型文件 (2个):
  models/marl_agents.pth, models/agent_rl.pth

运行时数据:
  ~/.agentrl/ (SQLite + FAISS 索引)
  workspace/ (语料库目录)
  %TEMP%/agentrl_pf_cache.json (程序缓存)
```
