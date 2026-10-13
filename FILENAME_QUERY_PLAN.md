# FILENAME_QUERY 意图添加计划

## 背景

用户在使用系统时输入"为我找到最近添加的文件名"，系统将其错误分类为 `query` 意图下的 `query_system` 子意图，导致搜索 Windows 系统程序而非文件系统中的文件。

## 需求

添加一个新的主意图 `FILENAME_QUERY`，专门处理文件名查询类请求，如：
- "为我找到最近添加的文件名"
- "最近修改的文件有哪些"
- "查找最近创建的文件"
- "列出最近使用的文件"

## 修改文件清单

### 1. agent_state.py
- 在 `IntentType` 枚举中添加 `FILENAME_QUERY = "filename_query"`
- 在 `RoutingPath` 枚举中添加 `FILENAME_QUERY = "filename_query"`
- 在 `SHORT_LABELS` 字典中添加 `IntentType.FILENAME_QUERY: "文件名查询"`

### 2. llm_intent.py
- 在 `VALID_INTENTS` 列表中添加 `"filename_query"`
- 在 `INTENT_GUIDE` 列表中添加描述

### 3. query_rewriter.py
- 在 `OP_KEYWORDS` 字典中添加 `"filename_query"` 键和相关关键词

### 4. agent_rl.py
- 在 `INTENT_ORDER` 列表中添加 `IntentType.FILENAME_QUERY`
- 在 `TASK_TEMPLATES` 字典中添加模板
- 在 `EVAL_REAL_SAMPLES` 字典中添加样本
- 在 `route()` 方法中添加映射

### 5. marl_agent.py
- 在 `IntentAgent.INTENT_LIST` 列表中添加 `IntentType.FILENAME_QUERY`
- 在 `IntentAgent.rule_keywords` 字典中添加关键词

### 6. router.py
- 在 `route()` 方法中添加 `FILENAME_QUERY` 意图的处理分支
- 实现 `_route_filename_query()` 方法

### 7. intent_corpus.csv
- 添加文件名查询的训练样本

## 实现细节

### FILENAME_QUERY 意图定义
- 专门处理文件名查询请求
- 支持时间范围查询（最近添加、最近修改等）
- 支持模糊文件名搜索

### 关键词
- "文件名", "最近添加", "最近修改", "最近创建", "最近使用"
- "最近保存", "最近编辑", "最近打开", "最近访问"
- "文件列表", "文件查询", "查找文件"

### 路由处理
- 使用文件系统扫描器查找文件
- 支持按时间排序
- 返回文件名列表