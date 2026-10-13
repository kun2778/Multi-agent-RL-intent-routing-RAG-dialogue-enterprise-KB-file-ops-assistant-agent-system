# FILENAME_QUERY 意图添加日志

## 背景
用户在使用系统时输入"为我找到最近添加的文件名"，系统将其错误分类为 `query` 意图下的 `query_system` 子意图，导致搜索 Windows 系统程序而非文件系统中的文件。

## 解决方案
添加了一个新的主意图 `FILENAME_QUERY`，专门处理文件名查询类请求。

## 修改文件清单

### 1. agent_state.py
- 添加 `IntentType.FILENAME_QUERY = "filename_query"`
- 添加 `RoutingPath.FILENAME_QUERY = "filename_query"`
- 更新 `SHORT_LABELS` 字典

### 2. llm_intent.py
- 在 `VALID_INTENTS` 列表中添加 `"filename_query"`
- 在 `INTENT_GUIDE` 列表中添加描述
- 更新 `SYSTEM_PROMPT` 中的意图数量（10类 -> 11类）

### 3. query_rewriter.py
- 在 `OP_KEYWORDS` 字典中添加 `"filename_query"` 键和相关关键词

### 4. agent_rl.py
- 在 `INTENT_ORDER` 列表中添加 `IntentType.FILENAME_QUERY`
- 在 `TASK_TEMPLATES` 字典中添加模板
- 在 `EVAL_REAL_SAMPLES` 字典中添加样本
- 在 `route()` 方法中添加映射
- 在 `_semantic_conflict_guard()` 方法中添加filename_query检测
- 在 `_safe_fallback()` 方法中添加filename_query检测

### 5. marl_agent.py
- 在 `IntentAgent.INTENT_LIST` 列表中添加 `IntentType.FILENAME_QUERY`
- 在 `IntentAgent.rule_keywords` 字典中添加关键词
- 在 `_semantic_conflict_guard()` 方法中添加filename_query检测

### 6. router.py
- 在 `route()` 方法中添加 `FILENAME_QUERY` 意图的处理分支
- 实现 `_route_filename_query()` 方法
- 实现 `_extract_time_range()` 方法
- 实现 `_extract_file_extension()` 方法
- 实现 `_execute_filename_query()` 方法
- 添加 `Optional` 到 typing 导入

### 7. intent_corpus.csv
- 添加29个文件名查询的训练样本

### 8. datasets/intent_corpus.csv
- 添加21个文件名查询的训练样本

## 功能特性

### FILENAME_QUERY 意图
专门处理文件名查询请求，如：
- "为我找到最近添加的文件名"
- "最近修改的文件有哪些"
- "查找最近创建的文件"
- "最近编辑过的文件名"

### 支持的时间范围
- 今天
- 昨天
- 本周
- 本月
- 最近N分钟/小时/天/周/月

### 支持的文件类型筛选
- 可以按扩展名筛选（如 .txt, .py 等）

### 搜索结果
- 显示文件名和修改时间
- 按修改时间降序排列
- 最多显示30个文件

## 测试验证
- 所有修改过的文件都通过了Python语法检查
- 训练数据已正确添加到两个intent_corpus.csv文件中
- 新增意图的标签统计正确显示