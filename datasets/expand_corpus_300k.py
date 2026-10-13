# -*- coding: utf-8 -*-
"""把小模型语料库扩充到 30 万条（确定性组合生成）。

输出: datasets/intent_corpus_bulk.csv (text,label) —— 机器生成的大批量语料，
      与手工维护的 intent_corpus.csv / datasets/intent_corpus.csv 分开存放。
      检索器 (intent_retriever) 与 CSV 训练 (train_strong) 自动合并读取。

规则:
  - 目标: 根 corpus + datasets 合并去重后恰为 300,000 条
  - 去重: 新文本必须不与任何现有文本（不论标签）重复，也不互相重复
  - 确定性: 固定种子，重复运行产出完全相同的文件（幂等）
  - 11 个意图全覆盖（含此前语料缺失的 file_analysis / file_scan /
    report_generate —— 这三类此前检索路完全无法投票）

用法:  python datasets/expand_corpus_300k.py
"""

import csv
import io
import os
import random
import sys
import zlib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from atomic_io import atomic_write_text  # noqa: E402

SEED = 20261006
TOTAL_TARGET = 300_000
OUT_PATH = os.path.join(BASE, "datasets", "intent_corpus_bulk.csv")
SRC_PATHS = [
    os.path.join(BASE, "intent_corpus.csv"),
    os.path.join(BASE, "datasets", "intent_corpus.csv"),
]

VALID_LABELS = {
    "add_file", "delete_file", "modify_file", "refactor", "query",
    "deep_search", "vague", "file_analysis", "file_scan", "report_generate",
    "filename_query",
}

# 固定配额（flex = query + report_generate 吸收余量，保证总数恰为 30 万）
QUOTA_FIXED = {
    "vague": 15000,
    "add_file": 30000,
    "delete_file": 28000,
    "modify_file": 28000,
    "refactor": 24000,
    "deep_search": 30000,
    "filename_query": 27000,
    "file_analysis": 27000,
    "file_scan": 27000,
}

# ============================================================
#  素材库（在 generate_corpus_v2 基础上扩容）
# ============================================================
NAMES_CN = [
    "笔记", "配置", "日志", "备份", "草稿", "计划", "会议", "总结", "报告",
    "数据", "清单", "账单", "教程", "文章", "说明书", "合同", "发票", "简历",
    "信件", "日记", "账本", "菜谱", "行程", "预算", "方案", "原型", "设计稿",
    "接口", "文档", "脚本", "模板", "样例", "归档", "索引", "目录", "白皮书",
    "题库", "笔记2", "草稿2", "周报", "月报", "年报", "日报", "需求", "排期",
    "复盘", "纪要", "通讯录", "备忘", "提醒",
]
NAMES_EN = [
    "alpha", "beta", "gamma", "demo", "draft", "final", "index", "main",
    "summary", "detail", "temp", "result", "input", "output", "sample",
    "template", "backup", "archive", "current", "latest", "old", "new",
    "session", "profile", "account", "payment", "order", "invoice", "ticket",
    "issue", "task", "sprint", "release", "patch", "hotfix", "config", "env",
    "secret", "token", "cert", "key", "seed", "init", "setup", "makefile",
    "runner", "worker", "queue", "cache", "state", "snapshot", "checkpoint",
    "trace", "metric", "alert", "event", "signal", "batch", "chunk", "shard",
    "node", "edge", "cluster", "worker2", "gateway", "proxy", "router",
    "switch", "bridge", "portal", "console", "dashboard", "widget", "plugin",
    "addon", "module", "layer", "pipe", "flow", "graph", "tree", "forest",
    "lake", "pond", "stream", "river", "cloud", "storm", "fog", "hail",
]
EXTS = [
    "txt", "md", "py", "json", "csv", "log", "yaml", "yml", "js", "ts",
    "html", "css", "sh", "sql", "xml", "ini", "conf", "cfg", "toml", "go",
    "java", "rs", "c", "h", "docx", "xlsx",
]
QUALS = [
    "202601", "202602", "202603", "202604", "202605", "202606",
    "202607", "202608", "202609", "202610", "202611", "202612",
    "v1", "v2", "v3", "v10", "v11", "v12",
    "old", "new", "final", "draft", "backup", "tmp", "prev", "next",
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
]

FILES = (
    [f"{n}.{e}" for n in NAMES_CN for e in EXTS]
    + [f"{n}.{e}" for n in NAMES_EN for e in EXTS]
    + [f"{n}-{q}.{e}" for n in NAMES_CN[:40] for q in QUALS for e in EXTS[:14]]
    + [f"{n}-{q}.{e}" for n in NAMES_EN[:40] for q in QUALS for e in EXTS[:14]]
)

DIRS = [
    "src", "lib", "docs", "tests", "config", "scripts", "assets", "images",
    "data", "logs", "tmp", "build", "dist", "vendor", "node_modules",
    "desktop", "download", "document", "picture", "backup", "archive",
    "project", "workspace", "backend", "frontend", "service", "deploy",
    "src/app", "src/core", "src/utils", "docs/api", "docs/guide",
    "data/raw", "data/processed", "logs/app", "logs/nginx", "etc/nginx",
    "etc/app", "var/log", "opt/app", "home/user", "home/user/docs",
]

MODULES = [
    "auth", "api", "db", "cache", "router", "models", "views", "utils",
    "forms", "tasks", "celery", "admin", "serializers", "middleware",
    "handlers", "validators", "services", "repos", "payment", "notify",
]

TOPICS = [
    "机器学习", "深度学习", "强化学习", "自然语言处理", "计算机视觉",
    "数据分析", "大数据", "云计算", "微服务", "容器化", "分布式系统",
    "网络安全", "区块链", "物联网", "边缘计算", "前端框架", "后端架构",
    "数据库优化", "性能调优", "代码审查", "项目管理", "敏捷开发",
    "持续集成", "自动化测试", "代码规范", "用户体验", "产品设计",
    "系统设计", "架构模式", "设计模式", "推荐系统", "知识图谱",
    "向量检索", "缓存策略", "消息队列", "服务网格", "灰度发布",
]

CONTENTS = [
    "会议纪要", "启动日志", "周报数据", "测试用例", "接口定义", "环境变量",
    "部署脚本", "数据库连接", "缓存配置", "日志轮转", "用户反馈", "性能指标",
    "错误堆栈", "依赖清单", "版本记录", "变更说明", "验收标准", "排期计划",
    "预算明细", "客户名单", "商品目录", "订单记录", "库存清单", "培训材料",
    "操作手册", "故障复盘", "安全审计", "容量规划", "压测结果", "迁移方案",
]

POLITE = [
    "请", "帮我", "麻烦", "能不能", "可以帮我", "我想", "帮忙", "劳驾",
    "辛苦", "有空的话", "麻烦你", "方便的话",
]
TIME_WORDS = ["现在", "马上", "尽快", "赶紧", "立刻", "抽空", "有空时", "待会"]
ACTIONS = [
    "搜索", "查找", "查询", "查看", "检索", "找到", "定位", "发现",
    "浏览", "翻阅", "扫一眼", "过一遍", "确认", "核实", "验证",
]
MEASURE = ["一下", "一遍", "一次", "看看", "瞅瞅", "瞧瞧"]

# ============================================================
#  各意图生成器（模板 × 参数，rng 定量抽样，全局去重）
# ============================================================

ADD_TPL = [
    "创建文件 {f}", "新建一个 {f}", "帮我写一个 {f}", "写入 {f}",
    "添加文件 {f} 内容为 {c}", "新建 {f} 保存 {c}", "生成 {f}",
    "帮我创建 {f} 然后写入内容", "创建一个叫 {f} 的文件",
    "把以下内容保存到 {f}: {c}", "写一个 {f} 文件", "帮我新建 {f}",
    "创建 {f} 放在当前目录", "添加 {f} 到项目中",
    "新建文件 {f} 里面写 {c}", "帮我生成一个 {f}", "创建 {f}，内容是 {c}",
    "写入新文件 {f}", "建一个 {f}", "帮我写 {f} 这个文件",
    "新建 {f} 并写入 {c}", "创建名为 {f} 的文件", "帮忙创建 {f}",
    "增加一个 {f}", "写个 {f} 吧", "帮我建 {f}", "新建一个叫 {f} 的文档",
    "创建 {f} 文件", "写一个 {f} 放进去", "帮我写个 {f} 出来",
    "添加 {f} 文件内容为 {c}", "生成一个新的 {f}", "创建空文件 {f}",
    "新建空白 {f}", "创建 {f} 存储 {c}", "帮我在 {d} 下创建 {f}",
    "新建 {f} 在 {d} 目录", "写一个脚本 {f}", "帮我写脚本 {f}",
    "创建 {f} 记录 {c}", "帮我写一个 {f} 文件内容是 {c}",
    "创建 {f} 然后写 {c}", "新建文件 {f} 用来存 {c}",
    "写一个 {f} 把 {c} 放进去", "创建一个 {f} 记录一下 {c}",
    "帮我创建 {m}.py 模块", "新建 {m}.js 文件", "写一个 {m} 的测试文件",
    "创建 {d}/{f}", "在 {d} 目录下新建 {f}", "帮我创建 {d} 里面的 {f}",
    "创建 {f} 放到 {d} 里", "新建 {f} 到 {d}", "帮我写一份{c}的文档存到 {f}",
    "写入{c}相关数据到 {f}", "创建 {f} 用来记录{c}信息",
    "帮我写一个关于{c}的 {f}", "写入{c}日志到 {f}", "create file {f}",
    "create a new file named {f}", "write content to {f}",
    "make a file {f} please", "new file {f} with content {c}",
    "add a file {f} to the project", "帮我 add 一个 {f}",
    "新建一个文件叫 {f}", "写一个 {f} 记录{c}", "生成文件 {f} 放 {d} 下",
]


DEL_TPL = [
    "删除文件 {f}", "移除 {f}", "把 {f} 删掉", "清理 {f}", "帮我删除 {f}",
    "删掉 {f}", "移除掉 {f}", "把 {f} 从项目里删去", "删除 {f} 这个文件",
    "清理掉 {f}", "把 {f} 给删了", "帮我把 {f} 删了", "删去 {f}", "去掉 {f}",
    "把那个 {f} 删掉", "帮我移除 {f}", "删除掉 {f}", "清理一下 {f}",
    "把 {f} 删除", "删掉这个 {f}", "移除 {f} 文件", "帮我把 {f} 移除",
    "把 {f} 清理掉", "删除 {f} 不要了", "清理掉旧的 {f}", "帮我把 {f} 去掉",
    "删除掉那个旧的 {f}", "把 {f} 从目录里删去", "不需要 {f} 了删掉",
    "帮我删掉 {f}", "删除备份 {f}", "移除临时文件 {f}", "清理日志 {f}",
    "把过期的 {f} 删了", "删除多余的 {f}", "清理无用的 {f}",
    "删掉旧版本的 {f}", "帮我删除那个 {f}", "移除 {f} 不用了",
    "把 {f} 拿掉", "删除 {f} 释放空间", "帮我把 {f} 清理了",
    "删掉没用的 {f}", "删除过时的 {f}", "移除废弃的 {f}",
    "清理不再需要的 {f}", "把 {f} 从仓库删去", "删除 {f} 保持整洁",
    "把 {d} 下的 {f} 删了", "删除 {d}/{f}", "移除 {d} 里面的 {f}",
    "清理 {d} 目录中的 {f}", "帮我把 {d}/{f} 删掉", "delete file {f}",
    "remove {f} please", "get rid of {f}", "delete {f} from {d}",
    "把 {f} 删了吧，{r}", "删除 {f}，{r}",
]


MOD_TPL = [
    "修改文件 {f} 内容为 {c}", "更新 {f} 改成 {c}", "编辑 {f}",
    "帮我改一下 {f}", "把 {f} 的内容替换一下", "更新 {f} 的内容",
    "修改 {f} 中的配置", "改一下 {f} 把内容换成 {c}", "编辑 {f} 改为 {c}",
    "帮我更新 {f}", "修改 {f} 改成 {c}", "把 {f} 更新为 {c}",
    "更新 {f} 里的 {c}", "改 {f} 中的参数", "修改 {f} 把 {c} 换掉",
    "编辑文件 {f} 加上 {c}", "帮我修改 {f}", "更新 {f} 的设置",
    "改一下 {f} 的内容", "修改 {f} 把配置改一下", "更新 {f} 修复bug",
    "编辑 {f} 增加新功能", "帮我改 {f} 里的配置", "修改 {f} 加上 {c}",
    "把 {f} 改成 {c}", "更新 {f} 里的内容为 {c}", "帮我编辑 {f}",
    "修改一下 {f}", "更新 {f} 的代码", "把 {f} 的内容修改为 {c}",
    "帮我把 {f} 改了", "编辑 {f} 修改为 {c}", "更新 {f} 的配置文件",
    "修改 {f} 里的 {c}", "帮我更新一下 {f}", "把 {f} 改一改",
    "修改 {f} 变成 {c}", "更新 {f} 加入 {c}", "编辑 {f} 替换内容",
    "帮我修改一下 {f} 的内容", "把 {f} 重新编辑一下", "更新 {f} 里的数据",
    "修改 {f} 的参数为 {c}", "帮我把 {f} 更新一下", "编辑 {f} 加入新内容",
    "修改 {f} 增加 {c}", "更新 {f} 的文档", "帮我把 {f} 改成 {c}",
    "把 {d} 下的 {f} 改一下", "修改 {d}/{f} 的内容",
    "更新 {d}/{f} 里的 {c}", "编辑 {d} 目录的 {f}", "改一下 {m}.py 的代码",
    "修改 {m} 模块的逻辑", "update {f} to {c}", "edit {f} please",
    "change the content of {f}", "modify {f} in {d}",
]


REF_TPL = [
    "整理文件夹", "重构项目结构", "把文件按类型分类归档", "帮我整理一下目录",
    "重命名文件", "移动文件到对应文件夹", "代码重构", "优化项目结构",
    "整理一下 {d} 目录", "把代码按模块分目录", "重构一下代码结构",
    "帮我整理文件夹", "把文件归类整理", "重新组织目录结构",
    "整理项目文件", "帮我把文件按类型分一下", "重构目录",
    "移动文件到正确的位置", "整理一下代码目录", "重命名一下文件",
    "优化一下目录结构", "帮我把文件整理一下", "按功能模块分类文件",
    "重构文件组织方式", "整理文档目录", "帮我归档旧文件",
    "重新整理项目结构", "把测试文件移到 tests 目录",
    "整理配置文件到 {d} 目录", "帮我把代码重构一下", "文件重新分类",
    "优化文件结构", "整理一下项目目录", "按类型把文件分好",
    "重构项目目录结构", "帮我把文件按日期归档", "整理日志文件",
    "移动文档到 {d} 目录", "重命名 {f} 为 {f2}", "帮我整理资源文件",
    "把图片移到 {d} 目录", "整理脚本到 {d} 目录", "按模块重组代码",
    "帮我归类一下文件", "重构整个项目", "把工具函数抽到 {d}",
    "整理依赖文件", "帮我把文件系统理一下", "优化目录布局",
    "重新组织文件结构", "整理一下整个项目", "把 {m} 相关的文件移到 {d}",
    "把 {f} 从 {d} 移到 {d2}", "把 {m}.py 和 {m2}.py 重命名",
    "把 {d} 里的文件按类型分到 {d2}", "整理 {d} 目录结构",
    "重构 {d} 下面的代码", "把 {d} 里的配置文件都移到 {d2}",
    "把 {f} 移到 {d} 目录下", "帮我对 {d} 做一次整理",
    "organize my files by type", "rename {f} to {f2}",
    "move {f} into {d}", "tidy up the {d} folder",
]


QRY_TPL = [
    "查询文件 {f} 里写了什么", "搜索 {f} 的内容", "{f} 存在吗",
    "帮我找一下 {f}", "查看 {f}", "{f} 里面有什么", "搜索 {c}",
    "查一下这个文件", "查看 {f} 的代码", "搜索关于{c}的资料",
    "查找 {f} 的位置", "查看 {f} 的内容", "搜索文件里的关键词",
    "帮我查一下 {f}", "{f} 文件在哪", "查看一下 {f}",
    "搜索包含 {c} 的文件", "查找相关文件", "这个项目有什么文件",
    "查看项目结构", "搜索 {c} 相关内容", "查一下 {f} 里写了啥",
    "帮我看看 {f}", "找一下 {c} 在哪里", "查看 {f} 的配置",
    "搜索代码中的函数", "查一下项目用了什么框架",
    "查看当前目录有哪些文件", "帮我查 {f}", "找找 {f} 这个文件",
    "搜索项目中的错误", "查看 {f} 的日志", "查一下系统状态",
    "查看文件列表", "搜索包含 {c} 的代码", "帮我找 {c}",
    "查看文档内容", "找一下配置文件在哪", "搜索 {f} 相关信息",
    "查一下这个目录", "帮我查看 {f} 的内容", "找找有没有 {f}",
    "查看 {f} 是否存在", "查一下 {c} 的定义", "看看 {f} 里有什么",
    "帮我搜索一下", "查找所有 Python 文件", "查看 README 内容",
    "搜索项目依赖", "在 {d} 里找 {f}", "查看 {d}/{f} 的内容",
    "搜索 {d} 目录下的 {c}", "找一下 {m}.py 在哪",
    "查看 {m} 模块的代码", "查一下 {d} 里面有什么文件",
    "找一下 {c} 相关的文件", "查看 {f} 和 {f2} 的差异",
    "查一下 {m} 的实现", "how do I find {f}", "where is {f}",
    "what is inside {f}", "search the web for {c}", "look up {c}",
    "check if {f} exists", "帮我查下 message window 是什么意思",
]


DEEP_TPL = [
    "对比分析 {f} 和 {f2}", "深度分析 {f} 的趋势与影响",
    "综合分析 {f} 和 {f2} 的差异", "帮我比较一下 {f} 和 {f2}",
    "分析 {f} 的优缺点", "深入研究 {f} 的发展方向",
    "评估 {f} 和 {f2} 的区别", "对比 {f} 和 {f2} 的性能",
    "深度分析项目的架构", "综合评估方案", "帮我分析一下 {c} 的影响",
    "比较 {c} 和 {c2} 的优劣", "深入分析 {c} 的问题", "评估 {c} 的可行性",
    "分析项目的技术栈", "对比不同方案的利弊", "深度分析代码质量",
    "综合分析系统性能", "帮我做一下 {c} 的分析", "深入研究 {c} 的实现",
    "分析 {c} 的发展趋势", "对比 {c} 和 {c2} 的效果",
    "深度分析用户行为", "综合分析数据趋势", "帮我分析 {c} 的问题",
    "深入分析错误原因", "评估项目的整体架构", "对比分析各个模块",
    "深度分析安全风险", "综合评估技术方案", "分析系统瓶颈在哪",
    "帮我分析一下 {c}", "深入分析 {c} 的瓶颈", "对比 {c} 的不同方案",
    "分析{c}的复杂度", "评估{c}方案的优缺点", "深度分析{c}逻辑",
    "综合分析{c}风险", "帮我做深度分析", "分析一下整体架构",
    "对比分析{c}和{c2}", "评估{c}的技术选型",
    "深入分析 {f} 中的 {c}", "评估 {f2} 对 {c} 的影响",
    "综合评估 {c} 在 {f} 中的应用", "深度对比 {c} 和 {c2} 的区别",
    "分析 {f} 和 {f2} 的架构差异", "帮我深入分析 {c} 的发展趋势",
    "评估 {c} 方案的可行性", "综合分析 {c} 的优缺点",
    "对比 {c} 和 {c2} 的实际效果", "深度分析 {c} 的应用场景",
    "compare {c} and {c2}", "analyze the pros and cons of {c}",
    "deep dive into {c}", "evaluate the tradeoffs of {c}",
]


FNQ_TPL = [
    "找一下最近修改的 {f}", "最近一天创建的文件有哪些",
    "最新的文件名有哪些", "最近操作过的文件",
    "帮我找最近的文件", "最近新添加了什么文件",
    "最近哪个文件被修改了", "查找最近一个月创建的 {f}",
    "最近下载的文件在哪里", "最近新建的文件有哪些",
    "找一下最近修改过的文件", "最新的文件是哪个",
    "哪些文件是今天创建的", "查找最近添加的文件",
    "最近使用的文档有哪些", "{t}创建的文件在哪",
    "{t}修改过的文件有哪些", "帮我列出{t}的文件",
    "找{t}下载的文件", "查一下{t}的 {f}",
    "看看{t}新建了什么文件", "有没有{t}的 {f}",
    "找找{t}删除前的备份", "最近一周的文件在哪",
    "今天新增了哪些文件", "昨天改过的文件有哪些",
    "这周创建的文档有哪些", "上个月的文件还能找到吗",
    "查找叫 {f} 的文件", "找一下名字带 {c} 的文件",
    "搜索文件名包含 {c} 的文件", "定位 {f} 在哪个目录",
    "find files created {t}", "find recently modified files",
    "locate {f}", "which files changed {t}",
]


FA_TPL = [
    "分析一下 {f} 的内容", "帮我分析 {f}", "看看 {f} 里有什么问题",
    "分析 {f} 的数据分布", "解读 {f}", "帮我看看 {f} 的结构",
    "分析 {f} 的关键指标", "评估 {f} 的质量", "解析 {f}",
    "分析日志 {f}", "帮我分析这份 {f}", "分析 {d} 下的 {f}",
    "深入看看 {f}", "分析 {f} 的访问趋势", "统计分析 {f}",
    "对 {f} 做个分析", "分析 {f} 中的异常", "拆解 {f} 的构成",
    "分析 {f} 和 {f2} 的数据", "帮我读一下 {f} 并分析",
    "analyze the {f} file", "analyze data in {f}",
    "give me an analysis of {f}", "看看这份{c}数据",
    "分析一下{c}的趋势", "帮我出一份{c}分析",
    "分析 {f} 里的{c}指标", "研究一下 {f} 的{c}表现",
    "分析{c}在 {f} 中的占比", "对比分析 {f} 的历史数据",
    "分析 {f} 的用户行为数据", "看看 {c} 的数据表现",
    "分析异常情况出现在哪里", "帮我分析失败原因",
    "分析 {f} 中的错误日志", "统计 {f} 里的错误类型",
]


SCAN_TPL = [
    "扫描 {d} 目录", "帮我扫描一下 {d}", "scan the {d} folder",
    "全盘扫描文件", "扫描磁盘找重复文件", "帮我扫一遍 {d}",
    "扫描 {d} 下的所有文件", "快速扫描一下系统文件",
    "扫描图片文件夹", "扫描文档目录", "扫描日志目录",
    "对 {d} 做一次文件扫描", "扫描一下有没有大文件",
    "扫描 {d} 里的可疑文件", "帮我扫描并列出文件",
    "扫描重复的 {f}", "扫描过期文件", "扫描临时文件",
    "帮我扫一下 {d} 里的 {f}", "扫描 {d} 找 {f}",
    "检查 {d} 里的文件", "扫一遍项目目录找孤儿文件",
    "扫描 {d} 统计文件数量", "扫描 {d} 里最大的文件",
    "帮我扫描电脑里的图片", "扫描整个目录树",
    "scan for duplicate files", "scan {d} and list everything",
    "扫一下 {d} 有没有多余的文件", "扫描 {d} 下面的脚本文件",
    "扫一遍 {d} 看看结构", "扫描 {d} 里的配置文件",
    "帮我做一次文件体检", "扫描 {d} 找出旧文件",
]


RPT_TPL = [
    "生成{c}报告", "帮我生成一份{c}周报", "生成 {f} 报告文件",
    "出一份{c}报表", "生成 {c} 的总结报告", "帮我生成日报",
    "生成 {t}的{c}报告", "汇总数据生成报告", "生成 {f} 作为报告",
    "生成一份 {c} 分析报告", "帮我出个{c}报告 {f}",
    "把 {c} 数据整理成报告", "生成 {c} 月报 {f}",
    "生成{c}报告放到 {d}", "给我生成一份{c}的文档报告",
    "自动生成 {f} 报告", "生成 {t} {c} 汇总报表",
    "生成项目 {c} 报告", "生成 {c} 对比报告 {f}",
    "生成 {c} 报告并发我", "帮我把 {c} 写成报告 {f}",
    "生成 {c} 周报 {f} 放 {d} 下", "出报表 {f}",
    "汇总生成 {c} 报表", "生成 {t}的 {f}",
    "generate a {c} report", "generate report {f}",
    "make a weekly report on {c}", "produce the {c} summary report",
    "帮我生成本周的 {c} 报告", "生成 {c} 报告 {f} 要包含图表",
    "整理{c}数据出一份报告", "生成 {c} 月度报表 {f}",
    "把 {c} 情况生成报告 {f}", "生成 {c} 报告，{t}的数据",
]


GREET = [
    "你好", "您好", "嗨", "哈喽", "嘿", "喂", "哟", "hello", "hi", "hey",
    "早上好", "上午好", "中午好", "下午好", "晚上好", "早安", "午安",
    "晚安", "在吗", "在么", "有人吗", "在不在", "哈喽呀", "嗨你好",
    "good morning", "good night", "good afternoon", "yo", "howdy",
]
BODY = [
    "你是谁", "你能做什么", "介绍一下自己", "你是AI吗", "你叫什么名字",
    "你多大了", "今天天气怎么样", "明天会下雨吗", "今天星期几",
    "讲个笑话", "给我讲个故事", "推荐一部电影", "推荐几本书",
    "怎么减肥", "学英语怎么学", "今天吃什么", "你喜欢什么",
    "你有什么爱好", "你会唱歌吗", "你会画画吗", "你会写诗吗",
    "世界有多大", "宇宙有外星人吗", "生命的意义是什么",
    "帮我一下", "请问", "我有个问题", "你能帮我吗", "需要帮助",
    "这个怎么做", "帮帮忙", "求助", "请教一下", "问一下",
    "嗯", "好的", "谢谢", "哈哈", "哦", "啊", "嗯嗯", "好吧", "行",
    "ok", "OK", "收到", "了解", "明白了", "知道了", "可以", "没问题",
    "牛", "厉害", "666", "好的好的", "emmm", "呃", "额", "啊这",
    "哈哈哈", "嘻嘻", "嘿嘿", "再见", "拜拜", "bye", "回头见",
    "随便打的", "你能识别中文吗", "这是一条测试消息", "随便聊聊",
    "无聊", "今天心情不错", "心情不好", "好累啊", "今天好开心",
    "明天要加班", "好困", "你觉得呢", "你说得对", "有道理",
    "我同意", "不太确定", "再说一遍", "随便发的", "test",
    "thanks a lot", "thank you so much", "see you later", "how are you",
    "what's up", "long time no see", "miss me", "tell me a joke",
]
SUF = ["", "呀", "啊", "哈", "呢", "吗", "！", "～", "～～", "哈喽",
       "，在吗", "，有人吗", "，帮我个忙", "，听得到吗", "～你好"]
FILLER_PREFIX = ["", "亲爱的", "朋友", "老铁", "同学", "老师", "小助手",
                 "在吗 ", "麻烦", "那个 "]


# ============================================================
#  单一数据驱动生成器：全部意图经 SPECS 由 gen_text 产出。
#  不拆 per-label 函数——结构雷同的克隆函数会被重复门禁拦截。
# ============================================================
DEL_REASONS = [
    "过期了", "没用了", "不需要了", "旧版本", "多余的",
    "废弃的", "临时文件", "缓存文件", "测试文件",
]
FNQ_TIMES = [
    "今天", "昨天", "本周", "上周", "本月", "上个月",
    "最近三天", "最近一周", "最近两天", "刚才", "上午", "晚上",
]
RPT_TIMES = [
    "本周", "上周", "本月", "上个月", "今天", "昨天",
    "第一季度", "第二季度", "全年", "最近一周",
]

SPECS = {
    "add_file": dict(tpl=ADD_TPL, pol=0.25,
                     params=dict(f=FILES, c=CONTENTS, d=DIRS, m=MODULES),
                     suffix=TIME_WORDS, suffix_p=0.12),
    "delete_file": dict(tpl=DEL_TPL, pol=0.25,
                        params=dict(f=FILES, d=DIRS, r=DEL_REASONS)),
    "modify_file": dict(tpl=MOD_TPL, pol=0.25,
                        params=dict(f=FILES, c=CONTENTS, d=DIRS,
                                    m=MODULES)),
    "refactor": dict(tpl=REF_TPL, pol=0.25,
                     params=dict(d=DIRS, d2=DIRS, f=FILES, f2=FILES,
                                 m=MODULES, m2=MODULES),
                     distinct=dict(d2="d")),
    "query": dict(tpl=QRY_TPL, pol=0.25,
                  params=dict(f=FILES, f2=FILES, c=TOPICS, d=DIRS,
                              m=MODULES),
                  suffix=MEASURE, suffix_p=0.1),
    "deep_search": dict(tpl=DEEP_TPL, pol=0.2,
                        params=dict(f=FILES, f2=FILES, c=TOPICS,
                                    c2=TOPICS),
                        distinct=dict(c2="c")),
    "filename_query": dict(tpl=FNQ_TPL, pol=0.25,
                           params=dict(f=FILES, c=TOPICS, t=FNQ_TIMES)),
    "file_analysis": dict(tpl=FA_TPL, pol=0.3,
                          params=dict(f=FILES, f2=FILES, d=DIRS,
                                      c=TOPICS)),
    "file_scan": dict(tpl=SCAN_TPL, pol=0.3,
                      params=dict(d=DIRS, f=FILES)),
    "report_generate": dict(tpl=RPT_TPL, pol=0.3,
                            params=dict(c=TOPICS, f=FILES, d=DIRS,
                                        t=RPT_TIMES)),
    "vague": None,
}


def gen_text(rng, label):
    """按标签确定性生成一条新语料（唯一的生成入口）。"""
    if label == "vague":
        text = FILLER_PREFIX[rng.randrange(len(FILLER_PREFIX))]
        text += GREET[rng.randrange(len(GREET))]
        text += SUF[rng.randrange(len(SUF))]
        if rng.random() < 0.55:
            text += BODY[rng.randrange(len(BODY))]
        return text
    spec = SPECS[label]
    chosen = {}
    for key, pool in spec["params"].items():
        val = pool[rng.randrange(len(pool))]
        avoid = spec.get("distinct", {}).get(key)
        if avoid is not None and avoid in chosen and val == chosen[avoid]:
            alts = [x for x in pool if x != chosen[avoid]]
            if alts:
                val = alts[rng.randrange(len(alts))]
        chosen[key] = val
    text = spec["tpl"][rng.randrange(len(spec["tpl"]))].format(**chosen)
    if rng.random() < spec["pol"]:
        text = POLITE[rng.randrange(len(POLITE))] + text
    if spec.get("suffix_p") and rng.random() < spec["suffix_p"]:
        pool = spec["suffix"]
        text += pool[rng.randrange(len(pool))]
    return text


# ============================================================
#  主流程
# ============================================================
def load_existing_pairs():
    """按 intent_retriever._load_corpus 同样的规则读取现有 (text,label) 集。"""
    pairs = set()
    texts = set()
    for path in SRC_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                text = (row.get("text") or "").strip()
                label = (row.get("label") or "").strip()
                if not text or label not in VALID_LABELS:
                    continue
                pairs.add((text, label))
                texts.add(text)
    return pairs, texts


def quotas(needed: int) -> dict:
    fixed_sum = sum(QUOTA_FIXED.values())
    if needed < 0:
        raise SystemExit(f"现有语料已超过 {TOTAL_TARGET}: needed={needed}")
    flex = needed - fixed_sum
    if flex < 0:
        raise SystemExit(f"配额超出目标: fixed={fixed_sum} needed={needed}")
    q = dict(QUOTA_FIXED)
    q["query"] = (flex + 1) // 2
    q["report_generate"] = flex // 2
    assert sum(q.values()) == needed, (sum(q.values()), needed)
    return q


def main():
    existing_pairs, existing_texts = load_existing_pairs()
    needed = TOTAL_TARGET - len(existing_pairs)
    plan = quotas(needed)
    print(f"现有 {len(existing_pairs)} 条（text 去重 {len(existing_texts)}），"
          f"需新增 {needed} 条")

    seen = set(existing_texts)   # 新文本必须全局不重复（不只同标签）
    rows = []
    for label in sorted(plan, key=lambda k: -plan[k]):
        quota = plan[label]
        if quota <= 0:
            continue
        rng = random.Random(zlib.crc32(f"{SEED}:{label}".encode()))
        out, guard = [], 0
        while len(out) < quota:
            guard += 1
            if guard > quota * 200:
                raise SystemExit(f"[{label}] 候选空间不足: "
                                 f"{len(out)}/{quota} (guard={guard})")
            text = gen_text(rng, label).strip()
            if not text or len(text) > 300 or text in seen:
                continue
            seen.add(text)
            out.append(text)
        rows.extend((t, label) for t in out)
        print(f"  {label:<16} +{len(out):>6}  (累计 {len(rows)})")

    assert len(rows) == needed, (len(rows), needed)
    assert len(existing_pairs) + len(rows) == TOTAL_TARGET

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["text", "label"])
    writer.writerows(rows)
    atomic_write_text(OUT_PATH, buf.getvalue(), encoding="utf-8")
    print(f"写出 {len(rows)} 条 → {OUT_PATH}")
    print(f"合并去重后总数 = {TOTAL_TARGET}")


if __name__ == "__main__":
    main()
