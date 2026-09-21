"""生成项目专用中文意图分类 CSV 数据集 v2（7000+ 条）。

组合生成：句式模板 × 修饰词 × 文件名 × 内容片段 × 增强变换。
输出: intent_corpus.csv (text,label) → 项目根目录。
"""

import csv
import os
import random

random.seed(42)

# ============================================================
#  素材库
# ============================================================
FILES = [
    "notes.txt", "config.yaml", "main.py", "app.json", "report.md",
    "readme.md", "data.csv", "settings.py", "index.html", "style.css",
    "utils.js", "test.py", "requirements.txt", "Dockerfile", "package.json",
    "train.py", "model.py", "utils.go", "server.rs", "Makefile",
    "database.sql", "schema.prisma", "logo.png", "backup.zip", "log.txt",
    "app.js", "router.py", "auth.py", "api.py", "test_utils.py",
    ".env", "docker-compose.yaml", "nginx.conf", "deploy.sh", "build.gradle",
    "pom.xml", "Cargo.toml", "go.mod", "tsconfig.json", "webpack.config.js",
    "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", ".gitignore", "Procfile",
]

FILES2 = [
    "notes.md", "config.json", "index.js", "app.ts", "report.txt",
    "data.json", "server.py", "client.py", "db.py", "cache.py",
    "auth.js", "routes.py", "models.py", "views.py", "forms.py",
    "admin.py", "tasks.py", "celery.py", "wsgi.py", "asgi.py",
    "test_auth.py", "test_api.py", "test_model.py", "conftest.py", "setup.py",
    "pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "mypy.ini",
    "README_zh.md", "docs/guide.md", "docs/api.md", "examples/demo.py",
]

DIRS = ["src", "lib", "docs", "tests", "config", "scripts", "assets",
        "images", "data", "logs", "tmp", "build", "dist", "vendor", "node_modules"]

MODULES = ["auth", "api", "db", "cache", "router", "models", "views",
           "utils", "forms", "tasks", "celery", "admin", "serializers",
           "middleware", "handlers", "validators", "services", "repos"]

TOPICS = [
    "机器学习", "深度学习", "强化学习", "自然语言处理", "计算机视觉",
    "数据分析", "大数据", "云计算", "微服务", "容器化",
    "分布式系统", "网络安全", "区块链", "物联网", "边缘计算",
    "前端框架", "后端架构", "数据库优化", "性能调优", "代码审查",
    "项目管理", "敏捷开发", "持续集成", "自动化测试", "代码规范",
    "用户体验", "产品设计", "系统设计", "架构模式", "设计模式",
]

ACTIONS = [
    "搜索", "查找", "查询", "查看", "检索", "找到", "定位", "发现",
    "浏览", "翻阅", "扫一眼", "过一遍", "确认", "核实", "验证",
]

MEASURE_WORDS = ["一下", "一遍", "一次", "看看", "瞅瞅", "瞧瞧"]

POLITE = ["请", "帮我", "麻烦", "能不能", "可以帮我", "我想", "帮忙",
          "劳驾", "辛苦", "有空的话"]

CONJUNCTIONS = ["然后", "接着", "之后", "再", "顺便", "另外", "同时"]

TIME_WORDS = ["现在", "马上", "尽快", "赶紧", "立刻", "抽空", "有空时"]

OPINION_WORDS = ["怎么样", "好不好", "行不行", "对不对", "合不合理",
                  "有没有问题", "能不能用", "可不可以"]

# ============================================================
#  生成函数
# ============================================================

def _pick(lst): return random.choice(lst)
def _picks(lst, n): return random.sample(lst, min(n, len(lst)))
def _maybe(lst, p=0.5): return random.choice(lst) if random.random() < p else ""


def gen_add_file():
    """生成 add_file 意图的多样化句子。~100 种句式。"""
    f, g = _pick(FILES), _pick(FILES2)
    c = _pick(TOPICS)
    d = _pick(DIRS)
    m = _pick(MODULES)

    base_templates = [
        f"创建文件 {f}", f"新建一个 {f}", f"帮我写一个 {f}", f"写入 {f}",
        f"添加文件 {f} 内容为 {c}", f"新建 {f} 保存 {c}", f"生成 {f}",
        f"帮我创建 {f} 然后写入内容", f"创建一个叫 {f} 的文件",
        f"把以下内容保存到 {f}: {c}", f"写一个 {f} 文件",
        f"帮我新建 {f}", f"创建 {f} 放在当前目录", f"添加 {f} 到项目中",
        f"新建文件 {f} 里面写 {c}", f"帮我生成一个 {f}",
        f"创建 {f}，内容是 {c}", f"写入新文件 {f}", f"建一个 {f}",
        f"帮我写 {f} 这个文件", f"新建 {f} 并写入 {c}",
        f"创建名为 {f} 的文件", f"帮忙创建 {f}", f"增加一个 {f}",
        f"写个 {f} 吧", f"帮我建 {f}", f"新建一个叫 {f} 的文档",
        f"创建 {f} 文件", f"写一个 {f} 放进去", f"帮我写个 {f} 出来",
        f"添加 {f} 文件内容为 {c}", f"生成一个新的 {f}",
        f"创建空文件 {f}", f"新建空白 {f}", f"创建 {f} 存储 {c}",
        f"帮我在 {d} 下创建 {f}", f"新建 {f} 在 {d} 目录",
        f"创建配置文件 {f}", f"写一个脚本 {f}", f"帮我写脚本 {f}",
        f"生成报告文件 {f}", f"创建 {f} 记录 {c}", f"新建 {f} 保存日志",
        f"帮我写一个 {f} 文件内容是 {c}", f"创建 {f} 然后写 {c}",
        f"新建文件 {f} 用来存 {c}", f"写一个 {f} 把 {c} 放进去",
        f"创建一个 {f} 记录一下 {c}", f"帮我创建 {m}.py 模块",
        f"新建 {m}.js 文件", f"创建 {m} 模块的代码文件",
        f"写一个 {m} 的测试文件", f"创建 {d}/{f}",
        f"在 {d} 目录下新建 {f}", f"帮我创建 {d} 里面的 {f}",
        f"创建 {f} 放到 {d} 里", f"新建 {f} 到 {d}",
        f"帮我写一份{c}的文档存到 {f}", f"写入{c}相关数据到 {f}",
        f"创建 {f} 用来记录{c}信息", f"新建 {f} 存放{c}配置",
        f"帮我写一个关于{c}的 {f}", f"生成{c}分析报告 {f}",
        f"写入{c}日志到 {f}", f"创建{c}记录文件 {f}",
    ]

    sentence = _pick(base_templates)

    if random.random() < 0.3:
        prefix = _pick(POLITE)
        sentence = prefix + sentence
    if random.random() < 0.2:
        sentence += "，" + _pick(TIME_WORDS)
    if random.random() < 0.15:
        sentence += "，" + _pick(DIRS) + "下面"

    return sentence


def gen_delete_file():
    f = _pick(FILES)
    g = _pick(FILES2)
    d = _pick(DIRS)
    reasons = ["过期了", "没用了", "不需要了", "旧版本", "多余的",
               "废弃的", "临时文件", "备份文件", "缓存文件", "测试文件"]

    base_templates = [
        f"删除文件 {f}", f"移除 {f}", f"把 {f} 删掉", f"清理 {f}",
        f"帮我删除 {f}", f"删掉 {f}", f"移除掉 {f}", f"把 {f} 从项目里删去",
        f"删除 {f} 这个文件", f"清理掉 {f}", f"把 {f} 给删了",
        f"帮我把 {f} 删了", f"删去 {f}", f"去掉 {f}",
        f"把那个 {f} 删掉", f"帮我移除 {f}", f"删除掉 {f}",
        f"清理一下 {f}", f"把 {f} 删除", f"删掉这个 {f}",
        f"移除 {f} 文件", f"帮我把 {f} 移除", f"把 {f} 清理掉",
        f"删除 {f} 不要了", f"删掉这个 {f}", f"清理掉旧的 {f}",
        f"帮我把 {f} 去掉", f"删除掉那个旧的 {f}", f"把 {f} 从目录里删去",
        f"不需要 {f} 了删掉", f"把 {f} 给移除掉", f"帮我删掉 {f}",
        f"删除备份 {f}", f"移除临时文件 {f}", f"清理日志 {f}",
        f"把过期的 {f} 删了", f"删除多余的 {f}", f"清理无用的 {f}",
        f"删掉旧版本的 {f}", f"帮我删除那个 {f}", f"移除 {f} 不用了",
        f"把 {f} 拿掉", f"删除 {f} 释放空间", f"帮我把 {f} 清理了",
        f"删掉没用的 {f}", f"删除过时的 {f}", f"移除废弃的 {f}",
        f"清理不再需要的 {f}", f"把 {f} 从仓库删去", f"删除 {f} 保持整洁",
        f"把 {d} 下的 {f} 删了", f"删除 {d}/{f}", f"移除 {d} 里面的 {f}",
        f"清理 {d} 目录中的 {f}", f"帮我把 {d}/{f} 删掉",
        f"把 {f} 删了吧{_pick(reasons)}", f"删除 {f}，{_pick(reasons)}",
        f"{_pick(reasons)}，把 {f} 删了", f"{f} {_pick(reasons)} 帮我删掉",
    ]

    sentence = _pick(base_templates)
    if random.random() < 0.3:
        sentence = _pick(POLITE) + sentence
    return sentence


def gen_modify_file():
    f = _pick(FILES)
    c = _pick(TOPICS)
    d = _pick(DIRS)
    m = _pick(MODULES)
    changes = ["配置", "参数", "代码", "内容", "设置", "逻辑", "格式", "结构"]

    base_templates = [
        f"修改文件 {f} 内容为 {c}", f"更新 {f} 改成 {c}",
        f"编辑 {f}", f"帮我改一下 {f}", f"把 {f} 的内容替换一下",
        f"更新 {f} 的内容", f"修改 {f} 中的配置",
        f"改一下 {f} 把内容换成 {c}", f"编辑 {f} 改为 {c}",
        f"帮我更新 {f}", f"修改 {f} 改成 {c}",
        f"把 {f} 更新为 {c}", f"更新 {f} 里的 {c}",
        f"改 {f} 中的参数", f"修改 {f} 把 {c} 换掉",
        f"编辑文件 {f} 加上 {c}", f"帮我修改 {f}",
        f"更新 {f} 的设置", f"改一下 {f} 的内容",
        f"修改 {f} 把{_pick(changes)}改一下", f"更新 {f} 修复bug",
        f"编辑 {f} 增加新功能", f"帮我改 {f} 里的配置",
        f"修改 {f} 加上 {c}", f"把 {f} 改成 {c}",
        f"更新 {f} 里的内容为 {c}", f"帮我编辑 {f}",
        f"修改一下 {f}", f"更新 {f} 的代码",
        f"把 {f} 的内容修改为 {c}", f"帮我把 {f} 改了",
        f"编辑 {f} 修改为 {c}", f"更新 {f} 的配置文件",
        f"修改 {f} 里的 {c}", f"帮我更新一下 {f}",
        f"把 {f} 改一改", f"修改 {f} 变成 {c}",
        f"更新 {f} 加入 {c}", f"编辑 {f} 替换内容",
        f"帮我修改一下 {f} 的内容", f"把 {f} 重新编辑一下",
        f"更新 {f} 里的数据", f"修改 {f} 的参数为 {c}",
        f"帮我把 {f} 更新一下", f"编辑 {f} 加入新内容",
        f"修改 {f} 增加 {c}", f"更新 {f} 的文档",
        f"帮我把 {f} 改成 {c}", f"修改 {f} 的{_pick(changes)}",
        f"把 {d} 下的 {f} 改一下", f"修改 {d}/{f} 的内容",
        f"更新 {d}/{f} 里的 {c}", f"编辑 {d} 目录的 {f}",
        f"改一下 {m}.py 的代码", f"修改 {m} 模块的逻辑",
        f"更新 {m} 的配置", f"编辑 {m}.js 的内容",
    ]

    sentence = _pick(base_templates)
    if random.random() < 0.3:
        sentence = _pick(POLITE) + sentence
    return sentence


def gen_refactor():
    d = _pick(DIRS)
    d2 = _pick([x for x in DIRS if x != d])
    m = _pick(MODULES)
    m2 = _pick([x for x in MODULES if x != m])
    f = _pick(FILES)
    f2 = _pick(FILES2)

    base_templates = [
        "整理文件夹", "重构项目结构", "把文件按类型分类归档",
        "帮我整理一下目录", "重命名文件", "移动文件到对应文件夹",
        "代码重构", "优化项目结构", f"整理一下 {d} 目录",
        "把代码按模块分目录", "重构一下代码结构",
        "帮我整理文件夹", "把文件归类整理",
        "重新组织目录结构", "整理项目文件",
        "帮我把文件按类型分一下", "重构目录",
        "移动文件到正确的位置", "整理一下代码目录",
        "重命名一下文件", "优化一下目录结构",
        "帮我把文件整理一下", "按功能模块分类文件",
        "重构文件组织方式", "整理文档目录",
        "帮我归档旧文件", "重新整理项目结构",
        "把测试文件移到 tests 目录", f"整理配置文件到 {d} 目录",
        "帮我把代码重构一下", "文件重新分类",
        "优化文件结构", "整理一下项目目录",
        "按类型把文件分好", "重构项目目录结构",
        "帮我把文件按日期归档", "整理日志文件",
        f"移动文档到 {d} 目录", f"重命名 {f} 为 {f2}",
        "帮我整理资源文件", f"把图片移到 {d} 目录",
        f"整理脚本到 {d} 目录", "按模块重组代码",
        "帮我归类一下文件", "重构整个项目",
        f"把工具函数抽到 {d}", "整理依赖文件",
        "帮我把文件系统理一下", "优化目录布局",
        "重新组织文件结构", "整理一下整个项目",
        f"把 {m} 相关的文件移到 {d}", f"把 {f} 从 {d} 移到 {d2}",
        f"把 {m}.py 和 {m2}.py 重命名", f"把 {d} 里的文件按类型分到 {d2}",
        f"整理 {d} 目录结构", f"重构 {d} 下面的代码",
        f"把 {d} 里的配置文件都移到 {d2}", f"代码模块化 重组 {d}",
        f"把 {f} 移到 {d} 目录下", f"帮我对 {d} 做一次整理",
        f"将 {d} 和 {d2} 的文件合并整理", f"把散落的文件归到对应目录",
    ]

    sentence = _pick(base_templates)
    if random.random() < 0.3:
        sentence = _pick(POLITE) + sentence
    return sentence


def gen_query():
    f = _pick(FILES)
    f2 = _pick(FILES2)
    c = _pick(TOPICS)
    d = _pick(DIRS)
    m = _pick(MODULES)

    base_templates = [
        f"查询文件 {f} 里写了什么", f"搜索 {f} 的内容", f"{f} 存在吗",
        f"帮我找一下 {f}", f"查看 {f}", f"{f} 里面有什么",
        "搜索强化学习", "查一下这个文件", "找到红警3的启动程序在哪里",
        f"查看 {f} 的代码", "搜索关于机器学习的资料",
        f"查找 {f} 的位置", f"查看 {f} 的内容",
        "搜索文件里的关键词", f"帮我查一下 {f}",
        f"{f} 文件在哪", f"查看一下 {f}",
        f"搜索包含 {c} 的文件", "查找相关文件",
        "这个项目有什么文件", "查看项目结构",
        f"搜索 {c} 相关内容", f"查一下 {f} 里写了啥",
        f"帮我看看 {f}", f"找一下 {c} 在哪里",
        f"查看 {f} 的配置", "搜索代码中的函数",
        "查一下项目用了什么框架", "查看当前目录有哪些文件",
        f"帮我查 {f}", f"找找 {f} 这个文件",
        "搜索项目中的错误", f"查看 {f} 的日志",
        "查一下系统状态", "查看文件列表",
        f"搜索包含 {c} 的代码", f"帮我找 {c}",
        "查看文档内容", "找一下配置文件在哪",
        f"搜索 {f} 相关信息", "查一下这个目录",
        f"帮我查看 {f} 的内容", f"找找有没有 {f}",
        f"查看 {f} 是否存在", "搜索最新的提交记录",
        f"查一下 {c} 的定义", f"看看 {f} 里有什么",
        "帮我搜索一下", "查找所有 Python 文件",
        "查看 README 内容", "搜索项目依赖",
        f"在 {d} 里找 {f}", f"查看 {d}/{f} 的内容",
        f"搜索 {d} 目录下的 {c}", f"找一下 {m}.py 在哪",
        f"查看 {m} 模块的代码", f"查一下 {d} 里面有什么文件",
        f"找一下 {c} 相关的文件", f"搜索 {d} 下包含 {c} 的文件",
        f"查看 {f} 和 {f2} 的差异", f"查一下 {m} 的实现",
    ]

    sentence = _pick(base_templates)
    if random.random() < 0.3:
        sentence = _pick(POLITE) + sentence
    return sentence


def gen_deep_search():
    f = _pick(FILES)
    f2 = _pick(FILES2)
    c = _pick(TOPICS)
    c2 = _pick([x for x in TOPICS if x != c])

    base_templates = [
        f"对比分析 {f} 和 {f2}", f"深度分析 {f} 的趋势与影响",
        f"综合分析 {f} 和 {f2} 的差异", f"帮我比较一下 {f} 和 {f2}",
        f"分析 {f} 的优缺点", f"深入研究 {f} 的发展方向",
        f"评估 {f} 和 {f2} 的区别", f"对比 {f} 和 {f2} 的性能",
        "深度分析项目的架构", "综合评估方案",
        f"帮我分析一下 {c} 的影响", f"比较 {c} 和 {c2} 的优劣",
        f"深入分析 {c} 的问题", f"评估 {c} 的可行性",
        "分析项目的技术栈", "对比不同方案的利弊",
        "深度分析代码质量", "综合分析系统性能",
        f"帮我做一下 {c} 的分析", f"深入研究 {c} 的实现",
        f"分析 {c} 的发展趋势", f"对比 {c} 和 {c2} 的效果",
        "深度分析用户行为", "综合分析数据趋势",
        f"帮我分析 {c} 的问题", "深入分析错误原因",
        "评估项目的整体架构", "对比分析各个模块",
        "深度分析安全风险", "综合评估技术方案",
        "分析系统瓶颈在哪", f"帮我分析一下 {c}",
        f"深入分析 {c} 的瓶颈", f"对比 {c} 的不同方案",
        f"分析{c}的复杂度", f"评估{c}方案的优缺点",
        f"深度分析{c}逻辑", f"综合分析{c}风险",
        "帮我做深度分析", "分析一下整体架构",
        f"对比分析{c}和{c2}", f"评估{c}的技术选型",
        f"深入分析 {f} 中的 {c}", f"评估 {f2} 对 {c} 的影响",
        f"综合评估 {c} 在 {f} 中的应用", f"深度对比 {c} 和 {c2} 的区别",
        f"分析 {f} 和 {f2} 的架构差异", f"帮我深入分析 {c} 的发展趋势",
        f"评估 {c} 方案的可行性", f"综合分析 {c} 的优缺点",
        f"对比 {c} 和 {c2} 的实际效果", f"深度分析 {c} 的应用场景",
    ]

    sentence = _pick(base_templates)
    if random.random() < 0.25:
        sentence = _pick(POLITE) + sentence
    return sentence


def gen_vague():
    """VAGUE 类：打招呼、闲聊、无操作意图、边界case。"""
    greetings = [
        "你好", "嗨", "在吗", "hi", "hello", "hey",
        "早上好", "下午好", "晚上好", "早安", "晚安",
        "嘿", "喂", "哈喽", "哟", "哟呵",
    ]
    chitchat = [
        "你是谁", "你能做什么", "介绍一下自己", "你是AI吗",
        "你叫什么名字", "你多大了", "你是男的女的",
        "今天天气怎么样", "明天会下雨吗", "今天星期几",
        "讲个笑话", "给我讲个故事", "推荐一部电影",
        "推荐几本书", "怎么减肥", "学英语怎么学",
        "今天吃什么", "晚安", "明天见",
        "你喜欢什么", "你有什么爱好", "你会唱歌吗",
        "你会画画吗", "你会写诗吗", "世界有多大",
        "宇宙有外星人吗", "生命的意义是什么",
    ]
    vague_requests = [
        "帮我一下", "请问", "我有个问题", "等等",
        "你能帮我吗", "需要帮助", "怎么用", "这个怎么做",
        "帮帮忙", "求助", "请教一下", "问一下",
    ]
    filler = [
        "嗯", "好的", "谢谢", "哈哈", "哦", "啊",
        "嗯嗯", "好吧", "行", "ok", "OK", "收到",
        "了解", "明白了", "知道了", "可以", "没问题",
        "牛", "厉害", "可以啊", "666", "好的好的",
        "emmm", "mmm", "呃", "额", "啊这",
        "哈哈哈", "嘻嘻", "嘿嘿", "嗯好的",
        "再见", "拜拜", "bye", "回头见",
    ]
    random_chitchat = [
        "123", "test", "随便打的", "你能识别中文吗",
        "这是一条测试消息", "随便聊聊", "无聊",
        "今天心情不错", "心情不好", "好累啊",
        "今天好开心", "明天要加班", "好困",
        "你觉得呢", "你说得对", "有道理",
        "我同意", "不太确定", "再说一遍",
    ]

    pool = (greetings * 3 + chitchat * 3 + vague_requests * 3 +
            filler * 2 + random_chitchat * 2)

    sentence = _pick(pool)
    if random.random() < 0.2:
        sentence = _pick(POLITE) + sentence
    if random.random() < 0.15:
        sentence += "？"
    return sentence


# ============================================================
#  主生成器
# ============================================================

GENERATORS = {
    "add_file": gen_add_file,
    "delete_file": gen_delete_file,
    "modify_file": gen_modify_file,
    "refactor": gen_refactor,
    "query": gen_query,
    "deep_search": gen_deep_search,
    "vague": gen_vague,
}

SAMPLES_PER_CLASS = 1000


def augment(text: str) -> str:
    prefixes = ["请", "帮我", "能不能", "我想", "帮忙", "麻烦", "可以帮我", "麻烦帮我"]
    suffixes = ["一下", "吧", "好吗", "谢谢", "可以吗", "哈", "呗", ""]
    r = random.random()
    if r < 0.1 and len(text) > 4:
        pos = random.randint(1, len(text) - 2)
        return text[:pos] + text[pos + 1:]
    if r < 0.25:
        return _pick(prefixes) + text + _pick(suffixes)
    if r < 0.35:
        return text + "？"
    if r < 0.4:
        return text + "。"
    return text


def main():
    rows = []
    for label, gen_fn in GENERATORS.items():
        for i in range(SAMPLES_PER_CLASS):
            text = gen_fn()
            if random.random() < 0.4:
                text = augment(text)
            rows.append((text, label))

    random.shuffle(rows)

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "intent_corpus.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples → {path}")
    from collections import Counter
    counts = Counter(label for _, label in rows)
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
