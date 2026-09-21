"""生成项目专用中文意图分类 CSV 数据集。

7类意图 × 每类 ~500 条 = ~3500 条训练样本。
输出: datasets/intent_corpus.csv (text,label)
"""

import csv
import os
import random
import re

random.seed(42)

FILES = [
    "notes.txt", "config.yaml", "main.py", "app.json", "report.md",
    "readme.md", "data.csv", "settings.py", "index.html", "style.css",
    "utils.js", "test.py", "requirements.txt", "Dockerfile", "package.json",
    "train.py", "model.py", "utils.go", "server.rs", "Makefile",
    "database.sql", "schema.prisma", "logo.png", "backup.zip", "log.txt",
]

CONTENT_FRAGMENTS = [
    "项目配置信息", "用户登录日志", "测试用例结果", "系统运行状态",
    "数据库连接池", "API 接口文档", "部署脚本内容", "错误处理逻辑",
    "性能优化方案", "版本更新记录", "权限管理规则", "缓存策略配置",
    "消息队列参数", "监控告警阈值", "代码审查意见", "需求分析文档",
]


def gen_add_file():
    templates = [
        "创建文件 {f}", "新建一个 {f}", "帮我写一个 {f}", "写入 {f}",
        "添加文件 {f} 内容为 {c}", "新建 {f} 保存 {c}", "生成 {f}",
        "帮我创建 {f} 然后写入内容", "创建一个叫 {f} 的文件",
        "把以下内容保存到 {f}: {c}", "写一个 {f} 文件",
        "帮我新建 {f}", "创建 {f} 放在当前目录", "添加 {f} 到项目中",
        "新建文件 {f} 里面写 {c}", "帮我生成一个 {f}",
        "创建 {f}，内容是 {c}", "写入新文件 {f}", "建一个 {f}",
        "帮我写 {f} 这个文件", "新建 {f} 并写入 {c}",
        "创建名为 {f} 的文件", "帮忙创建 {f}", "增加一个 {f}",
        "创建文件夹里的 {f}", "写个 {f} 吧", "帮我建 {f}",
        "新建一个叫 {f} 的文档", "创建 {f} 文件", "写一个 {f} 放进去",
        "帮我写个 {f} 出来", "添加 {f} 文件内容为 {c}",
        "生成一个新的 {f}", "创建空文件 {f}", "新建空白 {f}",
        "帮我创建一个 {f} 文件", "写入数据到 {f}", "创建 {f} 存储 {c}",
        "帮我在 src 下创建 {f}", "新建 {f} 在 docs 目录",
        "创建配置文件 {f}", "写一个脚本 {f}", "帮我写脚本 {f}",
        "生成报告文件 {f}", "创建 {f} 记录 {c}", "新建 {f} 保存日志",
        "帮我写一个 {f} 文件内容是 {c}", "创建 {f} 然后写 {c}",
        "新建文件 {f} 用来存 {c}", "写一个 {f} 把 {c} 放进去",
        "创建一个 {f} 记录一下 {c}",
    ]
    return random.choice(templates)


def gen_delete_file():
    templates = [
        "删除文件 {f}", "移除 {f}", "把 {f} 删掉", "清理 {f}",
        "帮我删除 {f}", "删掉 {f}", "移除掉 {f}", "把 {f} 从项目里删去",
        "删除 {f} 这个文件", "清理掉 {f}", "把 {f} 给删了",
        "帮我把 {f} 删了", "删去 {f}", "去掉 {f}",
        "把那个 {f} 删掉", "帮我移除 {f}", "删除掉 {f}",
        "清理一下 {f}", "把 {f} 删除", "删掉那个 {f}",
        "移除 {f} 文件", "帮我把 {f} 移除", "把 {f} 清理掉",
        "删除 {f} 不要了", "删掉这个 {f}", "清理掉旧的 {f}",
        "帮我把 {f} 去掉", "删除掉那个旧的 {f}", "把 {f} 从目录里删去",
        "不需要 {f} 了删掉", "把 {f} 给移除掉", "帮我删掉 {f}",
        "删除备份 {f}", "移除临时文件 {f}", "清理日志 {f}",
        "把过期的 {f} 删了", "删除多余的 {f}", "清理无用的 {f}",
        "删掉旧版本的 {f}", "帮我删除那个 {f}", "移除 {f} 不用了",
        "把 {f} 拿掉", "删除 {f} 释放空间", "帮我把 {f} 清理了",
        "删掉没用的 {f}", "删除过时的 {f}", "移除废弃的 {f}",
        "清理不再需要的 {f}", "把 {f} 从仓库删去", "删除 {f} 保持整洁",
    ]
    return random.choice(templates)


def gen_modify_file():
    templates = [
        "修改文件 {f} 内容为 {c}", "更新 {f} 改成 {c}",
        "编辑 {f}", "帮我改一下 {f}", "把 {f} 的内容替换一下",
        "更新 {f} 的内容", "修改 {f} 中的配置",
        "改一下 {f} 把内容换成 {c}", "编辑 {f} 改为 {c}",
        "帮我更新 {f}", "修改 {f} 改成 {c}",
        "把 {f} 更新为 {c}", "更新 {f} 里的 {c}",
        "改 {f} 中的参数", "修改 {f} 把 {c} 换掉",
        "编辑文件 {f} 加上 {c}", "帮我修改 {f}",
        "更新 {f} 的设置", "改一下 {f} 的内容",
        "修改 {f} 把代码改一下", "更新 {f} 修复bug",
        "编辑 {f} 增加新功能", "帮我改 {f} 里的配置",
        "修改 {f} 加上 {c}", "把 {f} 改成 {c}",
        "更新 {f} 里的内容为 {c}", "帮我编辑 {f}",
        "修改一下 {f}", "更新 {f} 的代码",
        "把 {f} 的内容修改为 {c}", "帮我把 {f} 改了",
        "编辑 {f} 修改为 {c}", "更新 {f} 的配置文件",
        "修改 {f} 里的 {c}", "帮我更新一下 {f}",
        "把 {f} 改一改", "修改 {f} 变成 {c}",
        "更新 {f} 加入 {c}", "编辑 {f} 替换内容",
        "帮我修改一下 {f} 的内容", "把 {f} 重新编辑一下",
        "更新 {f} 里的数据", "修改 {f} 的参数为 {c}",
        "帮我把 {f} 更新一下", "编辑 {f} 加入新内容",
        "修改 {f} 增加 {c}", "更新 {f} 的文档",
        "帮我把 {f} 改成 {c}", "修改 {f} 的代码",
    ]
    return random.choice(templates)


def gen_refactor():
    templates = [
        "整理文件夹", "重构项目结构", "把文件按类型分类归档",
        "帮我整理一下目录", "重命名文件", "移动文件到对应文件夹",
        "代码重构", "优化项目结构", "整理一下 src 目录",
        "把代码按模块分目录", "重构一下代码结构",
        "帮我整理文件夹", "把文件归类整理",
        "重新组织目录结构", "整理项目文件",
        "帮我把文件按类型分一下", "重构目录",
        "移动文件到正确的位置", "整理一下代码目录",
        "重命名一下文件", "优化一下目录结构",
        "帮我把文件整理一下", "按功能模块分类文件",
        "重构文件组织方式", "整理文档目录",
        "帮我归档旧文件", "重新整理项目结构",
        "把测试文件移到 tests 目录", "整理配置文件到 config 目录",
        "帮我把代码重构一下", "文件重新分类",
        "优化文件结构", "整理一下项目目录",
        "按类型把文件分好", "重构项目目录结构",
        "帮我把文件按日期归档", "整理日志文件",
        "移动文档到 docs 目录", "重命名 main 为 app",
        "帮我整理资源文件", "把图片移到 assets 目录",
        "整理脚本到 scripts 目录", "按模块重组代码",
        "帮我归类一下文件", "重构整个项目",
        "把工具函数抽到 utils", "整理依赖文件",
        "帮我把文件系统理一下", "优化目录布局",
        "重新组织文件结构", "整理一下整个项目",
    ]
    return random.choice(templates)


def gen_query():
    templates = [
        "查询文件 {f} 里写了什么", "搜索 {f} 的内容", "{f} 存在吗",
        "帮我找一下 {f}", "查看 {f}", "{f} 里面有什么",
        "搜索强化学习", "查一下这个文件", "找到红警3的启动程序在哪里",
        "查看 {f} 的代码", "搜索关于机器学习的资料",
        "查找 {f} 的位置", "查看 {f} 的内容",
        "搜索文件里的关键词", "帮我查一下 {f}",
        "{f} 文件在哪", "查看一下 {f}",
        "搜索包含 {c} 的文件", "查找相关文件",
        "这个项目有什么文件", "查看项目结构",
        "搜索 {c} 相关内容", "查一下 {f} 里写了啥",
        "帮我看看 {f}", "找一下 {c} 在哪里",
        "查看 {f} 的配置", "搜索代码中的函数",
        "查一下项目用了什么框架", "查看当前目录有哪些文件",
        "帮我查 {f}", "找找 {f} 这个文件",
        "搜索项目中的错误", "查看 {f} 的日志",
        "查一下系统状态", "查看文件列表",
        "搜索包含 {c} 的代码", "帮我找 {c}",
        "查看文档内容", "找一下配置文件在哪",
        "搜索 {f} 相关信息", "查一下这个目录",
        "帮我查看 {f} 的内容", "找找有没有 {f}",
        "查看 {f} 是否存在", "搜索最新的提交记录",
        "查一下 {c} 的定义", "看看 {f} 里有什么",
        "帮我搜索一下", "查找所有 Python 文件",
        "查看 README 内容", "搜索项目依赖",
    ]
    return random.choice(templates)


def gen_deep_search():
    templates = [
        "对比分析 {f} 和 {g}", "深度分析 {f} 的趋势与影响",
        "综合分析 {f} 和 {g} 的差异", "帮我比较一下 {f} 和 {g}",
        "分析 {f} 的优缺点", "深入研究 {f} 的发展方向",
        "评估 {f} 和 {g} 的区别", "对比 {f} 和 {g} 的性能",
        "深度分析项目的架构", "综合评估 {f} 的方案",
        "帮我分析一下 {f} 的影响", "比较 {f} 和 {g} 的优劣",
        "深入分析 {f} 的问题", "评估 {f} 的可行性",
        "分析项目的技术栈", "对比不同方案的利弊",
        "深度分析代码质量", "综合分析系统性能",
        "帮我评估一下 {f}", "深入研究 {f} 的实现",
        "分析 {f} 的发展趋势", "对比 {f} 和 {g} 的效果",
        "深度分析用户行为", "综合分析数据趋势",
        "帮我做一下 {f} 的分析", "深入分析错误原因",
        "评估项目的整体架构", "对比分析各个模块",
        "深度分析安全风险", "综合评估技术方案",
        "分析系统瓶颈在哪", "帮我分析 {f} 的问题",
        "深入分析性能瓶颈", "对比 {f} 的不同版本",
        "分析代码的复杂度", "评估方案的优缺点",
        "深度分析业务逻辑", "综合分析项目风险",
        "帮我分析一下趋势", "对比分析前后端",
        "分析数据库性能", "评估架构的可扩展性",
        "深入分析日志数据", "对比分析两个算法",
        "分析系统负载情况", "评估代码的可维护性",
        "深度分析网络延迟", "综合分析成本效益",
        "帮我做深度分析", "分析一下整体架构",
        "对比分析微服务和单体", "评估技术选型",
    ]
    return random.choice(templates)


def gen_vague():
    templates = [
        "你好", "嗨", "在吗", "你是谁", "帮我一下",
        "请问", "嗯", "好的", "谢谢", "哈哈",
        "今天天气怎么样", "讲个笑话", "随便聊聊",
        "你能做什么", "介绍一下自己", "无聊",
        "123", "test", "hello", "hi", "hey",
        "你是AI吗", "我有个问题", "等等",
        "哦", "啊", "嗯嗯", "好吧", "行",
        "牛", "厉害", "可以啊", "没问题",
        "今天心情不错", "明天会下雨吗", "你喜欢什么",
        "给我讲个故事", "推荐一部电影", "怎么减肥",
        "学英语怎么学", "推荐几本书", "今天吃什么",
        "晚安", "早上好", "中午好", "下午好",
        "你会唱歌吗", "你会画画吗", "你会写诗吗",
        "世界有多大", "宇宙有外星人吗", "生命的意义是什么",
        "help", "what", "who", "when", "where",
        "啊这", "emmm", "mmm", "呃", "额",
        "哈哈哈", "嘻嘻", "嘿嘿", "嗯好的", "收到",
        "了解", "明白了", "知道了", "OK", "ok",
        "再见", "拜拜", "bye", "See you", "回头见",
    ]
    return random.choice(templates)


GENERATORS = {
    "add_file": gen_add_file,
    "delete_file": gen_delete_file,
    "modify_file": gen_modify_file,
    "refactor": gen_refactor,
    "query": gen_query,
    "deep_search": gen_deep_search,
    "vague": gen_vague,
}

SAMPLES_PER_CLASS = 500


def augment(text: str) -> str:
    """简单数据增强。"""
    prefixes = ["请", "帮我", "能不能", "我想", "帮忙", "麻烦", "可以帮我"]
    suffixes = ["一下", "吧", "好吗", "谢谢", "可以吗", ""]
    r = random.random()
    if r < 0.15 and len(text) > 3:
        pos = random.randint(1, len(text) - 2)
        return text[:pos] + text[pos + 1:]
    if r < 0.3:
        return random.choice(prefixes) + text + random.choice(suffixes)
    if r < 0.4:
        return text + "？"
    return text


def fill_template(template: str) -> str:
    f = random.choice(FILES)
    g = random.choice([x for x in FILES if x != f])
    c = random.choice(CONTENT_FRAGMENTS)
    return template.format(f=f, g=g, c=c)


def main():
    os.makedirs("datasets", exist_ok=True)
    rows = []

    for label, gen_fn in GENERATORS.items():
        templates = [gen_fn() for _ in range(20)]
        for i in range(SAMPLES_PER_CLASS):
            if i < len(templates):
                text = fill_template(templates[i])
            else:
                base = fill_template(random.choice(templates))
                text = augment(base)
            rows.append((text, label))

    random.shuffle(rows)

    path = os.path.join("datasets", "intent_corpus.csv")
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
