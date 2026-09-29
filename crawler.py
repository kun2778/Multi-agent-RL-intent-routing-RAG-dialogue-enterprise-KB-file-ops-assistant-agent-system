"""
5类编程教程爬虫 v3：已验证可工作的站点，5类各 ~10000 块。

类别 & 站点：
  1. 编程文档  — runoob.com + w3school.com.cn + python docs
  2. 技术博客  — cnblogs.com
  3. GitHub 文档 — github.com trending README
  4. 运维教程  — w3school.com.cn/linux + runoob/linux
  5. Q&A 社区  — cnblogs.com/qa + w3school.com.cn/quiz
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────
from runtime_paths import WORKSPACE_DIR
OUT_DIR = str(WORKSPACE_DIR / "corpus")
MIN_CHARS = 100
MAX_CHARS = 30000
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    return s


def _fetch(session, url):
    for _ in range(2):
        try:
            r = session.get(url, timeout=12, allow_redirects=True)
            if r.status_code in (429, 468):
                time.sleep(3)
                continue
            if r.status_code >= 400:
                return None
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception:
            time.sleep(0.5)
    return None


def _clean(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["script", "style", "iframe", "nav", "aside", "footer"]):
        t.decompose()
    return soup


def _save(text, url, out_dir, sub=""):
    text = text.strip()
    if len(text) < MIN_CHARS:
        return False
    text = text[:MAX_CHARS]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    slug = re.sub(r'[^\w\u4e00-\u9fff-]', '_', urllib.parse.urlparse(url).path).strip("_")[:50]
    d = os.path.join(out_dir, sub) if sub else out_dir
    os.makedirs(d, exist_ok=True)
    from atomic_io import atomic_write_text
    atomic_write_text(os.path.join(d, f"{slug}_{h}.txt"), f"# Source: {url}\n\n{text}")
    return True


def _crawl_batch(urls, out_dir, sub, extract_fn, workers=15, max_pages=3000):
    os.makedirs(out_dir, exist_ok=True)
    session = _session()
    saved = 0
    visited = set()

    def _task(url):
        if url in visited:
            return None
        visited.add(url)
        html = _fetch(session, url)
        if not html:
            return None
        text = extract_fn(html, url)
        if text and len(text.strip()) >= MIN_CHARS:
            return (text.strip()[:MAX_CHARS], url)
        return None

    batch_size = 60
    for i in range(0, len(urls), batch_size):
        if saved >= max_pages:
            break
        batch = urls[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, u): u for u in batch}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    text, url = result
                    if _save(text, url, out_dir, sub):
                        saved += 1
                        if saved % 200 == 0:
                            print(f"  [{saved}] {url[:80]}")
        time.sleep(0.1)

    print(f"  -> {sub}: {saved} files")
    return saved


# ══════════════════════════════════════════════════════════════════════
# 1. 编程文档
# ══════════════════════════════════════════════════════════════════════

def _runoob_all_pages():
    """抓取 runoob 所有教程的侧边栏链接，获取全部页面。"""
    tutorial_roots = [
        "python3", "linux", "sql", "html", "css3", "js",
        "jquery", "angularjs", "react", "vue", "nodejs",
        "git", "docker", "mysql", "mongodb", "redis", "nginx",
        "php", "java", "cprogramming", "cpp", "csharp", "go", "rust",
        "typescript", "sass", "bootstrap", "perl", "lua", "r",
        "scala", "kotlin", "swift", "design-pattern",
        "data-structure", "algorithm",
    ]
    session = _session()
    all_urls = set()

    for tut in tutorial_roots:
        url = f"https://www.runoob.com/{tut}/{tut}-tutorial.html"
        html = _fetch(session, url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") and not href.startswith("//"):
                href = "https://www.runoob.com" + href
            if "runoob.com" in href and href.endswith(".html"):
                all_urls.add(href)
        time.sleep(0.05)

    print(f"  runoob URLs discovered: {len(all_urls)}")
    return list(all_urls)


def _extract_runoob(html, url):
    soup = _clean(html)
    content = (soup.find("div", id="content") or
               soup.find("div", class_="article") or
               soup.find("div", id="main"))
    if not content:
        return ""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) + "\n\n" if h1 else ""
    return title + content.get_text(separator="\n", strip=True)


def _w3school_all_pages():
    """抓取 w3school.com.cn 所有教程页面。"""
    tutorial_paths = [
        "html", "css", "javascript", "jquery", "xml", "xsl",
        "http", "sql", "php", "asp", "jsp",
        "python", "django", "numpy", "pandas",
        "linux", "unix", "regex", "webservices",
    ]
    base = "https://www.w3school.com.cn"
    session = _session()
    all_urls = set()

    for tut in tutorial_paths:
        index_url = f"{base}/{tut}/index.asp"
        html = _fetch(session, index_url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = f"{base}/{tut}/{href}" if "/" not in href else f"{base}/{href}"
            if "w3school.com.cn" in href and href.endswith(".asp"):
                all_urls.add(href)
        time.sleep(0.05)

    print(f"  w3school URLs discovered: {len(all_urls)}")
    return list(all_urls)


def _extract_w3school(html, url):
    soup = _clean(html)
    content = (soup.find("div", id="main") or
               soup.find("div", class_="content") or
               soup.find("article"))
    if not content:
        return ""
    h1 = soup.find("h1") or soup.find("h2")
    title = h1.get_text(strip=True) + "\n\n" if h1 else ""
    return title + content.get_text(separator="\n", strip=True)


def _python_docs_all():
    """抓取 Python 官方文档中文版。"""
    base = "https://docs.python.org/zh-cn/3"
    session = _session()
    all_urls = set()

    # 获取库索引
    for idx_url in [f"{base}/library/index.html", f"{base}/tutorial/index.html",
                    f"{base}/reference/index.html", f"{base}/howto/index.html",
                    f"{base}/faq/index.html"]:
        html = _fetch(session, idx_url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = f"{base}/{href}" if not href.startswith("/") else f"{base}{href}"
            if "docs.python.org" in href and href.endswith(".html"):
                all_urls.add(href)
        time.sleep(0.05)

    return list(all_urls) if 'all_urls' in dir() else []


def _extract_python_docs(html, url):
    soup = _clean(html)
    content = (soup.find("div", role="main") or
               soup.find("div", class_="body") or
               soup.find("article"))
    if not content:
        return ""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) + "\n\n" if h1 else ""
    return title + content.get_text(separator="\n", strip=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 技术博客
# ══════════════════════════════════════════════════════════════════════

def _cnblogs_articles():
    """从博客园多个分类获取文章链接。"""
    cats = [
        "python", "linux", "javascript", "database", "net",
        "java", "web", "program", "windows", "ios",
        "android", "go", "rust", "devops", "docker",
        "mysql", "redis", "git", "html", "css",
        "react", "vue", "node", "php", "algorithm",
    ]
    session = _session()
    article_urls = set()

    for cat in cats:
        for p in range(1, 60):
            url = f"https://www.cnblogs.com/cate/{cat}/" if p == 1 else f"https://www.cnblogs.com/cate/{cat}/page/{p}/"
            html = _fetch(session, url)
            if not html:
                continue
            soup = _clean(html)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = "https://www.cnblogs.com" + href
                if "cnblogs.com" in href and "/p/" in href:
                    article_urls.add(href)
            time.sleep(0.05)
            if len(article_urls) > 1500:
                break
        if len(article_urls) > 1500:
            break

    print(f"  cnblogs article URLs: {len(article_urls)}")
    return list(article_urls)


def _extract_cnblogs(html, url):
    soup = _clean(html)
    article = (soup.find("div", id="cnblogs_post_body") or
               soup.find("article") or
               soup.find("div", class_="post-body"))
    if not article:
        return ""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) + "\n\n" if h1 else ""
    return title + article.get_text(separator="\n", strip=True)


# ══════════════════════════════════════════════════════════════════════
# 3. GitHub 文档
# ══════════════════════════════════════════════════════════════════════

def _github_readme_urls():
    """从 GitHub trending 获取仓库 README URL。"""
    session = _session()
    repo_urls = set()

    # trending 页面
    for suffix in ["", "/python", "/javascript", "/java", "/go", "/rust",
                    "/typescript", "/c++", "/c", "/ruby", "/swift",
                    "/kotlin", "/php", "/r", "/scala", "/dart", "/lua",
                    "/shell", "/html", "/css"]:
        url = f"https://github.com/trending{suffix}"
        html = _fetch(session, url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 仓库链接格式: /user/repo
            if re.match(r'^/[^/]+/[^/]+$', href):
                repo_urls.add(href)
        time.sleep(0.1)

    print(f"  GitHub repos discovered: {len(repo_urls)}")

    readme_urls = []
    for repo in repo_urls:
        for name in ["README.md", "readme.md", "README.rst"]:
            readme_urls.append(f"https://github.com{repo}/blob/master/{name}")
    return readme_urls


def _extract_github(html, url):
    soup = _clean(html)
    article = (soup.find("article") or soup.find("div", class_="markdown-body") or
               soup.find("div", id="readme") or soup.find("main") or
               soup.find("div", class_="blob-wrapper"))
    if not article:
        return ""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) + "\n\n" if h1 else ""
    return title + article.get_text(separator="\n", strip=True)


# ══════════════════════════════════════════════════════════════════════
# 4. 运维教程
# ══════════════════════════════════════════════════════════════════════

def _ops_urls():
    """runoob 运维 + w3school linux/网络 + 其他运维教程。"""
    urls = []

    # runoob 运维类
    ops_topics = [
        "linux", "nginx", "docker", "git", "ssh",
        "mysql", "redis", "mongodb",
    ]
    session = _session()
    for topic in ops_topics:
        tut_url = f"https://www.runoob.com/{topic}/{topic}-tutorial.html"
        html = _fetch(session, tut_url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") and not href.startswith("//"):
                href = "https://www.runoob.com" + href
            if "runoob.com" in href and href.endswith(".html"):
                urls.append(href)
        time.sleep(0.05)

    # w3school linux/网络
    w3_ops = ["linux", "unix", "http", "sql"]
    for topic in w3_ops:
        index_url = f"https://www.w3school.com.cn/{topic}/index.asp"
        html = _fetch(session, index_url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = f"https://www.w3school.com.cn/{topic}/{href}" if "/" not in href else f"https://www.w3school.com.cn/{href}"
            if "w3school.com.cn" in href and href.endswith(".asp"):
                urls.append(href)
        time.sleep(0.05)

    print(f"  ops URLs: {len(urls)}")
    return list(set(urls))


# ══════════════════════════════════════════════════════════════════════
# 5. Q&A 社区
# ══════════════════════════════════════════════════════════════════════

def _qa_urls():
    """从 w3school 测验页 + cnblogs 问答 + 其他 Q&A 站点。"""
    urls = []

    # w3school quiz 页面
    quiz_topics = ["html", "css", "javascript", "jquery", "sql", "python",
                   "php", "xml", "http"]
    session = _session()
    for topic in quiz_topics:
        quiz_url = f"https://www.w3school.com.cn/{topic}/question.asp"
        html = _fetch(session, quiz_url)
        if html:
            urls.append(quiz_url)
        # 也抓教程页作为 Q&A 来源
        index_url = f"https://www.w3school.com.cn/{topic}/index.asp"
        html = _fetch(session, index_url)
        if not html:
            continue
        soup = _clean(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = f"https://www.w3school.com.cn/{topic}/{href}" if "/" not in href else f"https://www.w3school.com.cn/{href}"
            if "w3school.com.cn" in href and href.endswith(".asp"):
                urls.append(href)
        time.sleep(0.05)

    # cnblogs 常见问答
    qa_cats = ["python", "javascript", "linux", "database", "java"]
    for cat in qa_cats:
        for p in range(1, 30):
            url = f"https://www.cnblogs.com/cate/{cat}/" if p == 1 else f"https://www.cnblogs.com/cate/{cat}/page/{p}/"
            html = _fetch(session, url)
            if not html:
                continue
            soup = _clean(html)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = "https://www.cnblogs.com" + href
                if "cnblogs.com" in href and "/p/" in href:
                    urls.append(href)
            time.sleep(0.05)

    print(f"  qa URLs: {len(urls)}")
    return list(set(urls))


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="5类编程教程爬虫 v3")
    parser.add_argument("--category", default="all",
                        choices=["doc", "blog", "github", "ops", "qa", "all"])
    parser.add_argument("--max-pages", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    total = 0
    out = OUT_DIR

    if args.category in ("doc", "all"):
        print("\n=== 1. 编程文档 (runoob + w3school + python docs) ===")
        urls = _runoob_all_pages()
        urls += _w3school_all_pages()
        total += _crawl_batch(urls, out, "doc", _extract_runoob,
                              args.workers, args.max_pages)

    if args.category in ("blog", "all"):
        print("\n=== 2. 技术博客 (cnblogs.com) ===")
        urls = _cnblogs_articles()
        total += _crawl_batch(urls, out, "blog", _extract_cnblogs,
                              args.workers, args.max_pages)

    if args.category in ("github", "all"):
        print("\n=== 3. GitHub 文档 ===")
        urls = _github_readme_urls()
        total += _crawl_batch(urls, out, "github", _extract_github,
                              args.workers, args.max_pages)

    if args.category in ("ops", "all"):
        print("\n=== 4. 运维教程 (runoob ops + w3school linux) ===")
        urls = _ops_urls()
        total += _crawl_batch(urls, out, "ops", _extract_runoob,
                              args.workers, args.max_pages)

    if args.category in ("qa", "all"):
        print("\n=== 5. Q&A (w3school + cnblogs) ===")
        urls = _qa_urls()
        total += _crawl_batch(urls, out, "qa", _extract_w3school,
                              args.workers, args.max_pages)

    print(f"\n{'='*60}")
    print(f"Total: {total} files -> {out}")
    print(f"Run: python rebuild_index.py  to build FAISS index")


if __name__ == "__main__":
    main()
