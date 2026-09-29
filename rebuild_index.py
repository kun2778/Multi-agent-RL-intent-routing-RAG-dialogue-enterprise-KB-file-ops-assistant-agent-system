"""
重建 FAISS 向量索引：从 workspace/corpus/ 读取爬取的文本，分块后构建索引。

用法：
  python rebuild_index.py                  # 默认从 workspace/corpus/ 构建
  python rebuild_index.py --source ./data  # 自定义源目录
  python rebuild_index.py --stats          # 仅查看统计
"""

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from faiss_store import FaissVectorStore
from embedding import create_embedder
from runtime_paths import WORKSPACE_DIR, FAISS_DIR


def collect_text_files(source_dir: str) -> dict:
    """收集目录下所有可读文本文件。"""
    readable_exts = {
        "txt", "md", "yaml", "yml", "py", "json", "xml", "ini", "cfg",
        "html", "js", "ts", "c", "cpp", "h", "java", "go", "rs", "sh",
        "bat", "ps1", "sql", "log", "toml", "csv",
    }
    documents = {}
    for dirpath, _, filenames in os.walk(source_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lstrip(".").lower()
            if ext not in readable_exts:
                continue
            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, source_dir)
            try:
                with open(abs_path, "rb") as f:
                    raw = f.read()
                for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        if len(text.strip()) > 50:
                            documents[rel_path] = text
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue
            except OSError:
                continue
    return documents


def main():
    parser = argparse.ArgumentParser(description="重建 FAISS 向量索引")
    parser.add_argument("--source", default=None, help="文本源目录")
    parser.add_argument("--persist", default=None, help="索引存储目录")
    parser.add_argument("--name", default="default", help="索引名称")
    parser.add_argument("--chunk-size", type=int, default=500, help="块大小")
    parser.add_argument("--overlap", type=int, default=75, help="重叠字符数")
    parser.add_argument("--stats", action="store_true", help="仅显示统计")
    parser.add_argument("--embedder", default="auto", help="嵌入器后端 (auto/sentence_transformers/tfidf)")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="sentence-transformers 模型名")
    args = parser.parse_args()

    persist = args.persist or str(FAISS_DIR)
    source = args.source or str(WORKSPACE_DIR / "corpus")

    embedder = create_embedder(backend=args.embedder, model_name=args.model)
    print(f"Using embedder: {type(embedder).__name__} (dimension={embedder.dimension})")

    store = FaissVectorStore(
        dimension=embedder.dimension,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        persist_dir=persist,
    )

    if args.stats:
        if store.load(args.name):
            print(f"Index: {args.name}")
            for k, v in store.get_stats().items():
                print(f"  {k}: {v}")
            print("\nSource distribution:")
            for src, cnt in store.get_source_stats().items():
                print(f"  {src}: {cnt} chunks")
        else:
            print("No index found.")
        return

    # 收集文件
    source = os.path.abspath(source)
    print(f"Source: {source}")
    documents = collect_text_files(source)
    print(f"Found {len(documents)} text files")

    if not documents:
        print("No files to index.")
        return

    # 构建索引
    store.build_from_documents(documents)
    store.save(args.name)

    print(f"\nIndex saved to {persist}/{args.name}/")
    print(f"Stats: {store.get_stats()}")


if __name__ == "__main__":
    main()
