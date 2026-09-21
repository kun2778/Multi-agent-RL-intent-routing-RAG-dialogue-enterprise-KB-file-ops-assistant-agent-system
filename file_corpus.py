"""
文件语料库模块：管理文件系统的增/删/改/整理与向量检索。

重构后架构：
  - SQLite 注册表（替代 JSON，支持百万级文件）
  - 并行扫描器（os.scandir + 线程池 + 符号链接检测）
  - 增量索引（仅处理变更文件）
  - 可选 FaissVectorStore（500 字符分块、15% 重叠）
"""

import json
import os
import re
import shutil
import time
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss-cpu not installed. Using numpy-based fallback.")

from registry_db import RegistryDB, compute_file_hash
from parallel_scanner import ParallelScanner, ScanStats, READABLE_EXTENSIONS
from runtime_paths import DB_DIR, FAISS_DIR, WORKSPACE_DIR


# 可被整理分类的文件夹
FOLDER_CATEGORIES = {
    "code": [".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".sh", ".bat", ".ps1", ".html", ".css"],
    "config": [".yaml", ".yml", ".json", ".ini", ".cfg", ".toml", ".env", ".gitignore", ".requirements", ".xml"],
    "docs": [".txt", ".md", ".pdf", ".doc", ".docx", ".slnx", ".mui", ".log"],
    "data": [".csv", ".sql", ".xlsx", ".xls"],
}


def _tokenize(text: str) -> List[str]:
    """中英文混合分词：ASCII 单词 + CJK 单字 + CJK 二元组。"""
    tokens = re.findall(r'[A-Za-z0-9_]+', text.lower())
    cjk = re.findall(r'[\u4e00-\u9fff]', text)
    tokens.extend(cjk)
    tokens.extend([''.join(cjk[i:i + 2]) for i in range(len(cjk) - 1)])
    return tokens


class TfidfEmbedder:
    """TF-IDF 嵌入器，用于将文本映射为稠密向量（向量索引的输入）。"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, texts: List[str]):
        doc_counts = Counter()
        all_tokens: List[str] = []
        for text in texts:
            tokens = _tokenize(text)
            all_tokens.extend(tokens)
            for t in set(tokens):
                doc_counts[t] += 1
        total_docs = max(len(texts), 1)
        for token, count in doc_counts.items():
            self.idf[token] = np.log((total_docs + 1) / (count + 1)) + 1
        self.vocab = {t: i for i, t in enumerate(set(all_tokens))}
        self.is_fitted = True

    def transform(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            return np.random.randn(self.dimension).astype(np.float32)
        tokens = _tokenize(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        for token, count in counts.items():
            if token in self.vocab:
                idx = self.vocab[token] % self.dimension
                vector[idx] = (count / total) * self.idf.get(token, 1.0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector = np.random.randn(self.dimension).astype(np.float32) * 0.01
        return vector


class VectorIndex:
    """基于 FAISS 的向量索引，含 numpy 回退方案。"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = None
        self.vectors: List[np.ndarray] = []
        self.metadatas: List[Dict] = []
        self.doc_ids: List[str] = []

    def add(self, vectors: np.ndarray, documents: List[Dict], doc_ids: List[str]):
        if FAISS_AVAILABLE:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
        else:
            for v in vectors:
                self.vectors.append(v.copy())
        for i, doc in enumerate(documents):
            self.metadatas.append(doc)
            self.doc_ids.append(doc_ids[i])

    def reset(self):
        self.index = None
        self.vectors = []
        self.metadatas = []
        self.doc_ids = []

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            q = query_vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(q)
            scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
            results = [self.metadatas[idx] for idx in indices[0] if idx < len(self.metadatas)]
            return results, scores[0].tolist()
        if not self.vectors:
            return [], []
        q = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        sims = []
        for v in self.vectors:
            vn = v / (np.linalg.norm(v) + 1e-9)
            sims.append(float(np.dot(q, vn)))
        order = np.argsort(sims)[::-1][:top_k]
        results = [self.metadatas[i] for i in order]
        return results, [sims[i] for i in order]

    def stats(self) -> Dict:
        if FAISS_AVAILABLE and self.index is not None:
            return {"total_docs": self.index.ntotal, "dimension": self.dimension}
        return {"total_docs": len(self.vectors), "dimension": self.dimension}


def _read_text(path: str) -> Optional[str]:
    """尝试以多种编码读取文本文件；二进制文件返回 None。"""
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    if ext not in READABLE_EXTENSIONS:
        return None
    try:
        raw = open(path, "rb").read()
        for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return raw.decode("latin-1", errors="ignore")
    except OSError:
        return None


class FileCorpus:
    """文件语料库模块：管理文件系统的增/删/改/整理与向量检索。

    重构后特性：
      - SQLite 注册表（支持百万级文件，不存全文）
      - 并行扫描（os.scandir + 线程池 + 符号链接检测）
      - 增量索引（仅处理变更文件）
      - 可选 FaissVectorStore（500 字符分块、15% 重叠）
    """

    def __init__(self, root: str = None,
                 use_faiss_store: bool = False, faiss_store_name: str = "default",
                 max_workers: int = 8, max_file_size: int = 5 * 1024 * 1024,
                 excluded_dirs: Optional[Set[str]] = None):
        """
        Args:
            root: 扫描根目录（支持任意路径，如 "C:\\"）
            use_faiss_store: 是否启用 FaissVectorStore
            faiss_store_name: FAISS 存储名称
            max_workers: 并行扫描线程数
            max_file_size: 最大文件大小（字节），超过跳过
            excluded_dirs: 额外排除的目录路径集合
        """
        self.root = os.path.abspath(root or str(WORKSPACE_DIR))
        self.trash_dir = os.path.join(self.root, ".trash")
        self.backup_dir = os.path.join(self.root, ".backup")
        self.embedder = TfidfEmbedder(dimension=128)
        self.index = VectorIndex(dimension=128)
        self.max_workers = max_workers
        self.max_file_size = max_file_size
        self.excluded_dirs = excluded_dirs

        # SQLite 注册表（存储在 .runtime/db/ 下）
        DB_DIR.mkdir(parents=True, exist_ok=True)
        db_name = f"corpus_{self.root.replace(':', '_').replace('\\', '_').replace('/', '_')}.db"
        self.db_path = str(DB_DIR / db_name)
        self.registry = RegistryDB(self.db_path)

        # FaissVectorStore 集成（可选）
        self.use_faiss_store = use_faiss_store
        self.faiss_store_name = faiss_store_name
        self.faiss_store = None
        if use_faiss_store:
            try:
                from faiss_store import FaissVectorStore
                FAISS_DIR.mkdir(parents=True, exist_ok=True)
                self.faiss_store = FaissVectorStore(
                    dimension=128, chunk_size=500, chunk_overlap=75,
                    persist_dir=str(FAISS_DIR),
                )
                self.faiss_store.load(faiss_store_name)
            except ImportError:
                print("[WARNING] faiss_store 模块不可用，回退到内置索引。")
                self.use_faiss_store = False

        self._ensure_dirs()
        self._sync_filesystem()
        self._rebuild_index()

    def _ensure_dirs(self):
        """确保必要目录存在。"""
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.trash_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def _save_registry(self):
        """兼容旧接口：SQLite 已自动保存，此方法为空操作。"""
        pass

    def _load_registry(self):
        """兼容旧接口：SQLite 已自动加载，此方法为空操作。"""
        pass

    def _sync_filesystem(self):
        """并行扫描文件系统，增量更新注册表。"""
        scanner = ParallelScanner(
            root=self.root,
            registry=self.registry,
            max_workers=self.max_workers,
            max_file_size=self.max_file_size,
            excluded_dirs=self.excluded_dirs,
            progress_callback=self._scan_progress,
        )
        stats = scanner.scan()
        self._last_scan_stats = stats

    def _scan_progress(self, scanned: int, total: int, path: str):
        """扫描进度回调。"""
        if scanned % 5000 == 0:
            print(f"[Scanner] 已扫描 {scanned} 个文件...")

    # ---------------------------------------------------------- 语料库增删改
    def list_documents(self, limit: int = 100) -> List[Dict]:
        """列出文件（默认最多100个）。"""
        readable = self.registry.get_all_readable()[:limit]
        return [
            {
                "path": d["path"], "title": d["title"], "type": d["type"],
                "tags": json.loads(d.get("tags", "[]")),
                "size": d["size"], "readable": bool(d["readable"]),
            }
            for d in readable
        ]

    def get_document(self, rel_path: str) -> Optional[Dict]:
        """获取单个文件元数据。"""
        matched = self._resolve_target(rel_path)
        if not matched:
            return None
        doc = self.registry.get_file(matched)
        if not doc:
            return None
        doc["tags"] = json.loads(doc.get("tags", "[]"))
        return doc

    def add_document(self, name: str, content: str, tags: List[str] = None) -> Tuple[bool, str, Dict]:
        """添加文件到语料库。"""
        if not name:
            return False, "未指定文件名", {}
        name = name.strip()
        if name != os.path.basename(name):
            rel_dir = os.path.dirname(name)
            os.makedirs(os.path.join(self.root, rel_dir), exist_ok=True)
            rel_path = name
        else:
            rel_path = name

        abs_path = os.path.join(self.root, rel_path)
        if os.path.exists(abs_path) and self._resolve_target(rel_path):
            return False, f"文件 {rel_path} 已存在，请改用修改操作", {}

        os.makedirs(os.path.dirname(abs_path) or self.root, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 更新注册表
        ext = os.path.splitext(rel_path)[1].lstrip('.').lower()
        stat_result = os.stat(abs_path)
        self.registry.upsert_file(
            path=rel_path,
            title=os.path.splitext(os.path.basename(rel_path))[0],
            file_type=ext or "unknown",
            size=stat_result.st_size,
            mtime=stat_result.st_mtime,
            readable=True,
            content_hash=compute_file_hash(abs_path),
            tags=json.dumps(tags or []),
        )

        doc = self.registry.get_file(rel_path)
        self._rebuild_index(silent=True)

        # 增量添加到 FaissVectorStore
        if self.use_faiss_store and self.faiss_store is not None:
            self.faiss_store.add_document(rel_path, content, {"tags": tags or []})
            self.faiss_store.save(self.faiss_store_name)

        return True, f"已添加文件 {rel_path}", doc

    def delete_document(self, target: str) -> Tuple[bool, str, Dict]:
        """删除文件（移入 .trash 回收站）。"""
        rel_path = self._resolve_target(target)
        if not rel_path:
            return False, f"未找到文件: {target}", {}
        abs_path = os.path.join(self.root, rel_path)
        doc = self.registry.get_file(rel_path) or {}

        try:
            stamp = str(int(time.time()))
            trash_path = os.path.join(self.trash_dir, f"{os.path.basename(rel_path)}.{stamp}")
            shutil.move(abs_path, trash_path)
        except OSError as e:
            return False, f"删除失败: {e}", {}

        self.registry.delete_file(rel_path)
        self._rebuild_index(silent=True)

        # 从 FaissVectorStore 移除
        if self.use_faiss_store and self.faiss_store is not None:
            self.faiss_store.remove_document(rel_path)
            self.faiss_store.save(self.faiss_store_name)

        return True, f"已删除文件 {rel_path}（已回收至 .trash）", doc

    def modify_document(self, target: str, new_content: str = None, new_name: str = None,
                        tags: List[str] = None) -> Tuple[bool, str, Dict]:
        """修改文件内容/标签/文件名（修改前自动备份）。"""
        rel_path = self._resolve_target(target)
        if not rel_path:
            return False, f"未找到文件: {target}", {}
        abs_path = os.path.join(self.root, rel_path)
        doc = self.registry.get_file(rel_path) or {}
        changed = []

        if new_content is not None:
            bak_path = os.path.join(self.backup_dir, f"{os.path.basename(rel_path)}.{int(time.time())}.bak")
            try:
                shutil.copy2(abs_path, bak_path)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                changed.append("内容")
            except OSError as e:
                return False, f"写入失败: {e}", {}

        if new_name is not None and new_name != rel_path:
            new_abs = os.path.join(self.root, new_name)
            try:
                os.rename(abs_path, new_abs)
                self.registry.delete_file(rel_path)
                rel_path = new_name
                abs_path = new_abs
                changed.append("文件名")
            except OSError as e:
                if "内容" not in changed:
                    return False, f"重命名失败: {e}", {}

        # 更新注册表
        ext = os.path.splitext(rel_path)[1].lstrip('.').lower()
        stat_result = os.stat(abs_path)
        self.registry.upsert_file(
            path=rel_path,
            title=os.path.splitext(os.path.basename(rel_path))[0],
            file_type=ext or "unknown",
            size=stat_result.st_size,
            mtime=stat_result.st_mtime,
            readable=True,
            content_hash=compute_file_hash(abs_path),
            tags=json.dumps(tags or doc.get("tags", [])),
        )

        if tags is not None:
            changed.append("标签")
        if not changed:
            changed = ["无"]

        self._rebuild_index(silent=True)

        # 更新 FaissVectorStore
        if self.use_faiss_store and self.faiss_store is not None:
            if new_content is not None or new_name is not None:
                old_path = target if new_name is None else target
                self.faiss_store.remove_document(old_path)
                final_path = rel_path
                final_content = new_content if new_content is not None else self.read_text(final_path) or ""
                self.faiss_store.add_document(final_path, final_content, {"tags": tags or doc.get("tags", [])})
                self.faiss_store.save(self.faiss_store_name)

        doc = self.registry.get_file(rel_path) or {}
        return True, f"已修改 {rel_path}（{', '.join(changed)}）", doc

    def refactor(self, mode: str = "by_type", targets: List[str] = None) -> Tuple[bool, str, List[Dict]]:
        """整理语料库：按类型归类到 code/config/docs/data/other 子文件夹。"""
        moved = []
        all_files = self.registry.get_all_paths()
        to_move = []
        if targets:
            for t in targets:
                rel = self._resolve_target(t)
                if rel:
                    to_move.append(rel)
        else:
            to_move = list(all_files)

        for rel_path in to_move:
            if "/" in rel_path and rel_path.split("/")[0] in FOLDER_CATEGORIES:
                continue
            ext = os.path.splitext(rel_path)[1].lower()
            folder = "other"
            for cat, exts in FOLDER_CATEGORIES.items():
                if ext in exts:
                    folder = cat
                    break
            target_folder = os.path.join(self.root, folder)
            os.makedirs(target_folder, exist_ok=True)
            new_rel = os.path.join(folder, os.path.basename(rel_path))
            if new_rel == rel_path:
                continue
            if new_rel in all_files or os.path.exists(os.path.join(self.root, new_rel)):
                stamp = str(int(time.time()))
                new_rel = os.path.join(folder, f"{os.path.splitext(os.path.basename(rel_path))[0]}_{stamp}{ext}")
            try:
                os.rename(os.path.join(self.root, rel_path), os.path.join(self.root, new_rel))
            except OSError:
                continue
            self.registry.delete_file(rel_path)
            # 重新注册新路径
            ext_new = os.path.splitext(new_rel)[1].lstrip('.').lower()
            abs_new = os.path.join(self.root, new_rel)
            stat_result = os.stat(abs_new)
            self.registry.upsert_file(
                path=new_rel,
                title=os.path.splitext(os.path.basename(new_rel))[0],
                file_type=ext_new or "unknown",
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
                readable=True,
                content_hash=compute_file_hash(abs_new),
            )
            moved.append({"from": rel_path, "to": new_rel, "category": folder})

        if not moved:
            return False, "没有需要移动的文件（已处于正确分类位置）", []

        self._rebuild_index(silent=True)
        summary = "\n".join(f"  {m['from']} -> {m['to']} ({m['category']})" for m in moved)
        return True, f"已整理 {len(moved)} 个文件:\n{summary}", moved

    # ------------------------------------------------------------------ 检索
    def _rebuild_index(self, silent: bool = False):
        """重建 TF-IDF 索引（增量模式）。

        Args:
            silent: 是否静默模式
        """
        self.index.reset()
        readable = self.registry.get_all_readable()

        if not readable:
            self.embedder = TfidfEmbedder(dimension=128)
            self.embedder.is_fitted = False
            return

        # 读取文件内容用于 TF-IDF
        texts = []
        meta = []
        ids = []
        for d in readable:
            abs_path = os.path.join(self.root, d["path"])
            content = _read_text(abs_path) or d["path"]
            texts.append(content)
            meta.append({
                "path": d["path"], "title": d["title"], "type": d["type"],
                "tags": json.loads(d.get("tags", "[]")), "readable": True,
                "content": content,
            })
            ids.append(d["path"])

        if texts:
            self.embedder.fit(texts)
            vecs = np.array([self.embedder.transform(t) for t in texts], dtype=np.float32)
            self.index.add(vecs, meta, ids)

        # 同步重建 FaissVectorStore（如果启用）
        if self.use_faiss_store and self.faiss_store is not None:
            # 检查索引是否为空或需要重建
            faiss_chunks_count = len(self.faiss_store.chunks) if self.faiss_store.chunks else 0
            current_files_count = len(readable)

            should_rebuild = False

            if self.faiss_store.index is None or self.faiss_store.index.ntotal == 0:
                should_rebuild = True
                if not silent:
                    print("[FileCorpus] FAISS索引为空，需要重建")
            elif faiss_chunks_count == 0:
                should_rebuild = True
                if not silent:
                    print("[FileCorpus] FAISS chunks为空，需要重建")

            # 检查 chunks 中的源文件是否与当前文件匹配
            if not should_rebuild and self.faiss_store.chunks:
                chunk_sources = set(c["source"] for c in self.faiss_store.chunks)
                current_sources = set(d["path"] for d in readable)
                common = chunk_sources.intersection(current_sources)
                if len(common) < len(chunk_sources) * 0.1:
                    should_rebuild = True
                    if not silent:
                        print(f"[FileCorpus] FAISS chunks源文件不匹配({len(common)}/{len(chunk_sources)})，需要重建")

            if should_rebuild:
                if not silent:
                    print(f"[FileCorpus] 开始重建FAISS索引 ({current_files_count} 文件)...")
                documents = {}
                doc_meta = {}
                for d in readable:
                    abs_path = os.path.join(self.root, d["path"])
                    content = _read_text(abs_path) or ""
                    documents[d["path"]] = content
                    doc_meta[d["path"]] = {"tags": json.loads(d.get("tags", "[]"))}
                self.faiss_store.build_from_documents(documents, doc_meta)
                self.faiss_store.save(self.faiss_store_name)
            else:
                if not silent:
                    print(f"[FileCorpus] 使用已加载的FAISS索引 ({faiss_chunks_count} 块, {current_files_count} 文件)")

    def search(self, query: str, top_k: int = 5, filter_noise: bool = True) -> List[Dict]:
        """向量检索：返回得分倒序文档。优先使用 FaissVectorStore（如启用）。支持噪声过滤。"""
        # 优先使用 FaissVectorStore（支持分块检索）
        if self.use_faiss_store and self.faiss_store is not None:
            faiss_results = self.faiss_store.search(query, top_k, filter_noise=filter_noise)
            if faiss_results:
                results = []
                for r in faiss_results:
                    results.append({
                        "path": r["source"],
                        "title": os.path.splitext(os.path.basename(r["source"]))[0],
                        "type": os.path.splitext(r["source"])[1].lstrip('.'),
                        "content": r["text"],
                        "tags": r.get("metadata", {}).get("tags", []),
                        "source": r["source"],
                        "page": 1,
                        "score": r["score"],
                        "chunk_index": r.get("chunk_index", 0),
                        "start_offset": r.get("start_offset", 0),
                        "end_offset": r.get("end_offset", 0),
                    })
                return results

        # 回退到内置索引
        if self.index.stats()["total_docs"] == 0:
            return []
        qv = self.embedder.transform(query) if self.embedder.is_fitted else np.zeros(128, dtype=np.float32)
        docs, scores = self.index.search(qv, top_k * 2 if filter_noise else top_k)  # 检索更多结果以便过滤
        
        results = []
        for doc, score in zip(docs, scores):
            # 噪声过滤
            if filter_noise and self._is_noise_file(doc["path"]):
                # 如果是噪声文件但得分很高，仍然保留但降低优先级
                if score > 0.7:
                    score *= 0.3  # 大幅降低噪声文件的得分
                else:
                    continue  # 跳过低分噪声文件
            
            results.append({
                "path": doc["path"], "title": doc["title"], "type": doc["type"],
                "content": doc.get("content", ""), "tags": doc.get("tags", []),
                "source": doc["path"], "page": 1, "score": float(score),
            })
        
        # 按得分排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        if not results:
            fuzzy = self._keyword_fallback(query)
            results = fuzzy
        return results[:top_k]
    
    def _is_noise_file(self, path: str) -> bool:
        """检查文件是否是噪声文件。"""
        noise_patterns = [
            '.operation_log.json',
            '.log',
            '.tmp',
            '.lock',
            '.min.js',
            '.map',
            '.cache',
            '.bak',
            '.orig',
            '.swp',
        ]
        path_lower = path.lower()
        return any(noise in path_lower for noise in noise_patterns)

    def _keyword_fallback(self, query: str, top_k: int = 5) -> List[Dict]:
        """关键词回退搜索。"""
        keys = set(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', query.lower()))
        scored = []
        readable = self.registry.get_all_readable()
        for d in readable:
            abs_path = os.path.join(self.root, d["path"])
            content = _read_text(abs_path) or ""
            text = (content + " " + d["path"]).lower()
            hit = sum(1 for k in keys if k in text)
            if hit > 0:
                scored.append({
                    "path": d["path"], "title": d["title"], "type": d["type"],
                    "content": content, "tags": json.loads(d.get("tags", "[]")),
                    "source": d["path"], "page": 1, "score": float(hit),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ 工具
    def _resolve_target(self, target: str) -> Optional[str]:
        """把用户输入的目标解析为注册表路径（支持精确/模糊匹配）。"""
        if not target:
            return None
        target = target.strip().replace("/", "\\").lstrip(".\\/")

        all_paths = self.registry.get_all_paths()

        # 精确匹配
        if target in all_paths:
            return target

        # 规范化匹配
        norm = {r: r.replace("\\", "/") for r in all_paths}
        t_norm = target.replace("\\", "/")
        exact = [r for r, n in norm.items() if n == t_norm]
        if exact:
            return exact[0]

        # basename 匹配
        base = os.path.basename(target)
        if not base:
            return None
        by_base = [r for r in all_paths if os.path.basename(r) == base]
        if "\\" not in target and "/" not in target:
            if len(by_base) == 1:
                return by_base[0]
        else:
            for r in by_base:
                if os.path.dirname(r).replace("\\", "") == os.path.dirname(target).replace("\\", ""):
                    return r

        # 模糊匹配
        fuzzy = [r for r in all_paths if base.lower() in r.lower()]
        return fuzzy[0] if len(fuzzy) == 1 else (fuzzy[0] if fuzzy else None)

    def read_text(self, target: str) -> Optional[str]:
        """读取文件文本内容。"""
        rel = self._resolve_target(target)
        if not rel:
            return None
        return _read_text(os.path.join(self.root, rel))

    def get_file_metadata(self, target: str) -> Optional[str]:
        """获取文件元数据信息。"""
        rel = self._resolve_target(target)
        if not rel:
            return None

        full_path = os.path.join(self.root, rel)
        if not os.path.exists(full_path):
            return None

        try:
            stat_result = os.stat(full_path)
            mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat_result.st_mtime))
            create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat_result.st_ctime))

            ext = os.path.splitext(rel)[1].lower()
            is_readable = ext.lstrip('.') in READABLE_EXTENSIONS

            metadata = (
                f"文件名: {os.path.basename(rel)}\n"
                f"路径: {rel}\n"
                f"大小: {stat_result.st_size} 字节 ({stat_result.st_size / 1024:.1f} KB)\n"
                f"修改时间: {mod_time}\n"
                f"创建时间: {create_time}\n"
                f"扩展名: {ext}\n"
                f"可读类型: {'是' if is_readable else '否'}"
            )
            return metadata
        except Exception as e:
            return f"获取文件信息失败: {e}"

    def stats(self) -> Dict:
        """获取语料库统计信息。"""
        st = self.index.stats()
        reg_stats = self.registry.get_stats()
        result = {
            "total_files": reg_stats["total_files"],
            "readable": reg_stats["readable_files"],
            "index_docs": st["total_docs"],
            "workspace": self.root,
            "db_path": self.db_path,
        }
        if self.use_faiss_store and self.faiss_store is not None:
            result["faiss_store"] = self.faiss_store.get_stats()
        return result

    def enable_faiss_store(self, name: str = "default") -> bool:
        """启用 FaissVectorStore（延迟初始化）。"""
        try:
            from faiss_store import FaissVectorStore
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
            self.faiss_store = FaissVectorStore(
                dimension=128, chunk_size=500, chunk_overlap=75,
                persist_dir=str(FAISS_DIR),
            )
            self.faiss_store_name = name
            if not self.faiss_store.load(name):
                readable = self.registry.get_all_readable()
                if readable:
                    documents = {}
                    doc_meta = {}
                    for d in readable:
                        abs_path = os.path.join(self.root, d["path"])
                        content = _read_text(abs_path) or ""
                        documents[d["path"]] = content
                        doc_meta[d["path"]] = {"tags": json.loads(d.get("tags", "[]"))}
                    self.faiss_store.build_from_documents(documents, doc_meta)
                    self.faiss_store.save(name)
            self.use_faiss_store = True
            return True
        except ImportError:
            print("[WARNING] faiss_store 模块不可用。")
            return False

    def get_faiss_store_stats(self) -> Optional[Dict]:
        """获取 FaissVectorStore 统计信息。"""
        if self.faiss_store is not None:
            return self.faiss_store.get_stats()
        return None

    def seed_demo(self):
        """首次运行时写入示例语料，保证搜索/检索开箱可用。"""
        if self.registry.count() > 0:
            return
        demo = [
            ("docs/python简介.txt", "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。设计哲学强调代码可读性。"),
            ("docs/强化学习.txt", "强化学习是机器学习分支，智能体通过与环境交互获得奖励信号学习最优策略。Q-learning 与策略梯度是主要方法。"),
            ("config/app.yaml", "server:\n  host: 0.0.0.0\n  port: 8000\nlog:\n  level: info\n"),
            ("config/deep_search.yaml", "deep_search:\n  max_sub_queries: 4\n  top_k: 5\n  time_budget: 60\n"),
            ("code/main.py", "def main():\n    print('hello agent-rl')\n\nif __name__ == '__main__':\n    main()\n"),
        ]
        for path, content in demo:
            self.add_document(path, content)

    def close(self):
        """关闭数据库连接。"""
        if hasattr(self.registry, 'close'):
            self.registry.close()
