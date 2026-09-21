"""
FAISS 向量存储模块：支持大块文本分片、持久化索引、约 5 万块规模。

配置参数：
  - chunk_size: 500 字符
  - chunk_overlap: 75 字符（15% 重叠）
  - dimension: 128（默认 TF-IDF）或自定义 embedding 维度
  - index_type: "flat" (IndexFlatIP) / "ivf" (IndexIVFFlat)
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss-cpu not installed. FaissVectorStore will use numpy fallback.")


# ──────────────────────────────────────────────────────────────────────
# 文本分块器
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """单个文本块的结构。"""
    text: str
    source: str          # 来源文件路径
    chunk_index: int     # 该文件内的块序号
    start_offset: int    # 在原文中的起始偏移
    end_offset: int      # 在原文中的结束偏移
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": self.metadata,
        }


class TextChunker:
    """文本分块器：按固定大小 + 重叠窗口切分。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 75):
        """
        Args:
            chunk_size: 每个块的最大字符数（默认 500）
            overlap: 相邻块的重叠字符数（默认 75，即 15%）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str = "",
                   metadata: Optional[Dict] = None) -> List[TextChunk]:
        """将文本切分为多个块。

        策略：
          1. 优先在段落边界（\\n\\n）切分
          2. 段落过长则在句子边界（.!?。！？）切分
          3. 句子仍过长则按字符硬切
        """
        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        meta = metadata or {}

        # 按段落预分割
        paragraphs = re.split(r'\n\s*\n', text)
        buffer = ""
        chunk_idx = 0
        offset = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果加入当前段落不超过 chunk_size，合并
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            # buffer 已满，先输出 buffer
            if buffer:
                chunks.append(TextChunk(
                    text=buffer, source=source, chunk_index=chunk_idx,
                    start_offset=offset, end_offset=offset + len(buffer),
                    metadata=meta,
                ))
                chunk_idx += 1
                offset += len(buffer) - self.overlap
                buffer = ""

            # 当前段落本身超长，需要子切分
            if len(para) > self.chunk_size:
                sub_chunks = self._split_long_paragraph(para, source, chunk_idx, offset, meta)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                offset += len(para) - self.overlap
            else:
                buffer = para

        # 最后一块
        if buffer:
            chunks.append(TextChunk(
                text=buffer, source=source, chunk_index=chunk_idx,
                start_offset=offset, end_offset=offset + len(buffer),
                metadata=meta,
            ))

        return chunks

    def _split_long_paragraph(self, text: str, source: str, start_idx: int,
                               start_offset: int, meta: Dict) -> List[TextChunk]:
        """对超长段落按句子边界 → 硬切方式分块。"""
        chunks = []
        # 按句子分割
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        buffer = ""
        offset = start_offset
        idx = start_idx

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = (buffer + " " + sent).strip() if buffer else sent
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(TextChunk(
                    text=buffer, source=source, chunk_index=idx,
                    start_offset=offset, end_offset=offset + len(buffer),
                    metadata=meta,
                ))
                idx += 1
                offset += len(buffer) - self.overlap
                buffer = ""

            # 句子本身超长 → 按字符硬切
            if len(sent) > self.chunk_size:
                for i in range(0, len(sent), self.chunk_size - self.overlap):
                    piece = sent[i:i + self.chunk_size]
                    chunks.append(TextChunk(
                        text=piece, source=source, chunk_index=idx,
                        start_offset=offset, end_offset=offset + len(piece),
                        metadata=meta,
                    ))
                    idx += 1
                    offset += len(piece) - self.overlap
            else:
                buffer = sent

        if buffer:
            chunks.append(TextChunk(
                text=buffer, source=source, chunk_index=idx,
                start_offset=offset, end_offset=offset + len(buffer),
                metadata=meta,
            ))

        return chunks

    def chunk_documents(self, documents: Dict[str, str],
                        doc_metadata: Optional[Dict[str, Dict]] = None) -> List[TextChunk]:
        """批量切分多个文档。

        Args:
            documents: {file_path: text_content}
            doc_metadata: {file_path: {key: value}} 可选的文档级元数据
        """
        all_chunks = []
        for path, content in documents.items():
            meta = (doc_metadata or {}).get(path, {})
            chunks = self.chunk_text(content, source=path, metadata=meta)
            all_chunks.extend(chunks)
        return all_chunks


# ──────────────────────────────────────────────────────────────────────
# 简易 TF-IDF 嵌入器（与现有 TfidfEmbedder 兼容）
# ──────────────────────────────────────────────────────────────────────

class SimpleTfidfEmbedder:
    """轻量 TF-IDF 嵌入器，用于向量化文本块。"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[A-Za-z0-9_]+', text.lower())
        cjk = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(cjk)
        tokens.extend([''.join(cjk[i:i + 2]) for i in range(len(cjk) - 1)])
        return tokens

    def fit(self, texts: List[str]):
        from collections import Counter
        doc_counts = Counter()
        all_tokens: List[str] = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            for t in set(tokens):
                doc_counts[t] += 1
        total = max(len(texts), 1)
        for token, count in doc_counts.items():
            self.idf[token] = np.log((total + 1) / (count + 1)) + 1
        self.vocab = {t: i for i, t in enumerate(set(all_tokens))}
        self.is_fitted = True

    def transform(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            return np.random.randn(self.dimension).astype(np.float32)
        tokens = self._tokenize(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        from collections import Counter
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

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return np.array([self.transform(t) for t in texts], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# FAISS 向量存储
# ──────────────────────────────────────────────────────────────────────

class FaissVectorStore:
    """基于 FAISS 的向量存储，支持 ~5 万块规模。

    特性：
      - 文本分块（chunk_size=500, overlap=75）
      - FAISS IndexFlatIP（精确搜索）或 IndexIVFFlat（近似搜索）
      - 索引 + 元数据持久化
      - 增量添加 / 批量构建
    """

    def __init__(
        self,
        dimension: int = 128,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        index_type: str = "flat",
        persist_dir: str = "faiss_store",
    ):
        """
        Args:
            dimension: 向量维度
            chunk_size: 文本块大小（字符）
            chunk_overlap: 块间重叠字符
            index_type: "flat" (IndexFlatIP) 或 "ivf" (IndexIVFFlat, 需训练)
            persist_dir: 索引持久化目录
        """
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_type = index_type
        self.persist_dir = os.path.abspath(persist_dir)

        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = SimpleTfidfEmbedder(dimension=dimension)

        self.index: Optional[faiss.Index] = None if FAISS_AVAILABLE else None
        self.chunks: List[Dict] = []  # 存储块的元数据
        self.is_trained = False

        os.makedirs(self.persist_dir, exist_ok=True)

    # ──────────── 索引构建 ────────────

    def _create_index(self, n_vectors: int) -> None:
        """根据规模自动选择索引类型。"""
        if not FAISS_AVAILABLE:
            return

        if self.index_type == "ivf" and n_vectors > 1000:
            # IVF 需要聚类训练，nlist 取 sqrt(n)
            nlist = max(int(np.sqrt(n_vectors)), 1)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

    def build_from_documents(self, documents: Dict[str, str],
                             doc_metadata: Optional[Dict[str, Dict]] = None) -> int:
        """从文档字典构建索引。

        Args:
            documents: {file_path: text_content}
            doc_metadata: {file_path: {key: value}}

        Returns:
            总块数
        """
        all_chunks = self.chunker.chunk_documents(documents, doc_metadata)
        if not all_chunks:
            print("[FaissVectorStore] 无可索引的文本块。")
            return 0

        print(f"[FaissVectorStore] 分块完成: {len(all_chunks)} 个块 "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        # 提取文本并嵌入
        texts = [c.text for c in all_chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.fit_transform(texts)

        # 创建索引
        self._create_index(len(all_chunks))

        # 训练 IVF 索引（如需要）
        if FAISS_AVAILABLE and self.index_type == "ivf" and not self.is_trained:
            if hasattr(self.index, 'train'):
                self.index.train(vectors)
            self.is_trained = True

        # 添加向量
        if FAISS_AVAILABLE and self.index is not None:
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self.index.add(vecs)

        # 存储元数据
        self.chunks = [c.to_dict() for c in all_chunks]

        stats = self.get_stats()
        print(f"[FaissVectorStore] 索引构建完成: {stats}")
        return len(all_chunks)

    def build_from_corpus(self, corpus_root: str = "workspace") -> int:
        """从 workspace 目录构建索引。"""
        documents = {}
        for dirpath, _, filenames in os.walk(corpus_root):
            # 跳过隐藏目录
            if any(p.startswith('.') for p in dirpath.split(os.sep)):
                continue
            for name in filenames:
                if name.startswith('.') or name.endswith('.json'):
                    continue
                abs_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(abs_path, corpus_root)
                ext = os.path.splitext(name)[1].lstrip('.').lower()
                readable_exts = {
                    'txt', 'md', 'yaml', 'yml', 'py', 'json', 'xml', 'ini', 'cfg',
                    'env', 'csv', 'html', 'js', 'ts', 'c', 'cpp', 'h', 'java', 'go',
                    'rs', 'sh', 'bat', 'ps1', 'sql', 'log', 'toml',
                }
                if ext not in readable_exts:
                    continue
                try:
                    with open(abs_path, 'rb') as f:
                        raw = f.read()
                    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
                        try:
                            text = raw.decode(enc)
                            documents[rel_path] = text
                            break
                        except (UnicodeDecodeError, ValueError):
                            continue
                except OSError:
                    continue

        return self.build_from_documents(documents)

    # ──────────── 检索 ────────────

    def search(self, query: str, top_k: int = 5, filter_noise: bool = True) -> List[Dict]:
        """向量检索，返回得分倒序结果。支持噪声过滤。"""
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return self._numpy_fallback_search(query, top_k)

        qv = self.embedder.transform(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(qv)

        # 检索更多结果以便过滤
        k = min(top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(qv, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            source = chunk["source"]
            
            # 噪声过滤：跳过日志文件、临时文件等
            if filter_noise and self._is_noise_file(source):
                # 如果是噪声文件但得分很高，仍然保留但降低优先级
                if score > 0.7:
                    score *= 0.3  # 大幅降低噪声文件的得分
                else:
                    continue  # 跳过低分噪声文件
            
            results.append({
                "text": chunk["text"],
                "source": source,
                "chunk_index": chunk["chunk_index"],
                "start_offset": chunk["start_offset"],
                "end_offset": chunk["end_offset"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            })
        
        # 按得分排序并返回前top_k个
        results.sort(key=lambda x: x["score"], reverse=True)
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

    def _numpy_fallback_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """FAISS 不可用时的 numpy 回退。"""
        if not self.chunks:
            return []
        qv = self.embedder.transform(query)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        # 这里需要重新计算向量，简化处理
        return []

    def search_with_filter(self, query: str, top_k: int = 5,
                           source_filter: Optional[str] = None) -> List[Dict]:
        """带过滤条件的检索。"""
        results = self.search(query, top_k=top_k * 3)  # 多取一些再过滤
        if source_filter:
            results = [r for r in results if source_filter in r["source"]]
        return results[:top_k]

    # ──────────── 增量更新 ────────────

    def add_document(self, path: str, content: str,
                     metadata: Optional[Dict] = None) -> int:
        """增量添加单个文档。"""
        chunks = self.chunker.chunk_text(content, source=path, metadata=metadata)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        vectors = np.array([self.embedder.transform(t) for t in texts], dtype=np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self.index.add(vecs)

        self.chunks.extend([c.to_dict() for c in chunks])
        return len(chunks)

    def remove_document(self, path: str) -> int:
        """移除指定来源的所有块（需要重建索引）。"""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["source"] != path]
        removed = before - len(self.chunks)
        if removed > 0:
            self._rebuild_index_from_chunks()
        return removed

    def _rebuild_index_from_chunks(self):
        """从现有 chunks 重建索引。"""
        if not self.chunks:
            if self.index is not None:
                self.index.reset()
            return

        texts = [c["text"] for c in self.chunks]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        vectors = np.array([self.embedder.transform(t) for t in texts], dtype=np.float32)

        self._create_index(len(self.chunks))
        if FAISS_AVAILABLE and self.index is not None:
            if self.index_type == "ivf" and hasattr(self.index, 'train'):
                self.index.train(vectors)
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self.index.add(vecs)

    # ──────────── 持久化 ────────────

    def save(self, name: str = "default") -> str:
        """保存索引和元数据到磁盘。"""
        store_dir = os.path.join(self.persist_dir, name)
        os.makedirs(store_dir, exist_ok=True)

        # 保存 FAISS 索引
        if FAISS_AVAILABLE and self.index is not None:
            index_path = os.path.join(store_dir, "index.faiss")
            faiss.write_index(self.index, index_path)

        # 保存元数据
        meta_path = os.path.join(store_dir, "chunks.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunks,
                "config": {
                    "dimension": self.dimension,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "index_type": self.index_type,
                },
                "embedder_vocab": self.embedder.vocab,
                "embedder_idf": self.embedder.idf,
                "is_trained": self.is_trained,
            }, f, ensure_ascii=False, indent=2)

        path = os.path.abspath(store_dir)
        print(f"[FaissVectorStore] 已保存到: {path} ({len(self.chunks)} 块)")
        return path

    def load(self, name: str = "default") -> bool:
        """从磁盘加载索引。"""
        store_dir = os.path.join(self.persist_dir, name)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "chunks.json")

        if not os.path.exists(meta_path):
            print(f"[FaissVectorStore] 未找到索引: {meta_path}")
            return False

        # 加载元数据
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.is_trained = data.get("is_trained", False)
        self.embedder.vocab = data.get("embedder_vocab", {})
        self.embedder.idf = data.get("embedder_idf", {})
        self.embedder.is_fitted = bool(self.embedder.vocab)

        # 加载 FAISS 索引
        if FAISS_AVAILABLE and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"[FaissVectorStore] 已加载索引: {len(self.chunks)} 块, "
                  f"FAISS ntotal={self.index.ntotal}")
        else:
            print(f"[FaissVectorStore] 已加载元数据: {len(self.chunks)} 块 (FAISS 索引缺失)")

        return True

    # ──────────── 统计 ────────────

    def get_stats(self) -> Dict:
        """返回存储统计信息。"""
        ntotal = 0
        if FAISS_AVAILABLE and self.index is not None:
            ntotal = self.index.ntotal

        sources = set(c["source"] for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "faiss_ntotal": ntotal,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "unique_sources": len(sources),
            "index_type": self.index_type,
            "persist_dir": self.persist_dir,
        }

    def get_source_stats(self) -> Dict[str, int]:
        """返回每个来源的块数。"""
        stats: Dict[str, int] = {}
        for c in self.chunks:
            src = c["source"]
            stats[src] = stats.get(src, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))


# ──────────────────────────────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────────────────────────────

def create_faiss_store(
    dimension: int = 128,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    persist_dir: str = "faiss_store",
) -> FaissVectorStore:
    """创建并返回一个新的 FaissVectorStore 实例。"""
    return FaissVectorStore(
        dimension=dimension,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_dir=persist_dir,
    )


def load_or_create_faiss_store(
    name: str = "default",
    dimension: int = 128,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    persist_dir: str = "faiss_store",
) -> FaissVectorStore:
    """加载已有索引，不存在则创建新实例。"""
    store = FaissVectorStore(
        dimension=dimension,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_dir=persist_dir,
    )
    if not store.load(name):
        print(f"[FaissVectorStore] 未找到已有索引 '{name}'，将创建新索引。")
    return store
