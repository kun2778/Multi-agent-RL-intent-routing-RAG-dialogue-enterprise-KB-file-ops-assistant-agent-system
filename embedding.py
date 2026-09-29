"""
统一 Embedding 模块：优先使用 sentence-transformers，不可用时回退到 TF-IDF。

所有 Embedder 实现统一接口：
  - fit(texts: List[str]) -> None
  - transform(text: str) -> np.ndarray  (shape: (dimension,))
  - fit_transform(texts: List[str]) -> np.ndarray  (shape: (n, dimension))
  - is_fitted: bool
  - dimension: int
"""

import hashlib
import os
import re
from collections import Counter
from typing import List, Dict, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# SentenceTransformer Embedder
# ──────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的语义嵌入器。

    推荐模型:
      - BAAI/bge-small-zh-v1.5  (512维, ~90MB, 中文优化)
      - shibing624/text2vec-base-chinese (768维, ~400MB)
      - sentence-transformers/all-MiniLM-L6-v2 (384维, 英文为主)
    """

    # 指纹版本：向量构造方式一旦变化必须 +1，否则调用方的缓存指纹不变，
    # 会把「旧代码算出的向量」当成有效缓存继续用（回归集结果就和 CI 对不上）。
    schema_version = "st-v1"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = None, dimension: int = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请运行: pip install sentence-transformers"
            )
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.is_fitted = True  # 预训练模型无需 fit
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, texts: List[str]):
        """空操作，保持接口兼容。"""
        pass

    def transform(self, text: str) -> np.ndarray:
        """将单条文本编码为向量。"""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """批量编码文本。"""
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# TF-IDF Embedder (回退方案)
# ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """中英文混合分词。"""
    tokens = re.findall(r'[A-Za-z0-9_]+', text.lower())
    cjk = re.findall(r'[一-鿿]', text)
    tokens.extend(cjk)
    tokens.extend([''.join(cjk[i:i + 2]) for i in range(len(cjk) - 1)])
    return tokens


def _stable_hash_vector(text: str, dimension: int) -> np.ndarray:
    """由文本内容确定性派生的伪随机向量。

    替代 ``np.random.randn``：后者每次调用结果不同，导致同一查询在不同
    进程 / 不同次运行得到不同向量，回归评估无法复现。
    这里用 sha256 流当 PRNG，跨进程、跨机器、跨 Python 版本都一致。
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    need = dimension * 4
    buf = bytearray()
    counter = 0
    while len(buf) < need:
        buf += hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        counter += 1
    raw = np.frombuffer(bytes(buf[:need]), dtype=np.uint8).astype(np.float32)
    vec = raw / 127.5 - 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class TfidfEmbedder:
    """TF-IDF 嵌入器，作为 sentence-transformers 不可用时的回退方案。"""

    # 见 SentenceTransformerEmbedder.schema_version 的说明。
    # v2：vocab 改为 sorted() 编号 + OOV 改用 _stable_hash_vector（跨进程确定性）。
    schema_version = "tfidf-v2"

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
        total = max(len(texts), 1)
        for token, count in doc_counts.items():
            self.idf[token] = np.log((total + 1) / (count + 1)) + 1
        # 排序后再编号：set 的迭代顺序依赖 PYTHONHASHSEED（随进程变化），
        # 会让同一语料在不同进程产出不同的 vocab → 不同的向量 → 不同的召回。
        self.vocab = {t: i for i, t in enumerate(sorted(set(all_tokens)))}
        self.is_fitted = True

    def transform(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            return _stable_hash_vector(text, self.dimension)
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
            # 全 OOV（查询里没有任何已知 token）：给确定性非零向量，
            # 既避免所有 OOV 查询坍缩成同一个点，也保证可复现
            vector = (_stable_hash_vector(text, self.dimension) * 0.01
                      ).astype(np.float32)
        return vector

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return np.array([self.transform(t) for t in texts], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────

def _check_hf_reachable(timeout: int = 3) -> bool:
    """快速检查 HuggingFace 是否可达。"""
    import socket
    try:
        sock = socket.create_connection(("huggingface.co", 443), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, OSError):
        return False


def create_embedder(
    backend: str = "auto",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    dimension: int = None,
    device: str = None,
):
    """创建 Embedder 实例。

    Args:
        backend: "sentence_transformers" | "tfidf" | "auto"
                 auto 模式下优先尝试 sentence-transformers，不可用则回退 TF-IDF
        model_name: sentence-transformers 模型名
        dimension: 向量维度（None 则使用模型默认值或 TF-IDF 默认128）
        device: 推理设备 ("cpu", "cuda", None=自动)

    Returns:
        具备 fit/transform/fit_transform 接口的 Embedder 实例
    """
    if backend == "sentence_transformers" or (
        backend == "auto" and SENTENCE_TRANSFORMERS_AVAILABLE
    ):
        if backend == "auto" and not _check_hf_reachable():
            print("[Embedding] HuggingFace 不可达，跳过 sentence-transformers，使用 TF-IDF")
        else:
            try:
                return SentenceTransformerEmbedder(
                    model_name=model_name,
                    device=device,
                    dimension=dimension,
                )
            except Exception as e:
                print(f"[Embedding] sentence-transformers 加载失败: {e}，回退到 TF-IDF")

    return TfidfEmbedder(dimension=dimension or 128)
