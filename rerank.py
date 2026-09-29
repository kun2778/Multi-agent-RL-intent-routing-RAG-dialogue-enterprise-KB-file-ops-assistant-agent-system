"""Rerank 重排模块：向量召回之后对候选文档做精排精选。

三级后端（自动降级，任一不可用则回退下一级）：
  1. cross_encoder —— sentence-transformers CrossEncoder（真正的重排模型）
  2. embedding     —— 复用项目 Embedder 的语义相似度打分
  3. lexical       —— 分词重合度词法打分（零依赖回退）

环境变量：
  - RERANK_BACKEND: auto | cross_encoder | embedding | lexical（默认 auto）
  - RERANK_MODEL:   CrossEncoder 模型名（默认 cross-encoder/ms-marco-MiniLM-L-6-v2）
  - RERANK_TOP_CANDIDATES: 召回候选数倍数提示（默认 3）

对外接口：
  - get_reranker() -> Reranker 惰性单例
  - rerank_documents(query, docs, top_k=None) -> List[Dict]
      就地写回 score / rerank_score / rerank_backend 并按重排得分降序，截断 top_k
"""

import math
import os
import re
from typing import Dict, List, Optional

import numpy as np


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _doc_text(doc: Dict) -> str:
    """取出用于打分的文本（内容优先，回退到路径/标题）。"""
    text = doc.get("content") or doc.get("text") or ""
    if not text:
        text = f"{doc.get('path', '')} {doc.get('title', '')}"
    return str(text)


def _tokens(text: str) -> List[str]:
    """中英文混合分词：拉丁词 + CJK 字符 + CJK 二元组。"""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk)
    tokens.extend("".join(cjk[i:i + 2]) for i in range(len(cjk) - 1))
    return tokens


class BaseReranker:
    """重排器基类：统一 rerank(query, documents, top_k) 接口。"""

    name = "base"

    def rerank(self, query: str, documents: List[Dict],
               top_k: Optional[int] = None) -> List[Dict]:
        if not documents:
            return documents
        raw_scores = self._score(query, documents)
        results = []
        for doc, s in zip(documents, raw_scores):
            orig = _clamp01(doc.get("score", 0.0))
            rerank_s = _clamp01(s)
            final = _clamp01(0.75 * rerank_s + 0.25 * orig)
            doc.setdefault("vec_score", doc.get("score", 0.0))
            doc["rerank_score"] = round(rerank_s, 6)
            doc["rerank_backend"] = self.name
            doc["score"] = round(final, 6)
            results.append(doc)
        results.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        return results

    def _score(self, query: str, documents: List[Dict]) -> List[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """基于 CrossEncoder 的重排（需要 sentence-transformers + 模型）。"""

    name = "cross_encoder"

    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or os.environ.get(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
        self.model = CrossEncoder(self.model_name)

    def _score(self, query: str, documents: List[Dict]) -> List[float]:
        pairs = [(query, _doc_text(d)[:800]) for d in documents]
        logits = self.model.predict(pairs)
        logits = np.atleast_1d(np.asarray(logits, dtype=np.float64))
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class EmbeddingReranker(BaseReranker):
    """基于语义嵌入余弦相似度的重排（复用项目 Embedder）。"""

    name = "embedding"

    def __init__(self):
        from embedding import SentenceTransformerEmbedder, create_embedder
        emb = create_embedder(backend="sentence_transformers")
        if not isinstance(emb, SentenceTransformerEmbedder):
            raise ImportError("sentence-transformers Embedder 不可用")
        self.embedder = emb

    def _score(self, query: str, documents: List[Dict]) -> List[float]:
        texts = [_doc_text(d)[:800] for d in documents]
        doc_vecs = self.embedder.fit_transform(texts)
        q_vec = self.embedder.transform(query)
        q = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scores = []
        for v in doc_vecs:
            vn = v / (np.linalg.norm(v) + 1e-9)
            scores.append(float(np.dot(q, vn)))
        return scores


class LexicalReranker(BaseReranker):
    """词法重排（分词重合度），零依赖回退方案。"""

    name = "lexical"

    def _score(self, query: str, documents: List[Dict]) -> List[float]:
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return [0.0] * len(documents)
        scores = []
        for d in documents:
            d_tokens = set(_tokens(_doc_text(d)))
            if not d_tokens:
                scores.append(0.0)
                continue
            inter = len(q_tokens & d_tokens)
            denom = math.sqrt(len(q_tokens) * len(d_tokens))
            scores.append(inter / denom if denom else 0.0)
        return scores


_RERANKER: Optional[BaseReranker] = None


def _build_reranker() -> BaseReranker:
    """按 env 配置尝试构建重排器，失败自动降级。"""
    backend = (os.environ.get("RERANK_BACKEND") or "auto").lower()
    orders = {
        "auto": ["cross_encoder", "embedding", "lexical"],
        "cross_encoder": ["cross_encoder", "embedding", "lexical"],
        "embedding": ["embedding", "lexical"],
        "lexical": ["lexical"],
    }
    order = orders.get(backend, orders["auto"])
    factories = {
        "cross_encoder": lambda: CrossEncoderReranker(),
        "embedding": lambda: EmbeddingReranker(),
        "lexical": lambda: LexicalReranker(),
    }
    for name in order:
        try:
            reranker = factories[name]()
            print(f"[Rerank] 重排后端: {name}")
            return reranker
        except Exception as e:
            print(f"[Rerank] 后端 {name} 不可用（{e}），尝试降级...")
    print("[Rerank] 所有重排后端均不可用，使用词法重排兜底")
    return LexicalReranker()


def get_reranker() -> BaseReranker:
    """获取惰性单例 Reranker。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = _build_reranker()
    return _RERANKER


def rerank_documents(query: str, documents: List[Dict],
                     top_k: Optional[int] = None) -> List[Dict]:
    """对候选文档做重排精选（就地更新 score，返回重排后的列表）。

    Args:
        query: 检索查询
        documents: 候选文档（dict，需含 content/score 等字段）
        top_k: 精选保留数量（None 表示只重排不截断）

    Returns:
        按重排得分降序（并截断）的文档列表
    """
    if not documents:
        return documents
    return get_reranker().rerank(query, documents, top_k=top_k)
