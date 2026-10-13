"""检索式意图识别：从标注意图语料库召回相似例句，作为端到端分类的第四路投票信号。

设计:
  - 语料: datasets/intent_corpus.csv + 项目根 intent_corpus.csv +
         datasets/intent_corpus_bulk.csv（30 万条批量生成语料，自动去重合并）
  - 向量: embedding.create_embedder() 优先 sentence-transformers (bge-small-zh)，
         不可用时回退 TF-IDF（与文档检索共用同一 Embedder 工厂）
  - 索引: FAISS IndexFlatIP（可用时）/ numpy 暴力余弦回退
  - 持久化: .runtime/cache/intent_index/ 下缓存向量+标签，语料/嵌入器变化时自动重建
  - 接口: retrieve(query) -> List[hit]; vote(query) -> (IntentType, conf, evidence)
  - 增量: add_example(text, label) 支持在线补充标注例句（纠正后写入）

与端到端分类的关系:
  不替代 ClassificationHead / Q 网络 / LLM / 规则 —— 在 classify_intent
  的置信度仲裁中作为第 4 路候选参与加权投票，低相似度时自动弃权。
"""

import csv
import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent_state import IntentType
from atomic_io import atomic_savez, atomic_write_json
from embedding import create_embedder
from rwlock import RWLock
from runtime_paths import CACHE_DIR

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# 默认配置（可用 config/app.yaml intent_retrieval 段覆盖）
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIM = 0.30          # 低于该相似度弃权，避免 OOD 乱投
DEFAULT_MAX_SAMPLES = 8000      # 语料上限（超过则均匀抽样，控制首建耗时）
DEFAULT_PERSIST = True


def _load_yaml_intent_cfg() -> Dict:
    try:
        from providers.config_loader import get_config_loader
        app = get_config_loader().load_app_config() or {}
        return app.get("intent_retrieval") or {}
    except Exception:
        return {}


class IntentRetriever:
    """标注意图语料的向量检索器：retrieve 返回相似例句，vote 聚合为意图投票。"""

    def __init__(
        self,
        csv_paths: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIM,
        persist_dir: Optional[str] = None,
        persist: bool = DEFAULT_PERSIST,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ):
        cfg = _load_yaml_intent_cfg()
        self.top_k = int(cfg.get("top_k", top_k))
        self.min_similarity = float(cfg.get("min_similarity", min_similarity))
        self.max_samples = int(cfg.get("max_samples", max_samples))
        self.persist = bool(cfg.get("persist", persist))
        self.enabled = bool(cfg.get("enabled", True))

        base = os.path.dirname(os.path.abspath(__file__))
        if csv_paths is None:
            csv_paths = [
                os.path.join(base, "intent_corpus.csv"),
                os.path.join(base, "datasets", "intent_corpus.csv"),
                os.path.join(base, "datasets", "intent_corpus_bulk.csv"),
            ]
        self.csv_paths = [p for p in csv_paths if os.path.exists(p)]

        self.persist_dir = persist_dir or os.path.join(str(CACHE_DIR), "intent_index")

        self.embedder = None
        self.texts: List[str] = []
        self.labels: List[str] = []
        self.vectors: Optional[np.ndarray] = None   # (N, dim) L2-normalized
        self.index = None                           # faiss.Index or None (numpy fallback)
        self.dim: int = 0

        # 读写锁：检索是读多写少（每条请求都要查），只有 add_example/重建是写。
        # 用普通互斥锁会让并发检索互相排队。
        self._lock = RWLock()
        self._ready = False
        self._build_error: Optional[str] = None
        self.last_evidence: Optional[str] = None
        self.last_hit: Optional[Dict] = None

        if self.enabled:
            self._ensure_ready()

    # ────────────────────────────────────────────── 就绪 / 构建
    @property
    def ready(self) -> bool:
        return self._ready

    def _ensure_ready(self):
        if self._ready or not self.enabled:
            return
        with self._lock.write():  # 构建索引期间独占
            if self._ready:
                return
            try:
                t0 = time.time()
                texts, labels = self._load_corpus()
                if not texts:
                    self._build_error = "语料为空"
                    return

                cache_ok = False
                if self.persist:
                    cache_ok = self._try_load_cache(texts, labels)

                if not cache_ok:
                    self._build_from_corpus(texts, labels)
                    if self.persist:
                        self._save_cache()

                self._ready = True
                self._build_error = None
                backend = "FAISS" if (FAISS_AVAILABLE and self.index is not None) else "numpy"
                print(
                    f"  [意图检索] 就绪: {len(self.texts)} 条例句, dim={self.dim}, "
                    f"索引={backend}, 嵌入={type(self.embedder).__name__} "
                    f"({time.time() - t0:.1f}s)"
                )
            except Exception as e:
                self._build_error = str(e)
                self._ready = False
                print(f"  [意图检索] 构建失败({e})，本路投票停用。")

    # ────────────────────────────────────────────── 语料加载
    def _load_corpus(self) -> Tuple[List[str], List[str]]:
        seen = set()
        texts: List[str] = []
        labels: List[str] = []
        valid_values = {it.value for it in IntentType}

        for path in self.csv_paths:
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = (row.get("text") or "").strip()
                        label = (row.get("label") or "").strip()
                        if not text or not label or label not in valid_values:
                            continue
                        key = (text, label)
                        if key in seen:
                            continue
                        seen.add(key)
                        texts.append(text)
                        labels.append(label)
            except Exception as e:
                print(f"  [意图检索] 读取 {path} 失败: {e}")

        # 超量时按标签均匀抽样，保持类别覆盖
        if len(texts) > self.max_samples:
            texts, labels = self._balanced_subsample(texts, labels, self.max_samples)
        return texts, labels

    @staticmethod
    def _balanced_subsample(texts, labels, max_n):
        from collections import defaultdict
        buckets = defaultdict(list)
        for t, l in zip(texts, labels):
            buckets[l].append((t, l))
        per = max(1, max_n // max(len(buckets), 1))
        out_t, out_l = [], []
        for l in sorted(buckets.keys()):
            items = buckets[l][:per]
            out_t.extend(t for t, _ in items)
            out_l.extend(l for _, l in items)
        # 若仍超量，截断
        return out_t[:max_n], out_l[:max_n]

    # ────────────────────────────────────────────── 构建索引
    def _corpus_fingerprint(self) -> str:
        h = hashlib.md5()
        for p in sorted(self.csv_paths):
            try:
                st = os.stat(p)
                h.update(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime}".encode())
            except OSError:
                h.update(p.encode())
        h.update(type(self.embedder).__name__.encode())
        # 向量构造逻辑变了但语料没变时，必须让指纹失效：
        # 否则会继续用旧代码算出的向量，本地结果与 CI（冷启动重建）不一致。
        h.update(str(getattr(self.embedder, "schema_version", "")).encode())
        h.update(str(getattr(self.embedder, "dimension", 0)).encode())
        h.update(str(self.max_samples).encode())
        return h.hexdigest()

    def _build_from_corpus(self, texts: List[str], labels: List[str]):
        self.embedder = create_embedder(backend="auto")
        raw = self.embedder.fit_transform(texts)
        vecs = np.asarray(raw, dtype=np.float32)
        # L2 归一化 → 内积 = 余弦
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        self.texts = texts
        self.labels = labels
        self.vectors = vecs
        self.dim = vecs.shape[1]

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(np.ascontiguousarray(vecs))
        else:
            self.index = None

    # ────────────────────────────────────────────── 持久化
    def _cache_paths(self) -> Tuple[str, str]:
        os.makedirs(self.persist_dir, exist_ok=True)
        return (
            os.path.join(self.persist_dir, "intent_vectors.npz"),
            os.path.join(self.persist_dir, "intent_meta.json"),
        )

    def _try_load_cache(self, texts: List[str], labels: List[str]) -> bool:
        npz_path, meta_path = self._cache_paths()
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            return False
        try:
            self.embedder = create_embedder(backend="auto")
            want_fp = self._corpus_fingerprint()
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != want_fp:
                return False

            data = np.load(npz_path, allow_pickle=False)
            vecs = data["vectors"].astype(np.float32)
            cached_labels = [str(x) for x in data["labels"].tolist()]
            cached_texts = [str(x) for x in data["texts"].tolist()]

            if vecs.shape[0] != len(cached_labels) or vecs.shape[0] != len(cached_texts):
                return False
            if vecs.shape[0] != len(texts) or cached_texts != texts or cached_labels != labels:
                return False

            # 恢复 TF-IDF 词表（否则 transform 会返回随机向量）
            if hasattr(self.embedder, "vocab"):
                self.embedder.vocab = meta.get("embedder_vocab") or {}
                self.embedder.idf = meta.get("embedder_idf") or {}
                # json 键均为 str；idf 值还原为 float
                self.embedder.idf = {k: float(v) for k, v in self.embedder.idf.items()}
                self.embedder.is_fitted = bool(self.embedder.vocab)
            if not getattr(self.embedder, "is_fitted", True):
                self.embedder.fit(cached_texts)

            self.texts = cached_texts
            self.labels = cached_labels
            self.vectors = vecs
            self.dim = vecs.shape[1]
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(np.ascontiguousarray(vecs))
            else:
                self.index = None
            return True
        except Exception as e:
            print(f"  [意图检索] 缓存加载失败({e})，重建索引。")
            return False

    def _save_cache(self):
        if self.vectors is None:
            return
        try:
            npz_path, meta_path = self._cache_paths()
            # strings 用 unicode 数组（非 object），避免 allow_pickle 限制
            atomic_savez(
                npz_path,
                vectors=self.vectors,
                labels=np.array(self.labels, dtype=np.str_),
                texts=np.array(self.texts, dtype=np.str_),
            )
            meta = {
                "fingerprint": self._corpus_fingerprint(),
                "count": len(self.texts),
                "dim": self.dim,
                "embedder": type(self.embedder).__name__,
                "top_k": self.top_k,
                "min_similarity": self.min_similarity,
                "saved_at": time.time(),
            }
            # TF-IDF 词表随缓存持久化，否则二次启动 transform 返回随机向量
            if hasattr(self.embedder, "vocab"):
                meta["embedder_vocab"] = self.embedder.vocab
                meta["embedder_idf"] = self.embedder.idf
            atomic_write_json(meta_path, meta)
        except Exception as e:
            print(f"  [意图检索] 缓存保存失败({e})（不影响本次运行）。")

    # ────────────────────────────────────────────── 检索
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """返回相似例句列表 [{text,label,similarity,index}]，按相似度降序。"""
        if not self.enabled:
            return []
        self._ensure_ready()
        if not self._ready or not query or not query.strip():
            return []

        k = top_k or self.top_k
        with self._lock.read():   # 共享读：并发检索互不阻塞
            return self._search(query, k)

    def _search(self, query: str, k: int) -> List[Dict]:
        try:
            qv = np.asarray(self.embedder.transform(query), dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(qv)
            if norm > 0:
                qv = qv / norm

            if FAISS_AVAILABLE and self.index is not None:
                k = min(k, self.index.ntotal)
                sims, idxs = self.index.search(np.ascontiguousarray(qv), k)
                hits = []
                for sim, idx in zip(sims[0], idxs[0]):
                    if idx < 0:
                        continue
                    hits.append({
                        "text": self.texts[idx],
                        "label": self.labels[idx],
                        "similarity": float(sim),
                        "index": int(idx),
                    })
                return hits

            # numpy 回退
            sims = (self.vectors @ qv[0])
            order = np.argsort(-sims)[:k]
            return [{
                "text": self.texts[i],
                "label": self.labels[i],
                "similarity": float(sims[i]),
                "index": int(i),
            } for i in order]
        except Exception as e:
            print(f"  [意图检索] 查询失败: {e}")
            return []

    # ────────────────────────────────────────────── 投票
    def vote(self, query: str) -> Tuple[Optional[IntentType], float, Optional[str]]:
        """聚合 top-k 相似例句为一次意图投票。

        Returns:
            (intent, confidence, evidence) —— 无有效相似例句时 intent=None 弃权。
            confidence ∈ [0,1]：最高相似度 × 标签一致性（加权占比）。
        """
        self.last_evidence = None
        self.last_hit = None

        hits = self.retrieve(query)
        if not hits:
            return None, 0.0, None

        max_sim = hits[0]["similarity"]
        if max_sim < self.min_similarity:
            return None, 0.0, None

        # 近逐字命中：语料行标签即权威。
        # 否则按 top-k 求和聚合时，同一模板的近重复聚类（如 3×0.60）
        # 会以「条数和」压过逐字匹配（1×1.0），把精确命中投成别的标签。
        top = hits[0]
        if top["similarity"] >= 0.98:
            try:
                intent = IntentType(top["label"])
            except ValueError:
                intent = None
            if intent is not None:
                confidence = float(max(0.0, min(1.0, top["similarity"])))
                self.last_hit = top
                self.last_evidence = (
                    f"[{top['label']}] {top['text']} "
                    f"(sim={top['similarity']:.3f})"
                )
                return intent, confidence, self.last_evidence

        # 按标签累加相似度（相似度加权投票）
        score_by_label: Dict[str, float] = {}
        for h in hits:
            s = max(0.0, h["similarity"])
            score_by_label[h["label"]] = score_by_label.get(h["label"], 0.0) + s

        total = sum(score_by_label.values())
        if total <= 0:
            return None, 0.0, None

        best_label = max(score_by_label, key=score_by_label.get)
        agreement = score_by_label[best_label] / total          # 一致性 ∈ (0,1]
        confidence = float(max_sim * agreement)                 # 综合置信度
        confidence = max(0.0, min(1.0, confidence))

        try:
            intent = IntentType(best_label)
        except ValueError:
            return None, 0.0, None

        # 证据：命中该标签且相似度最高的一条例句
        evidence_hit = next((h for h in hits if h["label"] == best_label), hits[0])
        self.last_hit = evidence_hit
        self.last_evidence = (
            f"[{evidence_hit['label']}] {evidence_hit['text']} "
            f"(sim={evidence_hit['similarity']:.3f})"
        )
        return intent, confidence, self.last_evidence

    # ────────────────────────────────────────────── 增量补充
    def add_example(self, text: str, label: str, persist: Optional[bool] = None) -> bool:
        """在线补充一条标注例句（用户纠正/主动学习后调用）。"""
        text = (text or "").strip()
        label = (label or "").strip()
        if not text or label not in {it.value for it in IntentType}:
            return False
        if not self._ready:
            self._ensure_ready()
        if not self._ready or self.vectors is None:
            return False

        with self._lock.write():  # 追加向量 + 落盘：独占
            if (text, label) in set(zip(self.texts, self.labels)):
                return False
            try:
                qv = np.asarray(self.embedder.transform(text), dtype=np.float32).reshape(1, -1)
                norm = np.linalg.norm(qv)
                if norm > 0:
                    qv = qv / norm
                if qv.shape[1] != self.dim:
                    return False
                self.vectors = np.vstack([self.vectors, qv])
                self.texts.append(text)
                self.labels.append(label)
                if FAISS_AVAILABLE and self.index is not None:
                    self.index.add(np.ascontiguousarray(qv))
                do_save = self.persist if persist is None else persist
                if do_save:
                    self._save_cache()
                return True
            except Exception as e:
                print(f"  [意图检索] add_example 失败: {e}")
                return False

    # ────────────────────────────────────────────── 统计
    def stats(self) -> Dict:
        from collections import Counter
        return {
            "enabled": self.enabled,
            "ready": self._ready,
            "count": len(self.texts),
            "dim": self.dim,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "backend": "faiss" if (FAISS_AVAILABLE and self.index is not None) else "numpy",
            "embedder": type(self.embedder).__name__ if self.embedder else None,
            "label_dist": dict(Counter(self.labels)),
            "error": self._build_error,
        }

    def rebuild(self, force: bool = True) -> bool:
        """强制重建索引（语料文件更新后调用）。"""
        if force:
            with self._lock.write():  # 清空状态要独占；_ensure_ready 内部会再取写锁，故先释放
                self._ready = False
                self.vectors = None
                self.index = None
                self.texts = []
                self.labels = []
        self._ensure_ready()
        return self._ready


# ──────────────────────────────────────────────────────────────────────
# 惰性单例
# ──────────────────────────────────────────────────────────────────────
_RETRIEVER: Optional[IntentRetriever] = None
_RETRIEVER_LOCK = threading.Lock()


def get_intent_retriever() -> IntentRetriever:
    """获取全局共享的 IntentRetriever 单例（AgentRL 与 MARL 共用一份索引）。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = IntentRetriever()
    return _RETRIEVER
