from typing import Dict, List, Any
from agent_state import RetrievalResult
from file_corpus import FileCorpus


class FactQueryHandler:
    """
    事实查询分支：基于共享语料库的单轮向量检索（FAISS / TF-IDF）。
    """

    def __init__(self, corpus: FileCorpus = None, top_k: int = 5):
        self.corpus = corpus or FileCorpus()
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        docs = self.corpus.search(query, self.top_k)
        retrieval_results = []
        for doc in docs:
            retrieval_results.append({
                "path": doc["path"],
                "content": doc.get("content", ""),
                "title": doc.get("title", ""),
                "source": doc["path"],
                "page": doc.get("page", 1),
                "score": doc["score"],
                "metadata": doc,
            })

        result = RetrievalResult(
            documents=retrieval_results,
            score=retrieval_results[0]["score"] if retrieval_results else 0.0,
            query_used=query,
            round_num=1,
        )

        print(f"  [事实查询] FAISS 检索返回 {len(retrieval_results)} 个文档，最高得分: {result.score:.4f}")
        for doc in retrieval_results[:3]:
            print(f"    - [得分:{doc['score']:.4f}] {doc['path']} | {doc['content'][:50]}")
        if not retrieval_results:
            print("  [事实查询] 语料库中未找到相关内容。")
        return result

    def get_stats(self) -> Dict:
        return self.corpus.stats()


def build_retrieval_result(docs: List[Dict], query: str, round_num: int) -> RetrievalResult:
    """把 FileCorpus.search 结果包装成 RetrievalResult（deep_search 复用）。"""
    items = [{
        "path": d["path"],
        "content": d.get("content", ""),
        "title": d.get("title", ""),
        "source": d["path"],
        "page": d.get("page", 1),
        "score": d["score"],
        "metadata": d,
    } for d in docs]
    return RetrievalResult(
        documents=items,
        score=items[0]["score"] if items else 0.0,
        query_used=query,
        round_num=round_num,
    )