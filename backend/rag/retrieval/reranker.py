from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, docs: list, top_k: int = None):
        top_k = top_k or self.top_k

        if not docs:
            return []

        pairs = [(query, d["text"]) for d in docs]

        scores = self.model.predict(pairs)

        scored_docs = []
        for doc, score in zip(docs, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]