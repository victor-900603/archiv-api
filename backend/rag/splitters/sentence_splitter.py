import re
from uuid import uuid4
from langchain_core.documents import Document
from .base import BaseSplitter

class SentenceSplitter(BaseSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata

            sentences = re.split(r"(?<=[.!?]) +", text)

            current_sentences = []
            current_length = 0
            chunk_index = 0

            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                if current_length + len(s) <= self.chunk_size:
                    current_sentences.append(s)
                    current_length += len(s)
                else:
                    chunk_text = " ".join(current_sentences)

                    chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                **metadata,
                                "splitter": "sentence",
                                "chunk_index": chunk_index,
                                "chunk_id": str(uuid4()),
                            },
                        )
                    )

                    chunk_index += 1

                    if self.chunk_overlap > 0:
                        overlap_sentences = []
                        overlap_length = 0

                        for prev_s in reversed(current_sentences):
                            if overlap_length + len(prev_s) > self.chunk_overlap:
                                break
                            overlap_sentences.insert(0, prev_s)
                            overlap_length += len(prev_s)

                        current_sentences = overlap_sentences
                        current_length = overlap_length
                    else:
                        current_sentences = []
                        current_length = 0

                    current_sentences.append(s)
                    current_length += len(s)

            if current_sentences:
                chunk_text = " ".join(current_sentences)

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **metadata,
                            "splitter": "sentence",
                            "chunk_index": chunk_index,
                            "chunk_id": str(uuid4()),
                        },
                    )
                )

        return chunks
    
    
if __name__ == "__main__":
    from backend.rag.loaders.text_loader import TextLoader
    loader = TextLoader(r"backend\docs\example\example.txt")
    docs = loader.load()
    
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split(docs)
    
    for chunk in chunks:
        print(chunk.metadata)