import re
from .base import BaseSplitter

class SentenceSplitter(BaseSplitter):
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def split(self, documents):
        chunks = []
        for doc in documents:
            sentences = re.split(r'(?<=[.!?]) +', doc.page_content)
            current = ""

            for s in sentences:
                if len(current) + len(s) < self.chunk_size:
                    current += " " + s
                else:
                    chunks.append(current.strip())
                    current = s

            if current:
                chunks.append(current.strip())

        return chunks