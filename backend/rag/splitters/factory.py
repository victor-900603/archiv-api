from .recursive_splitter import RecursiveSplitter
from .sentence_splitter import SentenceSplitter

class SplitterFactory:
    _registry = {
        "pdf": RecursiveSplitter,
        "text": RecursiveSplitter,
    }
    @staticmethod
    def get(document_type: str):
        splitter_class = SplitterFactory._registry.get(document_type)
        if not splitter_class:
            raise ValueError(f"No splitter found for document type: {document_type}")
        return splitter_class()