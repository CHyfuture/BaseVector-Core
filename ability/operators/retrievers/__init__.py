"""检索器模块"""
# 使用延迟导入避免循环导入问题
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ability.operators.retrievers.base_retriever import BaseRetriever, RetrievalResult
    from ability.operators.retrievers.fulltext_retriever import FullTextRetriever
    from ability.operators.retrievers.hybrid_retriever import HybridRetriever
    from ability.operators.retrievers.keyword_retriever import KeywordRetriever
    from ability.operators.retrievers.phrase_match_retriever import PhraseMatchRetriever
    from ability.operators.retrievers.retriever_factory import RetrieverFactory
    from ability.operators.retrievers.semantic_retriever import SemanticRetriever
    from ability.operators.retrievers.text_match_retriever import TextMatchRetriever

# 实际导入（延迟导入，避免循环依赖）
def __getattr__(name: str):
    if name == "BaseRetriever" or name == "RetrievalResult":
        from ability.operators.retrievers.base_retriever import BaseRetriever, RetrievalResult
        return RetrievalResult if name == "RetrievalResult" else BaseRetriever
    elif name == "SemanticRetriever":
        from ability.operators.retrievers.semantic_retriever import SemanticRetriever
        return SemanticRetriever
    elif name == "KeywordRetriever":
        from ability.operators.retrievers.keyword_retriever import KeywordRetriever
        return KeywordRetriever
    elif name == "HybridRetriever":
        from ability.operators.retrievers.hybrid_retriever import HybridRetriever
        return HybridRetriever
    elif name == "FullTextRetriever":
        from ability.operators.retrievers.fulltext_retriever import FullTextRetriever
        return FullTextRetriever
    elif name == "TextMatchRetriever":
        from ability.operators.retrievers.text_match_retriever import TextMatchRetriever
        return TextMatchRetriever
    elif name == "PhraseMatchRetriever":
        from ability.operators.retrievers.phrase_match_retriever import PhraseMatchRetriever
        return PhraseMatchRetriever
    elif name == "RetrieverFactory":
        from ability.operators.retrievers.retriever_factory import RetrieverFactory
        return RetrieverFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "SemanticRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "FullTextRetriever",
    "TextMatchRetriever",
    "PhraseMatchRetriever",
    "RetrieverFactory",
]
