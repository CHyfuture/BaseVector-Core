"""
检索器工厂
"""
from typing import Dict, Optional

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import BaseRetriever
from ability.operators.retrievers.fulltext_retriever import FullTextRetriever
from ability.operators.retrievers.hybrid_retriever import HybridRetriever
from ability.operators.retrievers.keyword_retriever import KeywordRetriever
from ability.operators.retrievers.phrase_match_retriever import PhraseMatchRetriever
from ability.operators.retrievers.semantic_retriever import SemanticRetriever
from ability.operators.retrievers.text_match_retriever import TextMatchRetriever
from ability.operators.plugin_registry import PluginRegistry
from ability.utils.logger import logger

settings = get_settings()

# 内置检索器映射
BUILTIN_RETRIEVER_REGISTRY: Dict[str, type[BaseRetriever]] = {
    "semantic": SemanticRetriever,
    "keyword": KeywordRetriever,
    "hybrid": HybridRetriever,
    "fulltext": FullTextRetriever,
    "text_match": TextMatchRetriever,
    "phrase_match": PhraseMatchRetriever,
}


class RetrieverFactory:
    """检索器工厂类"""

    @staticmethod
    def create_retriever(mode: Optional[str] = None, config: Optional[Dict] = None) -> BaseRetriever:
        """
        根据模式创建对应的检索器

        Args:
            mode: 检索模式
                - semantic: 语义检索（向量相似度）
                - keyword: 关键词检索（BM25风格）
                - hybrid: 混合检索（语义+关键词融合）
                - fulltext: 全文检索（基于LIKE操作符）
                - text_match: 文本匹配（精确/模糊匹配）
                - phrase_match: 短语匹配（短语精确匹配）
            config: 检索器配置

        Returns:
            检索器实例

        Raises:
            ValueError: 如果模式不支持
        """
        if mode is None:
            mode = settings.SEARCH_MODE

        # 合并内置和插件注册表
        all_retrievers = {**BUILTIN_RETRIEVER_REGISTRY, **PluginRegistry.get_retrievers()}

        if mode not in all_retrievers:
            raise ValueError(
                f"Unsupported search mode: {mode}. "
                f"Supported modes: {', '.join(all_retrievers.keys())}"
            )

        # 合并默认配置
        default_config = {
            "top_k": settings.TOP_K,
        }
        if config:
            default_config.update(config)

        retriever_class = all_retrievers[mode]
        retriever = retriever_class(config=default_config)
        retriever.initialize()

        logger.info(f"Created retriever {retriever_class.__name__} with mode: {mode}")
        return retriever

    @staticmethod
    def get_supported_modes() -> list[str]:
        """
        获取所有支持的检索模式

        Returns:
            模式列表
        """
        all_retrievers = {**BUILTIN_RETRIEVER_REGISTRY, **PluginRegistry.get_retrievers()}
        return list(all_retrievers.keys())
