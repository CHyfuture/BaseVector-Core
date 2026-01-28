"""
文档切片器工厂
"""
from typing import Dict, Optional

from ability.config import get_settings
from ability.operators.chunkers.base_chunker import BaseChunker
from ability.operators.chunkers.fixed_chunker import FixedChunker
from ability.operators.chunkers.parent_child_chunker import ParentChildChunker
from ability.operators.chunkers.semantic_chunker import SemanticChunker
from ability.operators.chunkers.title_chunker import TitleChunker
from ability.operators.plugin_registry import PluginRegistry
from ability.utils.logger import logger

settings = get_settings()

# 内置切片器映射
BUILTIN_CHUNKER_REGISTRY: Dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
    "title": TitleChunker,
    "parent_child": ParentChildChunker,
}


class ChunkerFactory:
    """文档切片器工厂类"""

    @staticmethod
    def create_chunker(strategy: Optional[str] = None, config: Optional[Dict] = None) -> BaseChunker:
        """
        根据策略创建对应的切片器

        Args:
            strategy: 切片策略（fixed, semantic, title, parent_child）
            config: 切片器配置

        Returns:
            切片器实例

        Raises:
            ValueError: 如果策略不支持
        """
        if strategy is None:
            strategy = settings.CHUNK_STRATEGY

        # 合并内置和插件注册表
        all_chunkers = {**BUILTIN_CHUNKER_REGISTRY, **PluginRegistry.get_chunkers()}

        if strategy not in all_chunkers:
            raise ValueError(
                f"Unsupported chunking strategy: {strategy}. "
                f"Supported strategies: {', '.join(all_chunkers.keys())}"
            )

        # 合并默认配置
        default_config = {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        }
        if config:
            default_config.update(config)

        chunker_class = all_chunkers[strategy]
        chunker = chunker_class(config=default_config)
        chunker.initialize()

        logger.info(f"Created chunker {chunker_class.__name__} with strategy: {strategy}")
        return chunker

    @staticmethod
    def get_supported_strategies() -> list[str]:
        """
        获取所有支持的切片策略

        Returns:
            策略列表
        """
        all_chunkers = {**BUILTIN_CHUNKER_REGISTRY, **PluginRegistry.get_chunkers()}
        return list(all_chunkers.keys())
