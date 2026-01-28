"""
算子注册装饰器
简化插件开发
"""
from typing import Type

from ability.operators.plugin_registry import PluginRegistry


def register_parser(extension: str):
    """
    注册解析器的装饰器
    
    使用示例:
    @register_parser(".custom")
    class CustomParser(BaseParser):
        def _parse(self, file_path, **kwargs):
            # 自定义解析逻辑
            pass
    """
    def decorator(cls: Type):
        PluginRegistry.register_parser(extension, cls)
        cls._extension = extension  # 保存扩展名供自动发现使用
        return cls
    return decorator


def register_chunker(strategy: str):
    """
    注册切片器的装饰器
    
    使用示例:
    @register_chunker("custom")
    class CustomChunker(BaseChunker):
        def _chunk(self, text, **kwargs):
            # 自定义切片逻辑
            pass
    """
    def decorator(cls: Type):
        PluginRegistry.register_chunker(strategy, cls)
        cls._strategy = strategy
        return cls
    return decorator


def register_retriever(mode: str):
    """
    注册检索器的装饰器
    
    使用示例:
    @register_retriever("custom")
    class CustomRetriever(BaseRetriever):
        def _retrieve(self, query, top_k, tenant_id, **kwargs):
            # 自定义检索逻辑
            pass
    """
    def decorator(cls: Type):
        PluginRegistry.register_retriever(mode, cls)
        cls._mode = mode
        return cls
    return decorator
