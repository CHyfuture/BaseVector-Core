"""
插件注册机制
支持动态注册自定义算子
"""
from typing import Dict, Type
from pathlib import Path
import importlib.util
import inspect

from ability.operators.base import BaseOperator
from ability.operators.parsers.base_parser import BaseParser
from ability.operators.chunkers.base_chunker import BaseChunker
from ability.operators.retrievers.base_retriever import BaseRetriever
from ability.operators.storage.base_storage import BaseStorageOperator
from ability.utils.logger import logger


class PluginRegistry:
    """
    插件注册器
    支持动态注册和加载自定义算子
    """
    
    # 注册表（支持运行时扩展）
    _parsers: Dict[str, Type[BaseParser]] = {}
    _chunkers: Dict[str, Type[BaseChunker]] = {}
    _retrievers: Dict[str, Type[BaseRetriever]] = {}
    _storage_operators: Dict[str, Type[BaseStorageOperator]] = {}
    
    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[BaseParser]) -> None:
        """
        注册解析器
        
        Args:
            extension: 文件扩展名（如 ".pdf", ".custom"）
            parser_class: 解析器类（必须继承BaseParser）
            
        Raises:
            ValueError: 如果parser_class不是BaseParser的子类
        """
        if not issubclass(parser_class, BaseParser):
            raise ValueError(f"{parser_class} must inherit from BaseParser")
        
        extension = extension.lower()
        cls._parsers[extension] = parser_class
        logger.info(f"Registered parser: {extension} -> {parser_class.__name__}")
    
    @classmethod
    def register_chunker(cls, strategy: str, chunker_class: Type[BaseChunker]) -> None:
        """
        注册切片器
        
        Args:
            strategy: 切片策略名称（如 "fixed", "custom"）
            chunker_class: 切片器类（必须继承BaseChunker）
        """
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError(f"{chunker_class} must inherit from BaseChunker")
        
        cls._chunkers[strategy] = chunker_class
        logger.info(f"Registered chunker: {strategy} -> {chunker_class.__name__}")
    
    @classmethod
    def register_retriever(cls, mode: str, retriever_class: Type[BaseRetriever]) -> None:
        """
        注册检索器
        
        Args:
            mode: 检索模式（如 "semantic", "custom"）
            retriever_class: 检索器类（必须继承BaseRetriever）
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise ValueError(f"{retriever_class} must inherit from BaseRetriever")
        
        cls._retrievers[mode] = retriever_class
        logger.info(f"Registered retriever: {mode} -> {retriever_class.__name__}")
    
    @classmethod
    def register_storage_operator(cls, operation: str, operator_class: Type[BaseStorageOperator]) -> None:
        """
        注册存储算子
        
        Args:
            operation: 操作类型（如 "collection", "insert", "custom"）
            operator_class: 存储算子类（必须继承BaseStorageOperator）
        """
        if not issubclass(operator_class, BaseStorageOperator):
            raise ValueError(f"{operator_class} must inherit from BaseStorageOperator")
        
        cls._storage_operators[operation] = operator_class
        logger.info(f"Registered storage operator: {operation} -> {operator_class.__name__}")
    
    @classmethod
    def load_plugin(cls, plugin_path: Path) -> None:
        """
        从文件加载插件
        
        Args:
            plugin_path: 插件文件路径
            
        示例:
            PluginRegistry.load_plugin(Path("plugins/custom_parser.py"))
        """
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {plugin_path}")
        
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create spec for {plugin_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找所有BaseOperator子类并自动注册
            registered = False
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseOperator) and 
                    obj != BaseOperator and
                    obj.__module__ == module.__name__):
                    
                    # 检查是否有注册属性
                    if hasattr(obj, "_extension"):
                        cls.register_parser(obj._extension, obj)
                        registered = True
                    elif hasattr(obj, "_strategy"):
                        cls.register_chunker(obj._strategy, obj)
                        registered = True
                    elif hasattr(obj, "_mode"):
                        cls.register_retriever(obj._mode, obj)
                        registered = True
                    elif hasattr(obj, "_operation"):
                        cls.register_storage_operator(obj._operation, obj)
                        registered = True
            
            if registered:
                logger.info(f"Loaded plugin from {plugin_path}")
            else:
                logger.warning(f"No operators found in plugin {plugin_path}")
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_parsers(cls) -> Dict[str, Type[BaseParser]]:
        """获取所有注册的解析器（包括内置和插件）"""
        return cls._parsers.copy()
    
    @classmethod
    def get_chunkers(cls) -> Dict[str, Type[BaseChunker]]:
        """获取所有注册的切片器"""
        return cls._chunkers.copy()
    
    @classmethod
    def get_retrievers(cls) -> Dict[str, Type[BaseRetriever]]:
        """获取所有注册的检索器"""
        return cls._retrievers.copy()
    
    @classmethod
    def get_storage_operators(cls) -> Dict[str, Type[BaseStorageOperator]]:
        """获取所有注册的存储算子"""
        return cls._storage_operators.copy()
    
    @classmethod
    def unregister_parser(cls, extension: str) -> bool:
        """
        取消注册解析器
        
        Args:
            extension: 文件扩展名
            
        Returns:
            是否成功取消注册
        """
        extension = extension.lower()
        if extension in cls._parsers:
            del cls._parsers[extension]
            logger.info(f"Unregistered parser: {extension}")
            return True
        return False
    
    @classmethod
    def unregister_chunker(cls, strategy: str) -> bool:
        """取消注册切片器"""
        if strategy in cls._chunkers:
            del cls._chunkers[strategy]
            logger.info(f"Unregistered chunker: {strategy}")
            return True
        return False
    
    @classmethod
    def unregister_retriever(cls, mode: str) -> bool:
        """取消注册检索器"""
        if mode in cls._retrievers:
            del cls._retrievers[mode]
            logger.info(f"Unregistered retriever: {mode}")
            return True
        return False
    
    @classmethod
    def unregister_storage_operator(cls, operation: str) -> bool:
        """取消注册存储算子"""
        if operation in cls._storage_operators:
            del cls._storage_operators[operation]
            logger.info(f"Unregistered storage operator: {operation}")
            return True
        return False
    
    @classmethod
    def clear_all(cls) -> None:
        """清空所有注册表（谨慎使用）"""
        cls._parsers.clear()
        cls._chunkers.clear()
        cls._retrievers.clear()
        cls._storage_operators.clear()
        logger.warning("All plugin registries cleared")
