"""
存储算子工厂
根据操作类型创建对应的存储算子
"""
from typing import Dict, Optional

from ability.operators.storage.base_storage import BaseStorageOperator
from ability.operators.storage.collection_operator import CollectionOperator
from ability.operators.storage.delete_operator import DeleteOperator
from ability.operators.storage.insert_operator import InsertOperator
from ability.operators.storage.update_operator import UpdateOperator
from ability.utils.logger import logger

# 内置存储算子映射
BUILTIN_STORAGE_REGISTRY: Dict[str, type[BaseStorageOperator]] = {
    "collection": CollectionOperator,
    "insert": InsertOperator,
    "update": UpdateOperator,
    "delete": DeleteOperator,
}


class StorageFactory:
    """存储算子工厂类"""

    @staticmethod
    def create_operator(operation: Optional[str] = None, config: Optional[Dict] = None) -> BaseStorageOperator:
        """
        根据操作类型创建对应的存储算子
        
        Args:
            operation: 操作类型
                - "collection": 集合操作（创建、删除、列出、获取）
                - "insert": 数据插入
                - "update": 数据更新
                - "delete": 数据删除
            config: 算子配置字典
                
        Returns:
            存储算子实例
            
        Raises:
            ValueError: 如果操作类型不支持
        """
        if operation is None:
            raise ValueError("Operation type must be specified")
        
        if operation not in BUILTIN_STORAGE_REGISTRY:
            raise ValueError(
                f"Unsupported storage operation: {operation}. "
                f"Supported operations: {', '.join(BUILTIN_STORAGE_REGISTRY.keys())}"
            )
        
        operator_class = BUILTIN_STORAGE_REGISTRY[operation]
        operator = operator_class(config=config or {})
        operator.initialize()
        
        logger.info(f"Created storage operator {operator_class.__name__} for operation: {operation}")
        return operator

    @staticmethod
    def get_supported_operations() -> list[str]:
        """
        获取所有支持的操作类型
        
        Returns:
            操作类型列表
        """
        return list(BUILTIN_STORAGE_REGISTRY.keys())


# 为便捷使用提供快捷方法
def create_collection_operator(config: Optional[Dict] = None) -> CollectionOperator:
    """创建集合操作算子"""
    return StorageFactory.create_operator("collection", config)


def create_insert_operator(config: Optional[Dict] = None) -> InsertOperator:
    """创建数据插入算子"""
    return StorageFactory.create_operator("insert", config)


def create_update_operator(config: Optional[Dict] = None) -> UpdateOperator:
    """创建数据更新算子"""
    return StorageFactory.create_operator("update", config)


def create_delete_operator(config: Optional[Dict] = None) -> DeleteOperator:
    """创建数据删除算子"""
    return StorageFactory.create_operator("delete", config)
