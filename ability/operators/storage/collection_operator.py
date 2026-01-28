"""
集合操作算子
支持创建、删除、列出、获取Milvus集合
"""
from typing import Any, List

from pymilvus import Collection, DataType

from ability.config import get_settings
from ability.operators.storage.base_storage import BaseStorageOperator

settings = get_settings()


class CollectionOperator(BaseStorageOperator):
    """
    集合操作算子
    
    支持的操作：
    - create: 创建集合
    - delete: 删除集合
    - list: 列出所有集合
    - get: 获取集合对象
    - exists: 检查集合是否存在
    """

    def process(self, input_data: Any, **kwargs) -> Any:
        """
        处理集合操作
        
        Args:
            input_data: 操作类型（"create", "delete", "list", "get", "exists"）
            **kwargs: 操作参数
                - operation: 操作类型（如果input_data不是字符串，则从kwargs获取）
                - collection_name: 集合名称
                - tenant_id: 租户ID
                - 其他参数根据操作类型而定
                
        Returns:
            操作结果：
            - create: Collection对象
            - delete: None
            - list: 集合名称列表
            - get: Collection对象
            - exists: bool
        """
        operation = input_data if isinstance(input_data, str) else kwargs.get("operation", "get")
        
        if operation == "create":
            return self._create_collection(**kwargs)
        elif operation == "delete":
            return self._delete_collection(**kwargs)
        elif operation == "list":
            return self._list_collections(**kwargs)
        elif operation == "get":
            return self._get_collection(**kwargs)
        elif operation == "exists":
            return self._exists_collection(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _create_collection(self, **kwargs) -> Collection:
        """
        创建集合
        
        Args:
            collection_name: 集合名称（可选，从config或tenant_id生成）
            tenant_id: 租户ID（可选）
            dimension: 稠密向量维度（必需，如果创建稠密向量字段）
            description: 集合描述
            auto_id: 是否自动生成ID（默认True）
            primary_field: 主键字段名（默认"id"）
            dense_vector_field: 稠密向量字段名（可选，如"vector"）
            sparse_vector_field: 稀疏向量字段名（可选）
            primary_dtype: 主键字段类型（默认INT64）
            metadata_fields: 额外的元数据字段（List[FieldSchema]）
            varchar_max_length: VARCHAR字段最大长度（默认255）
            dense_index_params: 稠密向量索引参数（可选）
            sparse_index_params: 稀疏向量索引参数（可选）
            analyzer_params: 分词器参数（可选，如 "english" 或 "chinese"），用于 BM25 Function
            
        Returns:
            Collection对象
        """
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        dimension = kwargs.get("dimension")
        description = kwargs.get("description", "")
        auto_id = kwargs.get("auto_id", True)
        primary_field = kwargs.get("primary_field", "id")
        dense_vector_field = kwargs.get("dense_vector_field", "vector")
        sparse_vector_field = kwargs.get("sparse_vector_field", None)
        primary_dtype = kwargs.get("primary_dtype", DataType.INT64)
        metadata_fields = kwargs.get("metadata_fields", None)
        varchar_max_length = kwargs.get("varchar_max_length", 255)
        dense_index_params = kwargs.get("dense_index_params", None)
        sparse_index_params = kwargs.get("sparse_index_params", None)
        analyzer_params = kwargs.get("analyzer_params", None)
        
        self.logger.info(f"Creating collection: {collection_name}")
        
        collection = self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            description=description,
            auto_id=auto_id,
            primary_field=primary_field,
            dense_vector_field=dense_vector_field,
            sparse_vector_field=sparse_vector_field,
            primary_dtype=primary_dtype,
            metadata_fields=metadata_fields,
            varchar_max_length=varchar_max_length,
            dense_index_params=dense_index_params,
            sparse_index_params=sparse_index_params,
            analyzer_params=analyzer_params,
        )
        
        return collection

    def _delete_collection(self, **kwargs) -> None:
        """
        删除集合
        
        Args:
            collection_name: 集合名称（可选，从config或tenant_id生成）
            tenant_id: 租户ID（可选）
            
        Returns:
            None
        """
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        self.logger.info(f"Deleting collection: {collection_name}")
        self.client.delete_collection(collection_name)

    def _list_collections(self, **kwargs) -> List[str]:
        """
        列出所有集合
        
        Returns:
            集合名称列表
        """
        self.logger.info("Listing all collections")
        return self.client.list_collections()

    def _get_collection(self, **kwargs) -> Collection:
        """
        获取集合对象
        
        Args:
            collection_name: 集合名称（可选，从config或tenant_id生成）
            tenant_id: 租户ID（可选）
            
        Returns:
            Collection对象
        """
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        self.logger.info(f"Getting collection: {collection_name}")
        return self.client.get_collection(collection_name)

    def _exists_collection(self, **kwargs) -> bool:
        """
        检查集合是否存在
        
        Args:
            collection_name: 集合名称（可选，从config或tenant_id生成）
            tenant_id: 租户ID（可选）
            
        Returns:
            是否存在
        """
        from pymilvus import utility
        
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        return utility.has_collection(collection_name)
