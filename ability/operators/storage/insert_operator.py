"""
数据插入算子
支持批量插入数据到Milvus集合
"""
from typing import Any, List

from ability.operators.storage.base_storage import BaseStorageOperator


class InsertOperator(BaseStorageOperator):
    """
    数据插入算子
    
    支持批量插入数据到Milvus集合，数据必须符合集合的schema定义
    """

    def process(self, input_data: Any, **kwargs) -> List[int]:
        """
        插入数据到Milvus集合
        
        Args:
            input_data: 要插入的数据列表，每个元素是一个字典，必须包含集合schema中定义的所有字段
                - 向量字段：通常是 "vector"（List[float]）
                - 文本字段：通常是 "content"（str）
                - 其他元数据字段：根据schema定义
            **kwargs: 额外参数
                - collection_name: 集合名称（可选，从config或tenant_id生成）
                - tenant_id: 租户ID（可选）
                
        Returns:
            插入的ID列表
            
        Raises:
            ValueError: 如果数据格式不正确或缺少必需字段
        """
        if not isinstance(input_data, list):
            raise ValueError(f"Input data must be a list, got {type(input_data)}")
        
        if not input_data:
            self.logger.warning("Input data is empty, nothing to insert")
            return []
        
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        self.logger.info(f"Inserting {len(input_data)} entities into collection: {collection_name}")
        
        try:
            ids = self.client.insert(collection_name=collection_name, data=input_data)
            self.logger.info(f"Successfully inserted {len(ids)} entities")
            return ids
        except Exception as e:
            self.logger.error(f"Failed to insert data: {str(e)}", exc_info=True)
            raise

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据（应该是列表）
            
        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            return False
        
        if not isinstance(input_data, list):
            self.logger.error(f"Input data must be a list, got {type(input_data)}")
            return False
        
        if len(input_data) == 0:
            self.logger.warning("Input data list is empty")
            return False
        
        # 检查每个元素是否是字典
        for i, item in enumerate(input_data):
            if not isinstance(item, dict):
                self.logger.error(f"Item at index {i} must be a dict, got {type(item)}")
                return False
        
        return True
