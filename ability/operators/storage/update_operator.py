"""
数据更新算子
支持更新Milvus集合中的数据
"""
from typing import Any, Dict

from ability.operators.storage.base_storage import BaseStorageOperator


class UpdateOperator(BaseStorageOperator):
    """
    数据更新算子
    
    支持通过表达式更新Milvus集合中的数据
    Milvus通过delete + insert的方式实现更新，即先删除旧数据，再插入新数据
    """

    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        更新数据
        
        Args:
            input_data: 更新的数据列表（每个元素是字典，包含要更新的字段）
            **kwargs: 额外参数
                - collection_name: 集合名称（可选，从config或tenant_id生成）
                - tenant_id: 租户ID（可选）
                - expr: 过滤表达式，用于确定要更新的数据（如 "id in [1, 2, 3]"）
                - new_data: 新数据列表（如果input_data不是列表，从kwargs获取）
                
        Returns:
            更新结果字典：
            - deleted_count: 删除的记录数
            - inserted_count: 插入的记录数
            - inserted_ids: 新插入的ID列表
            
        Note:
            Milvus的更新操作实际上是通过delete + insert实现的。
            如果需要更新部分字段，请先查询原数据，合并字段后重新插入。
        """
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        # 获取更新的数据
        if isinstance(input_data, list):
            new_data = input_data
        else:
            new_data = kwargs.get("new_data", [])
        
        if not new_data:
            raise ValueError("No data provided for update")
        
        expr = kwargs.get("expr")
        if not expr:
            raise ValueError("Expression (expr) is required for update operation")
        
        self.logger.info(f"Updating data in collection: {collection_name}, expr: {expr}")
        
        # Milvus更新：先删除，再插入
        try:
            # 1. 删除旧数据
            deleted_count = self._delete_by_expr(collection_name, expr)
            
            # 2. 插入新数据
            inserted_ids = self.client.insert(collection_name=collection_name, data=new_data)
            inserted_count = len(inserted_ids)
            
            self.logger.info(
                f"Update completed: deleted {deleted_count} entities, "
                f"inserted {inserted_count} entities"
            )
            
            return {
                "deleted_count": deleted_count,
                "inserted_count": inserted_count,
                "inserted_ids": inserted_ids,
            }
        except Exception as e:
            self.logger.error(f"Failed to update data: {str(e)}", exc_info=True)
            raise

    def _delete_by_expr(self, collection_name: str, expr: str) -> int:
        """
        通过表达式删除数据（内部方法）
        
        Args:
            collection_name: 集合名称
            expr: 过滤表达式
            
        Returns:
            删除的记录数（估算）
        """
        try:
            collection = self.client.get_collection(collection_name)
            collection.load()
            
            # 先查询要删除的数据数量（用于估算）
            query_results = collection.query(expr=expr, output_fields=["id"], limit=1)
            if not query_results:
                return 0
            
            # 执行删除
            collection.delete(expr)
            collection.flush()
            
            self.logger.info(f"Deleted data matching expr: {expr}")
            
            # 注意：Milvus的delete不直接返回删除数量
            # 这里返回1表示删除操作已执行，实际删除数量需要调用方通过查询估算
            return 1
            
        except Exception as e:
            self.logger.error(f"Failed to delete by expr: {str(e)}", exc_info=True)
            raise

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据（应该是列表或None，如果为None则从kwargs获取）
            
        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            # 如果input_data是None，允许从kwargs获取new_data
            return True
        
        if input_data is not None and not isinstance(input_data, list):
            self.logger.error(f"Input data must be a list or None, got {type(input_data)}")
            return False
        
        return True
