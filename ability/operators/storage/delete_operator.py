"""
数据删除算子
支持通过表达式或ID列表删除Milvus集合中的数据
"""
from typing import Any, Dict, List, Union

from ability.operators.storage.base_storage import BaseStorageOperator


class DeleteOperator(BaseStorageOperator):
    """
    数据删除算子
    
    支持通过以下方式删除数据：
    - 表达式删除：使用Milvus过滤表达式（如 "id in [1, 2, 3]"）
    - ID列表删除：传入ID列表，自动构建表达式
    """

    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        删除数据
        
        Args:
            input_data: 
                - 如果是字符串：作为过滤表达式
                - 如果是列表：作为要删除的ID列表
                - 如果是None：从kwargs获取expr或ids
            **kwargs: 额外参数
                - collection_name: 集合名称（可选，从config或tenant_id生成）
                - tenant_id: 租户ID（可选）
                - expr: 过滤表达式（可选，优先级低于input_data）
                - ids: ID列表（可选，优先级低于input_data）
                
        Returns:
            删除结果字典：
            - deleted_count: 删除的记录数（估算）
            - expr: 使用的表达式
        """
        collection_name = kwargs.get("collection_name")
        tenant_id = kwargs.get("tenant_id")
        collection_name = self._get_collection_name(collection_name, tenant_id)
        
        # 确定删除表达式
        expr = None
        
        if isinstance(input_data, str):
            # input_data是表达式字符串
            expr = input_data
        elif isinstance(input_data, list):
            # input_data是ID列表
            expr = self._build_expr_from_ids(input_data)
        else:
            # 从kwargs获取
            if "expr" in kwargs:
                expr = kwargs["expr"]
            elif "ids" in kwargs:
                expr = self._build_expr_from_ids(kwargs["ids"])
        
        if not expr:
            raise ValueError("Either 'expr' or 'ids' must be provided for delete operation")
        
        self.logger.info(f"Deleting data from collection: {collection_name}, expr: {expr}")
        
        try:
            collection = self.client.get_collection(collection_name)
            collection.load()
            
            # 执行删除
            collection.delete(expr)
            collection.flush()
            
            self.logger.info(f"Successfully deleted data matching expr: {expr}")
            
            # 注意：Milvus的delete操作不直接返回删除数量
            # 这里返回表达式，调用方可以通过查询估算删除数量
            return {
                "deleted_count": None,  # Milvus不直接返回删除数量
                "expr": expr,
            }
        except Exception as e:
            self.logger.error(f"Failed to delete data: {str(e)}", exc_info=True)
            raise

    def _build_expr_from_ids(self, ids: List[Union[int, str]]) -> str:
        """
        从ID列表构建过滤表达式
        
        Args:
            ids: ID列表
            
        Returns:
            过滤表达式字符串
        """
        if not ids:
            raise ValueError("ID list cannot be empty")
        
        # 判断ID类型
        if isinstance(ids[0], str):
            # VARCHAR类型ID
            id_list_str = ", ".join([f'"{id_val}"' for id_val in ids])
            return f"id in [{id_list_str}]"
        else:
            # INT64类型ID
            id_list_str = ", ".join([str(id_val) for id_val in ids])
            return f"id in [{id_list_str}]"

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据（字符串、列表或None）
            
        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            # 如果input_data是None，允许从kwargs获取expr或ids
            return True
        
        if input_data is not None:
            if not isinstance(input_data, (str, list)):
                self.logger.error(
                    f"Input data must be str (expr), list (ids), or None, got {type(input_data)}"
                )
                return False
            
            if isinstance(input_data, list) and len(input_data) == 0:
                self.logger.error("ID list cannot be empty")
                return False
        
        return True
