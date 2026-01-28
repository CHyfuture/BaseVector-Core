"""
存储算子基类
所有Milvus存储操作算子都应继承此类
"""
from typing import Any, Dict, Optional

from ability.operators.base import BaseOperator
from ability.storage.milvus_client import milvus_client


class BaseStorageOperator(BaseOperator):
    """
    存储算子基类
    所有Milvus存储操作算子都应继承此类
    
    提供统一的Milvus客户端访问和基本验证
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化存储算子
        
        Args:
            config: 配置字典
                - collection_name: 集合名称（可选，可在process时传入）
                - tenant_id: 租户ID（可选）
        """
        super().__init__(config)
        self.client = milvus_client

    def _initialize(self) -> None:
        """初始化存储算子"""
        # 确保Milvus客户端已连接
        if not self.client.connected:
            self.client.connect()
        super()._initialize()

    def _get_collection_name(self, collection_name: Optional[str] = None, tenant_id: Optional[str] = None) -> str:
        """
        获取集合名称（支持多租户）
        
        Args:
            collection_name: 集合名称（可选，优先级最高）
            tenant_id: 租户ID（可选，从config或参数获取）
            
        Returns:
            完整的集合名称
        """
        from ability.config import get_settings
        
        settings = get_settings()
        
        # 优先使用传入的collection_name
        if collection_name:
            return collection_name
        
        # 使用config中的tenant_id或传入的tenant_id
        final_tenant_id = tenant_id or self.get_config("tenant_id") or settings.DEFAULT_TENANT_ID
        
        # 使用配置模板生成集合名称
        template = self.get_config("collection_template", settings.MILVUS_COLLECTION_TEMPLATE)
        return template.format(tenant_id=final_tenant_id)

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据（子类可以重写）
        
        Args:
            input_data: 输入数据
            
        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            return False
        return True
