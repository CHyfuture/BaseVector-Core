"""语义检索器，基于Milvus向量检索"""
from typing import Any, Dict, List, Optional

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import (
    RETRIEVAL_OUTPUT_FIELDS,
    BaseRetriever,
    RetrievalResult,
    metadata_from_result,
)
from ability.storage.milvus_client import milvus_client
from ability.utils.logger import logger

settings = get_settings()


class SemanticRetriever(BaseRetriever):
    """
    语义检索器，基于Milvus向量检索
    用户需要自行提供查询向量，通过 query_vector 参数传入
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化语义检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
        """
        super().__init__(config)

    def _initialize(self) -> None:
        """初始化检索器"""
        super()._initialize()  # 调用基类初始化（包括重排序模型）
        self.logger.info("Semantic retriever initialized")

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        语义检索

        Args:
            query: 查询文本（如果提供了 query_vector，此参数将被忽略）
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数
                - query_vector: 查询向量（List[float]），必需参数

        Returns:
            检索结果列表
        """
        # 1. 获取查询向量（用户必须提供）
        query_vector = kwargs.get("query_vector")
        if query_vector is None:
            raise ValueError(
                "query_vector is required for SemanticRetriever. "
                "Please provide the query vector in kwargs, e.g., "
                "retriever.process(query='text', query_vector=[0.1, 0.2, ...])"
            )
        
        anns_field = kwargs.get("anns_field", "vector")

        # 确保向量格式正确
        if isinstance(query_vector, list) and len(query_vector) > 0:
            if isinstance(query_vector[0], list):
                query_vector = query_vector[0]
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # 2. 确定集合名称：调用方传入优先，否则按模板 + tenant_id
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            if tenant_id:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=tenant_id)
            else:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=settings.DEFAULT_TENANT_ID)

        # 3. 构建过滤表达式（仅支持用户传入 milvus_expr）
        expr = None
        # 支持用户传入 milvus_expr（会在Service层做校验）
        user_expr = kwargs.get("milvus_expr")
        if user_expr:
            if expr:
                expr = f"({expr}) && ({user_expr})"
            else:
                expr = str(user_expr)

        # 在Milvus中检索
        try:
            search_results = milvus_client.search(
                collection_name=collection_name,
                vectors=[query_vector],
                top_k=top_k,
                expr=expr,
                output_fields=RETRIEVAL_OUTPUT_FIELDS,
                anns_field=anns_field,
            )
        except ValueError as e:
            logger.warning(f"Collection {collection_name} does not exist: {str(e)}")
            return []

        # 构建检索结果（与集合 schema 一致）
        results = []
        for result in search_results:
            chunk_id = result.get("id")
            if not chunk_id:
                continue

            content = result.get("content")
            if content is None:
                content = ""

            document_id = result.get("doc_id") or result.get("document_id")
            if document_id is None:
                document_id = 0

            score = result.get("score")
            if score is None:
                score = 0.0

            retrieval_result = RetrievalResult(
                chunk_id=chunk_id,
                document_id=document_id,
                content=content,
                score=score,
                metadata=metadata_from_result(result),
            )
            results.append(retrieval_result)

        return results
