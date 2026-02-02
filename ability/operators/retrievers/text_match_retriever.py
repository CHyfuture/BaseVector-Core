"""文本匹配检索器，支持精确匹配和模糊匹配"""
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


class TextMatchRetriever(BaseRetriever):
    """
    文本匹配检索器，支持精确匹配和模糊匹配
    可以根据配置选择使用精确匹配（==）或模糊匹配（LIKE）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文本匹配检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
                - match_type: 匹配类型，"exact"（精确匹配）或 "fuzzy"（模糊匹配），默认"fuzzy"
                - case_sensitive: 是否区分大小写（默认False）
        """
        super().__init__(config)
        self.match_type = self.get_config("match_type", "fuzzy")  # "exact" 或 "fuzzy"
        self.case_sensitive = self.get_config("case_sensitive", False)

    def _escape_special_chars(self, text: str) -> str:
        """
        转义特殊字符，用于构建 Milvus 查询表达式

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        return text.replace('"', '\\"').replace("'", "\\'").replace("%", "\\%").replace("_", "\\_")

    def _build_match_expression(self, query: str, field_name: str = "content") -> str:
        """
        构建匹配表达式

        Args:
            query: 查询文本
            field_name: 字段名称

        Returns:
            Milvus 查询表达式
        """
        escaped_query = self._escape_special_chars(query)

        if self.match_type == "exact":
            # 精确匹配：使用 == 操作符
            if self.case_sensitive:
                return f'{field_name} == "{escaped_query}"'
            else:
                # Milvus 不直接支持大小写不敏感的精确匹配，使用 LIKE 模拟
                return f'{field_name} like "{escaped_query}"'
        else:
            # 模糊匹配：使用 LIKE 操作符
            return f'{field_name} like "%{escaped_query}%"'

    def _calculate_match_score(self, query: str, content: str) -> float:
        """
        计算匹配分数

        Args:
            query: 查询文本
            content: 文档内容

        Returns:
            匹配分数
        """
        if self.match_type == "exact":
            # 精确匹配：完全匹配时返回1.0，否则返回0.0
            if self.case_sensitive:
                return 1.0 if query == content else 0.0
            else:
                return 1.0 if query.lower() == content.lower() else 0.0
        else:
            # 模糊匹配：计算匹配位置和频率
            query_lower = query.lower()
            content_lower = content.lower()

            if query_lower not in content_lower:
                return 0.0

            # 计算匹配次数
            match_count = content_lower.count(query_lower)
            # 计算匹配位置（越靠前分数越高）
            first_pos = content_lower.find(query_lower)
            position_score = 1.0 / (1.0 + first_pos / 100.0)

            # 综合分数
            score = match_count * position_score
            return score

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        文本匹配检索实现

        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数

        Returns:
            检索结果列表
        """
        if not query.strip():
            logger.warning("Query text is empty")
            return []

        # 1. 确定集合名称：调用方传入优先，否则按模板 + tenant_id
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            if tenant_id:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=tenant_id)
            else:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(
                    tenant_id=settings.DEFAULT_TENANT_ID
                )

        # 2. 构建过滤表达式
        expr = self._build_match_expression(query)

        # 3. 使用Milvus查询
        try:
            collection = milvus_client.get_collection(collection_name)
            collection.load()

            # 使用query方法查询所有匹配的文档（与集合 schema 一致）
            query_results = collection.query(
                expr=expr,
                output_fields=["id", *RETRIEVAL_OUTPUT_FIELDS],
                limit=top_k * 2,  # 获取更多候选结果以便排序
            )

            if not query_results:
                return []

            # 4. 对结果进行评分和排序
            scored_results = []
            for result in query_results:
                content = result.get("content", "")
                if not content:
                    continue

                score = self._calculate_match_score(query, content)

                if score > 0:
                    doc_id = result.get("doc_id") or result.get("document_id", 0)
                    scored_results.append(
                        {
                            "chunk_id": result.get("id"),
                            "document_id": doc_id,
                            "content": content,
                            "score": score,
                            "metadata": metadata_from_result(result),
                        }
                    )

            # 按分数排序并返回top_k
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_results[:top_k]

            # 5. 构建检索结果
            results = []
            for result_data in top_results:
                retrieval_result = RetrievalResult(
                    chunk_id=result_data["chunk_id"],
                    document_id=result_data["document_id"],
                    content=result_data["content"],
                    score=result_data["score"],
                    metadata=result_data["metadata"],
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            logger.warning(f"Text match retrieval failed for collection {collection_name}: {str(e)}")
            return []
