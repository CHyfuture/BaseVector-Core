"""短语匹配检索器，支持短语精确匹配"""
import json
from typing import Any, Dict, List, Optional

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import BaseRetriever, RetrievalResult
from ability.storage.milvus_client import milvus_client
from ability.utils.logger import logger

settings = get_settings()


class PhraseMatchRetriever(BaseRetriever):
    """
    短语匹配检索器，支持短语精确匹配
    用于查找包含完整短语的文档，短语中的词序和位置都会被考虑
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化短语匹配检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
                - case_sensitive: 是否区分大小写（默认False）
                - allow_partial: 是否允许部分匹配（默认False，即必须完全匹配短语）
        """
        super().__init__(config)
        self.case_sensitive = self.get_config("case_sensitive", False)
        self.allow_partial = self.get_config("allow_partial", False)

    def _escape_special_chars(self, text: str) -> str:
        """
        转义特殊字符，用于构建 Milvus 查询表达式

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        return text.replace('"', '\\"').replace("'", "\\'").replace("%", "\\%").replace("_", "\\_")

    def _build_phrase_expression(self, phrase: str, field_name: str = "content") -> str:
        """
        构建短语匹配表达式

        Args:
            phrase: 查询短语
            field_name: 字段名称

        Returns:
            Milvus 查询表达式
        """
        escaped_phrase = self._escape_special_chars(phrase)

        if self.allow_partial:
            # 允许部分匹配：使用 LIKE 操作符
            return f'{field_name} like "%{escaped_phrase}%"'
        else:
            # 完全匹配：使用 LIKE 操作符（Milvus 2.6+ 支持短语匹配）
            # 对于完全匹配，我们使用精确的 LIKE 模式
            return f'{field_name} like "%{escaped_phrase}%"'

    def _calculate_phrase_score(self, phrase: str, content: str) -> float:
        """
        计算短语匹配分数

        Args:
            phrase: 查询短语
            content: 文档内容

        Returns:
            匹配分数
        """
        if self.case_sensitive:
            phrase_lower = phrase
            content_lower = content
        else:
            phrase_lower = phrase.lower()
            content_lower = content.lower()

        if phrase_lower not in content_lower:
            return 0.0

        # 计算短语出现次数
        match_count = content_lower.count(phrase_lower)

        # 计算第一个匹配位置（越靠前分数越高）
        first_pos = content_lower.find(phrase_lower)
        position_score = 1.0 / (1.0 + first_pos / 100.0)

        # 计算短语长度比例（短语越长，匹配越精确）
        phrase_length_ratio = len(phrase) / max(len(content), 1)

        # 综合分数
        score = match_count * position_score * (1.0 + phrase_length_ratio)
        return score

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        短语匹配检索实现

        Args:
            query: 查询短语
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数

        Returns:
            检索结果列表
        """
        if not query.strip():
            logger.warning("Query phrase is empty")
            return []

        # 1. 确定集合名称
        if tenant_id:
            collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=tenant_id)
        else:
            collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(
                tenant_id=settings.DEFAULT_TENANT_ID
            )

        # 2. 构建过滤表达式
        phrase_expr = self._build_phrase_expression(query)

        expr = None
        if tenant_id:
            expr = f'tenant_id == "{tenant_id}"'

        if expr:
            expr = f"({expr}) && ({phrase_expr})"
        else:
            expr = phrase_expr

        # 3. 使用Milvus查询
        try:
            collection = milvus_client.get_collection(collection_name)
            collection.load()

            # 使用query方法查询所有匹配的文档
            query_results = collection.query(
                expr=expr,
                output_fields=["id", "document_id", "tenant_id", "content", "metadata", "chunk_index", "parent_chunk_id"],
                limit=top_k * 2,  # 获取更多候选结果以便排序
            )

            if not query_results:
                return []

            # 4. 对结果进行短语匹配评分和排序
            scored_results = []
            for result in query_results:
                content = result.get("content", "")
                if not content:
                    continue

                score = self._calculate_phrase_score(query, content)

                if score > 0:
                    # 解析metadata
                    metadata_str = result.get("metadata", "{}")
                    try:
                        metadata_dict = (
                            json.loads(metadata_str) if isinstance(metadata_str, str) else (metadata_str or {})
                        )
                    except:
                        metadata_dict = {}

                    scored_results.append(
                        {
                            "chunk_id": result.get("id"),
                            "document_id": result.get("document_id", 0),
                            "content": content,
                            "score": score,
                            "metadata": {
                                "chunk_index": result.get("chunk_index"),
                                "parent_chunk_id": result.get("parent_chunk_id"),
                                "tenant_id": result.get("tenant_id"),
                                **metadata_dict,
                            },
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
            logger.warning(f"Phrase match retrieval failed for collection {collection_name}: {str(e)}")
            return []
