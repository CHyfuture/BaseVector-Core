"""全文检索器，基于 Milvus LIKE 操作符实现全文检索"""
import re
from typing import Any, Dict, List, Optional

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import (
    BaseRetriever,
    RetrievalResult,
    metadata_from_result,
    resolve_output_fields,
)
from ability.storage.milvus_client import milvus_client
from ability.utils.logger import logger

settings = get_settings()


class FullTextRetriever(BaseRetriever):
    """
    全文检索器，基于 Milvus LIKE 操作符实现全文检索
    支持对查询文本进行分词，然后使用 LIKE 操作符进行模糊匹配
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化全文检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
                - min_match_count: 最少匹配的关键词数量（默认1）
                - candidate_multiplier: 候选结果倍数（默认10）
                - match_mode: 匹配模式，"or"（任一关键词匹配）或 "and"（所有关键词匹配），默认"or"
        """
        super().__init__(config)
        self.min_match_count = self.get_config("min_match_count", 1)
        self.candidate_multiplier = self.get_config("candidate_multiplier", 10)
        self.match_mode = self.get_config("match_mode", "or")  # "or" 或 "and"

    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词

        Args:
            text: 输入文本

        Returns:
            分词后的token列表
        """
        # 中文字符单独作为token
        chinese_pattern = r"[\u4e00-\u9fff]+"
        # 英文单词作为token
        english_pattern = r"[a-zA-Z0-9]+"

        tokens = []
        # 提取中文
        for match in re.finditer(chinese_pattern, text):
            tokens.append(match.group())
        # 提取英文单词
        for match in re.finditer(english_pattern, text):
            tokens.append(match.group().lower())

        return tokens

    def _escape_special_chars(self, text: str) -> str:
        """
        转义特殊字符，用于构建 Milvus 查询表达式

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        # 转义 Milvus 表达式中的特殊字符
        return text.replace('"', '\\"').replace("'", "\\'").replace("%", "\\%").replace("_", "\\_")

    def _calculate_fulltext_score(self, query_tokens: List[str], content: str) -> float:
        """
        计算全文检索分数（基于关键词匹配频率）

        Args:
            query_tokens: 查询关键词列表
            content: 文档内容

        Returns:
            匹配分数
        """
        content_lower = content.lower()
        matched_count = 0
        total_matches = 0

        for token in query_tokens:
            token_lower = token.lower()
            count = content_lower.count(token_lower)
            if count > 0:
                matched_count += 1
                total_matches += count

        # 如果匹配的关键词太少，返回0
        if matched_count < self.min_match_count:
            return 0.0

        # 计算分数：匹配的关键词比例 * 总匹配次数
        match_ratio = matched_count / len(query_tokens) if query_tokens else 0.0
        score = match_ratio * (1.0 + total_matches / max(len(query_tokens), 1))

        return score

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        全文检索实现

        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数

        Returns:
            检索结果列表
        """
        # 1. 确定集合名称：调用方传入优先，否则按模板 + tenant_id
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            if tenant_id:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=tenant_id)
            else:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(
                    tenant_id=settings.DEFAULT_TENANT_ID
                )

        # 2. 对查询进行分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("Query has no valid tokens after tokenization")
            return []

        # 3. 构建过滤表达式
        expr = None

        # 3.1 先接入上游传入的 milvus_expr（例如 'chunk_type == "child"'）
        user_expr = kwargs.get("milvus_expr")
        if user_expr:
            expr = str(user_expr)

        # 构建全文检索过滤表达式（使用LIKE操作符）
        keyword_filters = []
        for token in query_tokens:
            escaped_token = self._escape_special_chars(token)
            # 使用LIKE进行模糊匹配
            keyword_filters.append(f'content like "%{escaped_token}%"')

        if keyword_filters:
            # 根据匹配模式构建表达式
            if self.match_mode.lower() == "and":
                # AND模式：所有关键词都必须匹配
                keyword_expr = " && ".join(f"({f})" for f in keyword_filters)
            else:
                # OR模式：任一关键词匹配即可
                keyword_expr = " || ".join(f"({f})" for f in keyword_filters)

            if expr:
                # 同时满足上游 milvus_expr 与全文关键词匹配条件
                expr = f"({expr}) && ({keyword_expr})"
            else:
                expr = keyword_expr

        # 4. 使用Milvus查询
        try:
            output_fields = resolve_output_fields(
                collection_name,
                kwargs.get("output_fields"),
            )
            collection = milvus_client.get_collection(collection_name)
            collection.load()

            # 使用query方法查询所有匹配的文档（与集合 schema 一致，动态 output_fields）
            query_results = collection.query(
                expr=expr,
                output_fields=["id", *output_fields],
                limit=top_k * self.candidate_multiplier,  # 获取更多候选结果以便排序
            )

            if not query_results:
                return []

            # 5. 对结果进行全文检索评分和排序
            scored_results = []
            for result in query_results:
                content = result.get("content", "")
                if not content:
                    continue

                score = self._calculate_fulltext_score(query_tokens, content)

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

            # 6. 构建检索结果
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
            logger.warning(f"Full-text retrieval failed for collection {collection_name}: {str(e)}")
            return []
