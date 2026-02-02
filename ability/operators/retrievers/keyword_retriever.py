"""关键词检索器"""
import re
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


class KeywordRetriever(BaseRetriever):
    """关键词检索器，基于关键词匹配和BM25算法"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_match_count = self.get_config("min_match_count", 1)
        self.tf_normalization_factor = self.get_config("tf_normalization_factor", 100.0)
        self.candidate_multiplier = self.get_config("candidate_multiplier", 10)

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词实现"""
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

    def _calculate_keyword_score(self, query_tokens: List[str], content: str) -> float:
        """计算关键词匹配分数（简单的TF-IDF风格）"""
        content_tokens = self._tokenize(content)
        content_lower = content.lower()

        score = 0.0
        matched_count = 0

        for token in query_tokens:
            token_lower = token.lower()
            # 计算token在内容中出现的频率
            count = content_lower.count(token_lower)
            if count > 0:
                matched_count += 1
                # 使用简单的词频权重
                score += count * (1.0 / (1.0 + len(content_tokens) / self.tf_normalization_factor))

        # 如果匹配的关键词太少，返回0
        if matched_count < self.min_match_count:
            return 0.0

        # 归一化分数
        if len(query_tokens) > 0:
            score = score / len(query_tokens)

        return score

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """关键词检索实现"""
        # 1. 确定集合名称
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

        # 构建关键词过滤表达式（使用LIKE操作符）
        keyword_filters = []
        for token in query_tokens:
            # 转义特殊字符
            escaped_token = token.replace('"', '\\"').replace("'", "\\'")
            # 使用LIKE进行模糊匹配
            keyword_filters.append(f'content like "%{escaped_token}%"')

        if keyword_filters:
            keyword_expr = " || ".join(f"({f})" for f in keyword_filters)
            if expr:
                expr = f"({expr}) && ({keyword_expr})"
            else:
                expr = keyword_expr

        # 4. 尝试使用Milvus查询（如果支持LIKE操作符）
        try:
            collection = milvus_client.get_collection(collection_name)
            collection.load()

            # 使用query方法查询所有匹配的文档（与集合 schema 一致）
            query_results = collection.query(
                expr=expr,
                output_fields=["id", *RETRIEVAL_OUTPUT_FIELDS],
                limit=top_k * self.candidate_multiplier,  # 获取更多候选结果以便排序
            )

            if not query_results:
                return []

            # 5. 对结果进行关键词评分和排序
            scored_results = []
            for result in query_results:
                content = result.get("content", "")
                if not content:
                    continue

                score = self._calculate_keyword_score(query_tokens, content)

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
            logger.warning(f"Keyword retrieval failed for collection {collection_name}: {str(e)}")
            # 如果LIKE操作符不支持，返回空列表
            return []
