"""
检索器基类
"""
from typing import Any, Dict, List, Optional

from ability.config import get_settings
from ability.operators.base import BaseOperator

settings = get_settings()


class RetrievalResult:
    """检索结果数据类"""

    def __init__(
        self,
        chunk_id: int,
        document_id: int,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化检索结果

        Args:
            chunk_id: 块ID
            document_id: 文档ID
            content: 块内容
            score: 相似度分数
            metadata: 元数据
        """
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self):
        return f"RetrievalResult(chunk_id={self.chunk_id}, score={self.score:.4f})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class BaseRetriever(BaseOperator):
    """
    检索器基类
    所有检索器都应继承此类
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
                - rerank_enabled: 是否启用重排序（默认使用配置中的RERANK_ENABLED）
                - rerank_model_name: 重排序模型名称（默认使用配置中的RERANK_MODEL_NAME）
                - similarity_threshold: 相似度阈值（默认使用配置中的SIMILARITY_THRESHOLD）
        """
        super().__init__(config)
        self.top_k = self.get_config("top_k", 10)
        self.rerank_enabled = self.get_config("rerank_enabled", settings.RERANK_ENABLED)
        self.rerank_model_name = self.get_config("rerank_model_name", settings.RERANK_MODEL_NAME)
        self.similarity_threshold = self.get_config("similarity_threshold", settings.SIMILARITY_THRESHOLD)
        self.reranker = None

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据（查询文本）

        Args:
            input_data: 查询文本

        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            return False

        if not isinstance(input_data, str):
            self.logger.error(f"Input must be a string, got {type(input_data)}")
            return False

        if not input_data.strip():
            self.logger.error("Query text is empty")
            return False

        return True

    def _initialize(self) -> None:
        """初始化检索器（包括重排序模型）"""
        super()._initialize()  # 调用基类的初始化方法
        # 如果启用重排序，初始化重排序模型
        if self.rerank_enabled:
            try:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(self.rerank_model_name)
                self.logger.info(f"Reranker model '{self.rerank_model_name}' initialized")
            except ImportError:
                self.logger.warning(
                    "sentence-transformers not installed, reranking will be disabled. "
                    "Install it with: pip install sentence-transformers"
                )
                self.rerank_enabled = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize reranker model: {str(e)}, reranking will be disabled")
                self.rerank_enabled = False

    def process(
        self,
        query: str,
        top_k: Optional[int] = None,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        检索文档（抽象方法，子类必须实现）

        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            tenant_id: 租户ID（用于过滤）
            **kwargs: 额外的检索参数
                - rerank_enabled: 是否启用重排序（覆盖配置）
                - similarity_threshold: 相似度阈值（覆盖配置）

        Returns:
            检索结果列表（已应用重排序和阈值过滤）
        """
        top_k = top_k or self.top_k
        self.logger.info(f"Retrieving top {top_k} results for query: {query[:50]}...")

        # 获取检索候选数量（如果启用重排序，需要更多候选）
        candidate_multiplier = self.get_config("candidate_multiplier", settings.RETRIEVAL_CANDIDATE_MULTIPLIER)
        rerank_enabled = kwargs.pop("rerank_enabled", self.rerank_enabled)
        similarity_threshold = kwargs.pop("similarity_threshold", self.similarity_threshold)

        # 计算候选数量
        candidate_k = top_k * candidate_multiplier if rerank_enabled else top_k

        # 执行检索
        results = self._retrieve(query, candidate_k, tenant_id, **kwargs)

        # 归一化分数到 [0, 1] 区间
        if results:
            results = self._normalize_scores(results)

        # 应用阈值过滤
        if similarity_threshold > 0.0:
            original_count = len(results)
            results = self._filter_by_threshold(results, similarity_threshold)
            self.logger.info(
                f"Threshold filtering: {original_count} -> {len(results)} results "
                f"(threshold={similarity_threshold:.3f})"
            )

        # 应用重排序
        if rerank_enabled and self.reranker and results:
            original_results = results.copy()
            results = self._rerank(query, results, top_k)
            self.logger.info(f"Reranking: {len(original_results)} -> {len(results)} results")

        # 确保不超过top_k
        results = results[:top_k]

        self.logger.info(f"Retrieved {len(results)} results")
        return results

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        内部检索方法，子类必须实现

        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数

        Returns:
            检索结果列表
        """
        raise NotImplementedError("Subclass must implement _retrieve method")

    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        归一化分数到 [0, 1] 区间

        Args:
            results: 检索结果列表

        Returns:
            归一化后的结果列表（原地修改）
        """
        if not results:
            return results

        # 获取分数范围
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # 如果所有分数相同或分数差为0，无需归一化
        if max_score == min_score:
            # 如果分数不在 [0, 1] 区间，统一设为1.0
            if max_score != 1.0:
                for result in results:
                    result.score = 1.0
            return results

        # Min-Max 归一化到 [0, 1] 区间
        score_range = max_score - min_score
        for result in results:
            result.score = (result.score - min_score) / score_range

        return results

    def _filter_by_threshold(
        self, results: List[RetrievalResult], threshold: float
    ) -> List[RetrievalResult]:
        """
        根据相似度阈值过滤结果

        Args:
            results: 检索结果列表
            threshold: 相似度阈值（0-1之间）

        Returns:
            过滤后的结果列表
        """
        if threshold <= 0.0:
            return results

        filtered_results = [r for r in results if r.score >= threshold]
        return filtered_results

    def _rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """
        使用重排序模型对检索结果进行重新排序

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回Top-K结果

        Returns:
            重排序后的结果列表
        """
        if not self.reranker or not results:
            return results

        try:
            # 准备重排序的输入对：[(query, content1), (query, content2), ...]
            pairs = [(query, result.content) for result in results]

            # 执行重排序（批量处理）
            rerank_scores = self.reranker.predict(pairs)

            # 更新结果分数
            for i, result in enumerate(results):
                # 重排序分数通常是相关性分数，直接使用
                result.score = float(rerank_scores[i])

            # 按新分数排序
            results.sort(key=lambda x: x.score, reverse=True)

            # 返回top_k
            return results[:top_k]

        except Exception as e:
            self.logger.warning(f"Reranking failed: {str(e)}, returning original results")
            return results
