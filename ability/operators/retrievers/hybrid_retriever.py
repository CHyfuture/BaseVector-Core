"""
混合检索器
结合语义检索和关键词检索，使用RRF（倒数排名融合）算法
"""
from typing import Any, Dict, List, Optional

from ability.operators.retrievers.base_retriever import BaseRetriever, RetrievalResult
from ability.operators.retrievers.keyword_retriever import KeywordRetriever
from ability.operators.retrievers.semantic_retriever import SemanticRetriever


class HybridRetriever(BaseRetriever):
    """混合检索器，结合语义检索和关键词检索"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化混合检索器

        Args:
            config: 配置字典
                - top_k: 返回Top-K结果（默认10）
                - semantic_weight: 语义检索权重（默认0.5）
                - keyword_weight: 关键词检索权重（默认0.5）
                - fusion_method: 融合方法（"rrf"、"weighted"或自定义函数，默认"rrf"）
                - fusion_function: 自定义融合函数（可选，当fusion_method="custom"时使用）
                   函数签名: (semantic_results: List[RetrievalResult], 
                            keyword_results: List[RetrievalResult], 
                            top_k: int) -> List[RetrievalResult]
                - rrf_k: RRF算法的k参数（默认60）
                - semantic_config: 语义检索器配置
                - keyword_config: 关键词检索器配置
        """
        super().__init__(config)
        self.semantic_weight = self.get_config("semantic_weight", 0.5)
        self.keyword_weight = self.get_config("keyword_weight", 0.5)
        self.fusion_method = self.get_config("fusion_method", "rrf")
        self.fusion_function = self.get_config("fusion_function", None)
        self.rrf_k = self.get_config("rrf_k", 60)

        semantic_config = self.get_config("semantic_config", {})
        keyword_config = self.get_config("keyword_config", {})

        self.semantic_retriever = SemanticRetriever(config=semantic_config)
        self.keyword_retriever = KeywordRetriever(config=keyword_config)

    def _initialize(self) -> None:
        """初始化子检索器"""
        super()._initialize()  # 调用基类初始化（包括重排序模型）
        self.semantic_retriever.initialize()
        self.keyword_retriever.initialize()
        self.logger.info("Hybrid retriever initialized")

    def _retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            tenant_id: 租户ID
            **kwargs: 额外的检索参数

        Returns:
            检索结果列表
        """
        # 1. 并行执行两种检索
        # 注意：在子检索器中禁用重排序和阈值过滤，因为融合后会统一应用
        candidate_k = top_k * 2
        sub_kwargs = {**kwargs, "rerank_enabled": False, "similarity_threshold": 0.0}
        # 语义检索需要 query_vector（由调用方提供并透传）
        semantic_results = self.semantic_retriever.process(query, candidate_k, tenant_id, **sub_kwargs)
        keyword_results = self.keyword_retriever.process(query, candidate_k, tenant_id, **sub_kwargs)

        # 2. 融合结果
        if self.fusion_method == "rrf":
            fused_results = self._rrf_fusion(semantic_results, keyword_results, top_k)
        elif self.fusion_method == "weighted":
            fused_results = self._weighted_fusion(semantic_results, keyword_results, top_k)
        elif self.fusion_method == "custom" and self.fusion_function:
            # 使用自定义融合函数
            fused_results = self.fusion_function(semantic_results, keyword_results, top_k)
        else:
            # 如果fusion_method是callable，直接使用
            if callable(self.fusion_method):
                fused_results = self.fusion_method(semantic_results, keyword_results, top_k)
            else:
                # 默认使用RRF
                self.logger.warning(
                    f"Unknown fusion method: {self.fusion_method}, falling back to RRF"
                )
                fused_results = self._rrf_fusion(semantic_results, keyword_results, top_k)

        return fused_results

    def _rrf_fusion(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        使用RRF（倒数排名融合）算法融合结果

        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            top_k: 返回Top-K结果

        Returns:
            融合后的结果列表
        """
        # RRF参数
        k = self.rrf_k

        # 构建chunk_id到分数的映射
        scores = {}

        # 处理语义检索结果
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank)
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "semantic_score": rrf_score,
                    "keyword_score": 0.0,
                }
            else:
                scores[chunk_id]["semantic_score"] = rrf_score

        # 处理关键词检索结果
        for rank, result in enumerate(keyword_results, 1):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank)
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": rrf_score,
                }
            else:
                scores[chunk_id]["keyword_score"] = rrf_score

        # 计算融合分数
        fused_results = []
        for chunk_id, score_data in scores.items():
            result = score_data["result"]
            fused_score = (
                self.semantic_weight * score_data["semantic_score"]
                + self.keyword_weight * score_data["keyword_score"]
            )
            result.score = fused_score
            fused_results.append(result)

        # 按分数排序
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results[:top_k]

    def _weighted_fusion(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        使用加权融合算法融合结果

        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            top_k: 返回Top-K结果

        Returns:
            融合后的结果列表
        """
        # 归一化分数到[0, 1]区间
        if semantic_results:
            max_semantic_score = max(r.score for r in semantic_results)
            if max_semantic_score > 0:
                for result in semantic_results:
                    result.score = result.score / max_semantic_score

        if keyword_results:
            max_keyword_score = max(r.score for r in keyword_results)
            if max_keyword_score > 0:
                for result in keyword_results:
                    result.score = result.score / max_keyword_score

        # 构建chunk_id到结果的映射
        results_map = {}

        # 处理语义检索结果
        for result in semantic_results:
            chunk_id = result.chunk_id
            if chunk_id not in results_map:
                results_map[chunk_id] = {
                    "result": result,
                    "semantic_score": result.score * self.semantic_weight,
                    "keyword_score": 0.0,
                }
            else:
                results_map[chunk_id]["semantic_score"] = result.score * self.semantic_weight

        # 处理关键词检索结果
        for result in keyword_results:
            chunk_id = result.chunk_id
            if chunk_id not in results_map:
                results_map[chunk_id] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.score * self.keyword_weight,
                }
            else:
                results_map[chunk_id]["keyword_score"] = result.score * self.keyword_weight

        # 计算融合分数
        fused_results = []
        for chunk_id, score_data in results_map.items():
            result = score_data["result"]
            fused_score = score_data["semantic_score"] + score_data["keyword_score"]
            result.score = fused_score
            fused_results.append(result)

        # 按分数排序
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results[:top_k]
