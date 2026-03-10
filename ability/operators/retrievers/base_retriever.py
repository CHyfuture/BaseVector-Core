"""
检索器基类
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ability.config import get_settings
from ability.operators.base import BaseOperator
from ability.storage.milvus_client import milvus_client

settings = get_settings()

# 与插入集合字段对齐：检索时请求的标量字段（不含 id 与向量字段），在无法动态获取集合 schema 时作为兜底
RETRIEVAL_OUTPUT_FIELDS: List[str] = [
    "doc_id",
    "chunk_index",
    "parent_chunk_id",
    "chunk_type",
    "position_start",
    "position_end",
    "section_title",
    "section_path",
    "page",
    "content",
    "abstract_text",
    "keywords_text",
    "summary_text",
    "title",
    "authors",
    "institutions",
    "tags",
]


def _resolve_model_path_or_id(model_name: str) -> tuple[str, bool]:
    """
    Resolve model reference.
    - Local-like path: convert to absolute path and return local_only=True
    - Hub model id: return local_only=False
    """
    name = (model_name or "").strip()
    if not name:
        return name, False

    is_local_like = (
        name.startswith("workspace/")
        or name.startswith("workspace\\")
        or name.startswith("./")
        or name.startswith(".\\")
        or os.path.isabs(name)
        or (len(name) >= 2 and name[1] == ":")
        or "\\" in name
    )
    if not is_local_like:
        return name, False

    if name.startswith("workspace/") or name.startswith("workspace\\"):
        root = Path.cwd()
        resolved = (root / name.replace("\\", "/")).resolve()
    else:
        resolved = Path(name).resolve()
    return str(resolved), True


def _is_jina_listwise_reranker(model_name_or_path: str) -> bool:
    return "jina-reranker-v3" in (model_name_or_path or "").lower()


def resolve_output_fields(
    collection_name: str,
    user_output_fields: Optional[List[str]] = None,
) -> List[str]:
    """
    解析本次检索应请求的标量字段列表。
    优先使用调用方传入的 user_output_fields；否则从集合 schema 动态获取标量字段；失败时兜底为 RETRIEVAL_OUTPUT_FIELDS。
    """
    if user_output_fields is not None and len(user_output_fields) > 0:
        return list(user_output_fields)
    try:
        return milvus_client.get_scalar_fields(collection_name)
    except Exception:
        return list(RETRIEVAL_OUTPUT_FIELDS)


def metadata_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """从 Milvus 单条结果构建 RetrievalResult.metadata，按 result 中实际返回的字段动态生成（排除 id/distance/score）。"""
    exclude = {"id", "distance", "score"}
    return {k: v for k, v in result.items() if k not in exclude}


class RetrievalResult:
    """检索结果数据类"""

    def __init__(
        self,
        chunk_id: int,
        document_id: Union[int, str],
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化检索结果

        Args:
            chunk_id: 块ID
            document_id: 文档ID（可为整数或字符串，如论文标题）
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
        self.reranker_backend: Optional[str] = None

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
            model_name_or_path, local_only = _resolve_model_path_or_id(self.rerank_model_name)
            self.rerank_model_name = model_name_or_path
            use_jina_listwise = _is_jina_listwise_reranker(model_name_or_path)

            try:
                if use_jina_listwise:
                    from transformers import AutoConfig, AutoModel

                    load_kwargs = {
                        "trust_remote_code": True,
                        "local_files_only": local_only,
                    }
                    config_kwargs = dict(load_kwargs)
                    try:
                        config = AutoConfig.from_pretrained(
                            model_name_or_path,
                            **config_kwargs,
                        )
                    except TypeError:
                        config_kwargs.pop("local_files_only", None)
                        config = AutoConfig.from_pretrained(
                            model_name_or_path,
                            **config_kwargs,
                        )

                    if getattr(config, "tie_word_embeddings", None):
                        # Jina listwise reranker replaces lm_head with Identity.
                        # Disable tie weights to avoid AttributeError on lm_head.weight.
                        config.tie_word_embeddings = False

                    model_kwargs = dict(load_kwargs)
                    try:
                        self.reranker = AutoModel.from_pretrained(
                            model_name_or_path,
                            config=config,
                            torch_dtype="auto",
                            **model_kwargs,
                        )
                    except TypeError:
                        model_kwargs.pop("local_files_only", None)
                        self.reranker = AutoModel.from_pretrained(
                            model_name_or_path,
                            config=config,
                            **model_kwargs,
                        )

                    if not hasattr(self.reranker, "rerank"):
                        raise ValueError(
                            f"Reranker model '{model_name_or_path}' has no rerank() method"
                        )
                    self.reranker_backend = "jina_listwise"
                else:
                    from sentence_transformers import CrossEncoder

                    cross_encoder_kwargs = {
                        "trust_remote_code": True,
                        "local_files_only": local_only,
                    }
                    try:
                        self.reranker = CrossEncoder(
                            model_name_or_path,
                            **cross_encoder_kwargs,
                        )
                    except TypeError:
                        cross_encoder_kwargs.pop("local_files_only", None)
                        self.reranker = CrossEncoder(
                            model_name_or_path,
                            **cross_encoder_kwargs,
                        )

                    tokenizer = getattr(self.reranker, "tokenizer", None)
                    if (
                        tokenizer is not None
                        and tokenizer.pad_token_id is None
                        and tokenizer.eos_token is not None
                    ):
                        tokenizer.pad_token = tokenizer.eos_token

                    self.reranker_backend = "cross_encoder"

                self.logger.info(
                    f"Reranker model '{model_name_or_path}' initialized "
                    f"(backend={self.reranker_backend}, local_only={local_only})"
                )
            except ImportError:
                if use_jina_listwise:
                    self.logger.warning(
                        "transformers not installed, reranking will be disabled. "
                        "Install it with: pip install transformers"
                    )
                else:
                    self.logger.warning(
                        "sentence-transformers not installed, reranking will be disabled. "
                        "Install it with: pip install sentence-transformers"
                    )
                self.rerank_enabled = False
                self.reranker = None
                self.reranker_backend = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize reranker model: {str(e)}, reranking will be disabled")
                self.rerank_enabled = False
                self.reranker = None
                self.reranker_backend = None

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
                - output_fields: 本次检索要返回的标量字段列表（可选）；不传则按集合 schema 动态解析

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
            if self.reranker_backend == "jina_listwise":
                documents = [result.content for result in results]
                ranked_items = self.reranker.rerank(
                    query=query,
                    documents=documents,
                    top_n=min(top_k, len(documents)),
                )

                reranked_results: List[RetrievalResult] = []
                for item in ranked_items:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("index")
                    if not isinstance(index, int) or index < 0 or index >= len(results):
                        continue

                    result = results[index]
                    if "relevance_score" in item:
                        result.score = float(item["relevance_score"])
                    reranked_results.append(result)

                if reranked_results:
                    return reranked_results[:top_k]
                return results[:top_k]

            pairs = [(query, result.content) for result in results]

            # 执行重排序（批量处理）
            try:
                rerank_scores = self.reranker.predict(pairs)
            except Exception as e:
                error_message = str(e).lower()
                if "batch sizes > 1" in error_message and "no padding token" in error_message:
                    rerank_scores = self.reranker.predict(pairs, batch_size=1)
                else:
                    raise

            # 更新结果分数
            for result, rerank_score in zip(results, rerank_scores):
                # 重排序分数通常是相关性分数，直接使用
                result.score = float(rerank_score)

            # 按新分数排序
            results.sort(key=lambda x: x.score, reverse=True)

            # 返回top_k
            return results[:top_k]

        except Exception as e:
            self.logger.warning(f"Reranking failed: {str(e)}, returning original results")
            return results
