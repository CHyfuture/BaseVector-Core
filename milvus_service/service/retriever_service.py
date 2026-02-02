"""
RetrieverService

对 ability 中的检索算子做封装，统一暴露：
- 多种检索模式（semantic / keyword / hybrid / fulltext / text_match / phrase_match）
- 所需入参模型
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import RetrievalResult
from ability.operators.retrievers.retriever_factory import RetrieverFactory
from ability.storage.milvus_client import milvus_client


settings = get_settings()


class MilvusConnectionParams(BaseModel):
    """
    Milvus 连接配置（可选入参）

    说明：
    - 如果不提供，则使用全局 Settings 中的 MILVUS_HOST / MILVUS_PORT / MILVUS_USER / MILVUS_PASSWORD / MILVUS_DB_NAME；
    - 如果提供任意字段，则会在本次检索调用前覆盖全局客户端的对应字段。
    """

    host: Optional[str] = Field(
        default=None,
        description="Milvus 服务主机名或 IP，例如 'localhost' 或 '10.0.0.2'",
    )
    port: Optional[int] = Field(
        default=None,
        description="Milvus 服务端口号，例如 19530",
    )
    user: Optional[str] = Field(
        default=None,
        description="Milvus 用户名，如果未启用鉴权可以留空",
    )
    password: Optional[str] = Field(
        default=None,
        description="Milvus 密码，如果未启用鉴权可以留空",
    )
    db_name: Optional[str] = Field(
        default=None,
        description="Milvus 数据库名称，默认通常为 'default'",
    )


class BaseSearchRequest(BaseModel):
    """
    通用检索请求基类

    字段说明：
    - query: 查询文本
    - top_k: 返回结果数量，不填则使用全局 TOP_K（默认值：Settings.TOP_K，通常为 10）
    - rerank_enabled: 是否启用重排序（覆盖全局配置 Settings.RERANK_ENABLED，默认 False）
    - similarity_threshold: 相似度阈值（覆盖全局配置 Settings.SIMILARITY_THRESHOLD，默认 0.0）
    - rerank_model_name: 重排序模型名称（覆盖全局配置 Settings.RERANK_MODEL_NAME）
    - retrieval_candidate_multiplier: 检索候选倍数（覆盖全局配置 Settings.RETRIEVAL_CANDIDATE_MULTIPLIER，默认 2）
    - milvus_search_params: Milvus 搜索参数（覆盖全局配置，如 metric_type、nprobe 等）
    - milvus_connection: Milvus 连接配置（覆盖全局配置）
    
    配置优先级：请求参数 > 全局 Settings 默认值
    """

    query: str = Field(..., description="查询文本")
    top_k: Optional[int] = Field(
        default=None,
        description="Top-K 结果数量，不填则使用全局配置 Settings.TOP_K（默认值：10）",
    )
    rerank_enabled: Optional[bool] = Field(
        default=None,
        description="是否启用重排序，未设置则使用全局配置 Settings.RERANK_ENABLED（默认值：False）",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="相似度阈值（0-1），小于该值的结果将被过滤，未设置则使用全局配置 Settings.SIMILARITY_THRESHOLD（默认值：0.0）",
    )
    rerank_model_name: Optional[str] = Field(
        default=None,
        description="重排序模型名称或路径，覆盖全局配置 Settings.RERANK_MODEL_NAME（默认值：'workspace/jina-reranker-v3'）。支持 HuggingFace Hub 模型标识符或本地路径",
    )
    retrieval_candidate_multiplier: Optional[int] = Field(
        default=None,
        ge=1,
        description="检索候选倍数，在重排序前检索的候选文档数量 = top_k * retrieval_candidate_multiplier，覆盖全局配置 Settings.RETRIEVAL_CANDIDATE_MULTIPLIER（默认值：2）",
    )
    milvus_search_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Milvus 搜索参数字典，覆盖全局配置。格式示例："
            "  - IVF系列: {'metric_type': 'L2', 'params': {'nprobe': 10}}"
            "  - HNSW: {'metric_type': 'L2', 'params': {'ef': 100}}"
            "  - FLAT: {'metric_type': 'L2'} (FLAT不需要params)"
            "如果不提供，则使用全局配置 Settings.MILVUS_METRIC_TYPE 和 Settings.MILVUS_NPROBE"
        ),
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次检索要使用的 Milvus 连接配置，不填则使用全局默认配置（Settings.MILVUS_HOST / MILVUS_PORT 等）",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Milvus 集合名称。不填时使用 MILVUS_COLLECTION_TEMPLATE + tenant_id 生成；填写时直接使用该集合名",
    )
    output_fields: Optional[List[str]] = Field(
        default=None,
        description="本次检索要返回的标量字段列表（不含 id/向量字段）。不填则按集合 schema 动态解析；填写则仅返回指定字段",
    )


class SemanticSearchRequest(BaseSearchRequest):
    """语义检索请求"""

    query_vector: List[float] = Field(
        ...,
        description="查询向量，必须与集合中存储的向量维度一致，需由上层模型生成",
    )
    anns_field: str = Field(
        default="vector",
        description="Milvus 中用于向量检索的字段名，默认 'vector'",
    )
    milvus_expr: Optional[str] = Field(
        default=None,
        description="额外的 Milvus 过滤表达式，例如 'chunk_index < 3'",
    )


class KeywordSearchRequest(BaseSearchRequest):
    """关键词检索请求（BM25 风格）"""

    # 目前具体实现主要依赖 query/top_k，可预留额外参数
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外参数，透传给具体检索器实现",
    )


class HybridSearchRequest(BaseSearchRequest):
    """混合检索请求（语义 + 关键词）"""

    query_vector: List[float] = Field(
        ...,
        description="查询向量，必须与集合中存储的向量维度一致，需由上层模型生成",
    )
    anns_field: str = Field(
        default="vector",
        description="Milvus 中用于向量检索的字段名，默认 'vector'",
    )
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外参数，例如语义/关键词权重等，透传给具体检索器",
    )


class FulltextSearchRequest(BaseSearchRequest):
    """全文检索请求（LIKE 模式等）"""

    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外参数，透传给具体检索器实现",
    )


class TextMatchSearchRequest(BaseSearchRequest):
    """文本匹配检索请求"""

    match_type: str = Field(
        default="fuzzy",
        description="匹配类型：'exact' 精确匹配 或 'fuzzy' 模糊匹配（默认）",
    )
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外参数，透传给具体检索器实现",
    )


class PhraseMatchSearchRequest(BaseSearchRequest):
    """短语匹配检索请求"""

    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外参数，透传给具体检索器实现",
    )


class RetrievalResultDTO(BaseModel):
    """检索结果 DTO，与 RetrievalResult 基本等价，但便于序列化"""

    chunk_id: int = Field(..., description="块 ID")
    document_id: Union[int, str] = Field(..., description="文档 ID（可为整数或字符串，如论文标题）")
    content: str = Field(..., description="文本内容")
    score: float = Field(..., description="相似度分数（0-1 或重排序后的得分）")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "与当前结果相关的元数据字典，通常包含："
            "chunk_index（块序号）/ parent_chunk_id（父块 ID）/ "
            "以及上游写入时附加的任意业务字段（例如 source/file_name/page_ranges 等），会原样透传"
        ),
    )

    @classmethod
    def from_result(cls, r: RetrievalResult) -> "RetrievalResultDTO":
        """从 RetrievalResult 实例构造 DTO"""
        return cls(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            content=r.content,
            score=r.score,
            metadata=r.metadata or {},
        )


class RetrieverService:
    """检索相关 service，封装 RetrieverFactory"""

    @staticmethod
    def _apply_milvus_connection(params: Optional[MilvusConnectionParams]) -> None:
        """
        根据请求中提供的 Milvus 连接配置，动态更新全局 Milvus 客户端。

        如果 params 为 None，则保持现有全局配置不变。
        """
        if params is None:
            return

        milvus_client.update_connection_config(
            host=params.host,
            port=params.port,
            user=params.user,
            password=params.password,
            db_name=params.db_name,
        )

    @staticmethod
    def _build_common_kwargs(request: BaseSearchRequest) -> Dict[str, Any]:
        """
        构造通用可选参数字典
        
        将请求中的配置字段传递给检索器的 process() 方法
        """
        kwargs: Dict[str, Any] = {}
        if request.rerank_enabled is not None:
            kwargs["rerank_enabled"] = request.rerank_enabled
        if request.similarity_threshold is not None:
            kwargs["similarity_threshold"] = request.similarity_threshold
        if request.milvus_search_params is not None:
            kwargs["search_params"] = request.milvus_search_params
        if request.collection_name is not None:
            kwargs["collection_name"] = request.collection_name
        if request.output_fields is not None:
            kwargs["output_fields"] = request.output_fields
        return kwargs
    
    @staticmethod
    def _build_retriever_config(request: BaseSearchRequest) -> Dict[str, Any]:
        """
        构造检索器初始化配置字典
        
        将请求中的配置字段传递给 RetrieverFactory.create_retriever() 的 config 参数
        """
        config: Dict[str, Any] = {}
        if request.rerank_enabled is not None:
            config["rerank_enabled"] = request.rerank_enabled
        if request.rerank_model_name is not None:
            config["rerank_model_name"] = request.rerank_model_name
        if request.retrieval_candidate_multiplier is not None:
            # BaseRetriever.process() 中使用的是 candidate_multiplier，需要映射
            config["candidate_multiplier"] = request.retrieval_candidate_multiplier
        if request.similarity_threshold is not None:
            config["similarity_threshold"] = request.similarity_threshold
        return config

    @staticmethod
    def semantic_search(request: SemanticSearchRequest) -> List[RetrievalResultDTO]:
        """语义检索"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        retriever = RetrieverFactory.create_retriever(mode="semantic", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        kwargs.update(
            {
                "query_vector": request.query_vector,
                "anns_field": request.anns_field,
                "milvus_expr": request.milvus_expr,
            }
        )

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def keyword_search(request: KeywordSearchRequest) -> List[RetrievalResultDTO]:
        """关键词检索（BM25 风格）"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        retriever = RetrieverFactory.create_retriever(mode="keyword", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        if request.extra_params:
            kwargs.update(request.extra_params)

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def hybrid_search(request: HybridSearchRequest) -> List[RetrievalResultDTO]:
        """混合检索（语义 + 关键词）"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        retriever = RetrieverFactory.create_retriever(mode="hybrid", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        kwargs.update(
            {
                "query_vector": request.query_vector,
                "anns_field": request.anns_field,
            }
        )
        if request.extra_params:
            kwargs.update(request.extra_params)

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def fulltext_search(request: FulltextSearchRequest) -> List[RetrievalResultDTO]:
        """全文检索"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        retriever = RetrieverFactory.create_retriever(mode="fulltext", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        if request.extra_params:
            kwargs.update(request.extra_params)

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def text_match_search(request: TextMatchSearchRequest) -> List[RetrievalResultDTO]:
        """文本匹配检索"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        # 通过 config 传入 match_type
        retriever_config["match_type"] = request.match_type
        retriever = RetrieverFactory.create_retriever(mode="text_match", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        if request.extra_params:
            kwargs.update(request.extra_params)

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def phrase_match_search(
        request: PhraseMatchSearchRequest,
    ) -> List[RetrievalResultDTO]:
        """短语匹配检索"""
        RetrieverService._apply_milvus_connection(request.milvus_connection)
        retriever_config = RetrieverService._build_retriever_config(request)
        retriever = RetrieverFactory.create_retriever(mode="phrase_match", config=retriever_config)

        kwargs = RetrieverService._build_common_kwargs(request)
        if request.extra_params:
            kwargs.update(request.extra_params)

        results = retriever.process(
            query=request.query,
            top_k=request.top_k or settings.TOP_K,
            **kwargs,
        )
        return [RetrievalResultDTO.from_result(r) for r in results]

    @staticmethod
    def get_supported_modes() -> list[str]:
        """返回当前系统支持的检索模式列表"""
        return RetrieverFactory.get_supported_modes()

