"""
配置管理模块（仅包含 operators 需要的配置）
支持：环境变量 / .env / config/config.yaml
优先级：ENV > .env > YAML > 代码默认
"""
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """配置（仅 operators 需要的配置项）"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== 配置文件路径 ====================
    # YAML配置文件路径（可选）
    # 支持相对路径和绝对路径，相对路径基于当前工作目录
    # 如果文件不存在，将使用默认配置值
    CONFIG_YAML_PATH: str = "config/config.yaml"

    # ==================== 日志配置 ====================
    # 日志级别
    # 可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # 用于控制日志输出的详细程度
    LOG_LEVEL: str = "INFO"

    # ==================== Milvus 向量数据库配置 ====================
    # Milvus 服务器主机地址
    # 用于连接 Milvus 向量数据库，SemanticRetriever 需要此配置
    MILVUS_HOST: str = "192.168.31.51"

    # Milvus 服务器端口号
    # 默认端口为 19530，根据实际部署情况调整
    MILVUS_PORT: int = 19530

    # Milvus 用户名
    # 如果 Milvus 启用了认证，需要设置此值
    # 留空表示不使用认证
    MILVUS_USER: str = ""

    # Milvus 密码
    # 如果 Milvus 启用了认证，需要设置此值
    # 留空表示不使用认证
    MILVUS_PASSWORD: str = ""

    # Milvus 数据库名称
    # 指定要使用的 Milvus 数据库，默认为 "default"
    MILVUS_DB_NAME: str = "base_database_vector"

    # Milvus 索引类型
    # 可选值: IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, FLAT 等
    # IVF_FLAT: 平衡精度和性能的常用选择
    # HNSW: 适合高精度场景，但内存占用较大
    # FLAT: 最高精度，但查询速度较慢
    MILVUS_INDEX_TYPE: str = "HNSW"

    # Milvus 距离度量类型
    # 可选值: L2（欧氏距离）, IP（内积）, COSINE（余弦相似度）
    # L2: 适用于大多数场景，值越小越相似
    # IP: 适用于归一化向量，值越大越相似
    # COSINE: 适用于文本相似度，值越大越相似
    MILVUS_METRIC_TYPE: str = "IP"

    # Milvus IVF 索引的聚类中心数量（nlist）
    # 影响索引构建速度和查询精度
    # 建议值: 数据量 / 1000 到 数据量 / 100
    # 值越大，索引构建越慢，但查询精度可能更高
    MILVUS_NLIST: int = 1024

    # Milvus 查询时的搜索聚类中心数量（nprobe）
    # 影响查询速度和精度
    # 建议值: nlist 的 1/10 到 1/100
    # 值越大，查询越慢，但精度越高
    MILVUS_NPROBE: int = 10

    # Milvus 集合名称模板
    # 支持使用 {tenant_id} 占位符实现多租户
    # 例如: "documents_{tenant_id}" 会生成 "documents_tenant1", "documents_tenant2" 等
    MILVUS_COLLECTION_TEMPLATE: str = "documents_{tenant_id}"

    # ==================== 文档切片配置 ====================
    # 切片大小（字符数）
    # 每个文本块的最大字符数
    # 建议值: 256-1024，根据模型上下文窗口调整
    # 较小的值: 更精确的检索，但可能丢失上下文
    # 较大的值: 保留更多上下文，但检索精度可能下降
    CHUNK_SIZE: int = 512

    # 切片重叠大小（字符数）
    # 相邻切片之间的重叠字符数，用于保持上下文连续性
    # 建议值: CHUNK_SIZE 的 5%-20%
    # 值太小: 可能丢失跨切片的上下文
    # 值太大: 会产生大量重复内容
    CHUNK_OVERLAP: int = 50

    # 切片策略
    # 可选值: "fixed"（固定大小）, "semantic"（语义分割）, "title"（标题分割）, "parent_child"（父子关系）
    # fixed: 按固定大小切分，速度快但可能切断语义
    # semantic: 基于语义相似度切分，保持语义完整性
    # title: 基于标题层级切分，适合结构化文档
    # parent_child: 基于父子关系切分，适合层次化文档
    CHUNK_STRATEGY: str = "parent_child"

    # ==================== 文档解析配置（MinerU / LibreOffice） ====================
    # 是否启用 MinerU 解析（PDF/Word/PPT 等）
    # True: 使用 MinerU API 解析，非 PDF 先经 LibreOffice 转为 PDF 再解析，返回 Markdown
    # False: 使用内置解析器（PyMuPDF / python-docx / python-pptx）
    MINERU_ENABLED: bool = True

    # MinerU 服务地址（例如 http://localhost:8000）
    MINERU_URL: str = "http://192.168.1.5:8000"

    # MinerU 解析后端（按 MinerU 文档配置，如 "pdf"）
    MINERU_BACKEND: str = "pdf"

    # MinerU 解析方法（按 MinerU 文档配置）
    MINERU_PARSE_METHOD: str = "auto"

    # MinerU 调用锁等待超时（秒），避免并发打满服务
    MINERU_LOCK_TIMEOUT: int = 3600
    MINERU_LOCK_WAIT_TIMEOUT: int = 600

    # MinerU API 请求超时（秒）
    MINERU_REQUEST_TIMEOUT: int = 3000

    # LibreOffice 转 PDF 命令超时（秒）
    LIBREOFFICE_CONVERT_TIMEOUT: int = 3000

    # ==================== 检索配置 ====================
    # 检索结果数量（Top-K）
    # 每次检索返回的最相关文档数量
    # 建议值: 5-20，根据应用场景调整
    # 值太小: 可能遗漏相关文档
    # 值太大: 返回结果过多，影响性能
    TOP_K: int = 10

    # 检索模式
    # 可选值: "semantic"（语义检索）, "keyword"（关键词检索）, "hybrid"（混合检索）
    # semantic: 仅使用向量相似度检索，适合语义理解场景
    # keyword: 仅使用关键词匹配检索，适合精确匹配场景
    # hybrid: 结合语义和关键词检索，通常效果最好
    SEARCH_MODE: str = "hybrid"

    # 是否启用重排序（Rerank）
    # 对检索结果进行二次排序，提高结果相关性
    # True: 启用重排序，需要 RERANK_MODEL_NAME 配置
    # False: 不启用，直接返回检索结果
    RERANK_ENABLED: bool = False

    # 重排序模型名称或路径
    # 支持两种方式：
    # 1. HuggingFace Hub 模型标识符（自动下载到缓存）:
    #    - "BAAI/bge-reranker-large"（中文推荐，精度高）
    #    - "BAAI/bge-reranker-base"（中文，速度更快）
    #    - "BAAI/bge-reranker-large-v2-m3"（多语言）
    # 2. 本地模型路径（绝对路径或相对路径）:
    #    - 绝对路径: "/path/to/workspace/bge-reranker-large"
    #    - 相对路径: "workspace/bge-reranker-large"（相对于项目根目录）
    #    - 本地模型目录应包含 config.json、pytorch_model.bin 等文件
    # 注意: 启用重排序可以显著提高检索精度，但会增加计算开销
    # 建议: 将本地模型放在项目根目录下的 workspace/ 目录中
    RERANK_MODEL_NAME: str = "workspace/jina-reranker-v3"

    # 检索候选倍数
    # 在重排序前检索的候选文档数量 = TOP_K * RETRIEVAL_CANDIDATE_MULTIPLIER
    # 例如: TOP_K=10, RETRIEVAL_CANDIDATE_MULTIPLIER=2，则先检索 20 个，再重排序选出 10 个
    # 建议值: 2-5，值越大精度可能越高，但计算开销也越大
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 2

    # 相似度阈值
    # 过滤检索结果的最小相似度分数（0-1之间）
    # 分数低于此阈值的结果将被过滤掉，不返回
    # 0.0: 不过滤，返回所有结果
    # 0.5-0.7: 中等阈值，过滤掉不太相关的结果
    # 0.7-0.9: 高阈值，只返回高度相关的结果
    # 注意: 不同检索模式（semantic/keyword/hybrid）的分数范围可能不同，需要根据实际情况调整
    SIMILARITY_THRESHOLD: float = 0.0

    # 关键词全文检索语言
    # PostgreSQL/Milvus 全文检索使用的语言配置
    # 可选值: "simple"（简单分词）, "english", "chinese" 等
    # simple: 通用配置，适合大多数场景
    # 根据实际文档语言选择合适的配置
    KEYWORD_FTS_LANGUAGE: str = "simple"

    # ==================== 多租户权限配置 ====================
    # 是否启用多租户模式
    # True: 启用多租户，不同租户的数据隔离
    # False: 单租户模式，所有数据共享
    # 启用后，检索和存储操作需要提供 tenant_id
    ENABLE_MULTI_TENANT: bool = True

    # 默认租户 ID
    # 当未指定 tenant_id 时使用的默认租户标识
    # 仅在 ENABLE_MULTI_TENANT=True 时生效
    DEFAULT_TENANT_ID: str = "default"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """自定义配置来源顺序：ENV > .env > YAML > 默认"""

        def yaml_settings_source() -> dict[str, Any]:
            try:
                import yaml
            except Exception:
                return {}

            # 避免递归：直接使用默认值或从环境变量读取
            # 不能创建Settings实例，否则会导致无限递归
            config_yaml_path = os.getenv("CONFIG_YAML_PATH", "config/config.yaml")
            cfg_path = Path(config_yaml_path)
            if not cfg_path.is_absolute():
                cfg_path = Path.cwd() / cfg_path
            if not cfg_path.exists():
                return {}

            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                return {}

            flat: dict[str, Any] = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str):
                        flat[k] = v

            # 映射结构化YAML到扁平配置
            if isinstance(data, dict):
                # retrieval
                rt = data.get("retrieval")
                if isinstance(rt, dict):
                    if "default_top_k" in rt:
                        flat.setdefault("TOP_K", rt["default_top_k"])
                    if "candidate_multiplier" in rt:
                        flat.setdefault("RETRIEVAL_CANDIDATE_MULTIPLIER", rt["candidate_multiplier"])
                    if "keyword_fts_language" in rt:
                        flat.setdefault("KEYWORD_FTS_LANGUAGE", rt["keyword_fts_language"])
                    if "similarity_threshold" in rt:
                        flat.setdefault("SIMILARITY_THRESHOLD", rt["similarity_threshold"])
                    rerank = rt.get("rerank")
                    if isinstance(rerank, dict):
                        if "enabled" in rerank:
                            flat.setdefault("RERANK_ENABLED", rerank["enabled"])
                        if "model" in rerank:
                            flat.setdefault("RERANK_MODEL_NAME", rerank["model"])

                # milvus
                mv = data.get("milvus")
                if isinstance(mv, dict):
                    idx = mv.get("index")
                    if isinstance(idx, dict):
                        if "index_type" in idx:
                            flat.setdefault("MILVUS_INDEX_TYPE", idx["index_type"])
                        if "metric_type" in idx:
                            flat.setdefault("MILVUS_METRIC_TYPE", idx["metric_type"])
                        if "nlist" in idx:
                            flat.setdefault("MILVUS_NLIST", idx["nlist"])
                    sv = mv.get("search")
                    if isinstance(sv, dict):
                        if "nprobe" in sv:
                            flat.setdefault("MILVUS_NPROBE", sv["nprobe"])
                    if "collection_template" in mv:
                        flat.setdefault("MILVUS_COLLECTION_TEMPLATE", mv["collection_template"])

                # mineru / 文档解析
                mineru = data.get("mineru") or data.get("parsers")
                if isinstance(mineru, dict):
                    if "enabled" in mineru:
                        flat.setdefault("MINERU_ENABLED", mineru["enabled"])
                    if "url" in mineru:
                        flat.setdefault("MINERU_URL", mineru["url"])
                    if "backend" in mineru:
                        flat.setdefault("MINERU_BACKEND", mineru["backend"])
                    if "parse_method" in mineru:
                        flat.setdefault("MINERU_PARSE_METHOD", mineru["parse_method"])
                    if "lock_timeout" in mineru:
                        flat.setdefault("MINERU_LOCK_TIMEOUT", mineru["lock_timeout"])
                    if "lock_wait_timeout" in mineru:
                        flat.setdefault("MINERU_LOCK_WAIT_TIMEOUT", mineru["lock_wait_timeout"])
                    if "request_timeout" in mineru:
                        flat.setdefault("MINERU_REQUEST_TIMEOUT", mineru["request_timeout"])
                    if "libreoffice_convert_timeout" in mineru:
                        flat.setdefault("LIBREOFFICE_CONVERT_TIMEOUT", mineru["libreoffice_convert_timeout"])

            return flat

        return (
            env_settings,
            dotenv_settings,
            yaml_settings_source,
            init_settings,
            file_secret_settings,
        )


# 全局配置实例（支持动态设置）
_global_settings: Optional[Settings] = None


def create_settings(**kwargs) -> Settings:
    """
    创建自定义配置实例（类似 Redis 客户端创建方式）
    
    用法：
        # 方式1：完全自定义配置
        custom_settings = create_settings(
            MILVUS_HOST="10.0.0.2",
            MILVUS_PORT=19530,
            TOP_K=20,
            RERANK_ENABLED=True,
        )
        
        # 方式2：基于现有配置，只覆盖部分字段
        base_settings = get_settings()
        custom_settings = create_settings(
            **base_settings.model_dump(),
            MILVUS_HOST="10.0.0.2",  # 只覆盖这一个字段
        )
    
    Args:
        **kwargs: 配置参数字典，支持 Settings 类的所有字段
        
    Returns:
        Settings 实例
        
    示例：
        ```python
        # 创建自定义配置
        my_settings = create_settings(
            MILVUS_HOST="prod-milvus.example.com",
            MILVUS_PORT=19530,
            MILVUS_USER="admin",
            MILVUS_PASSWORD="secret",
            TOP_K=20,
            RERANK_ENABLED=True,
        )
        
        # 设置为全局配置
        set_settings(my_settings)
        
        # 后续所有调用都会使用这个配置
        from milvus_service import RetrieverService
        results = RetrieverService.semantic_search(...)
        ```
    """
    return Settings(**kwargs)


def set_settings(settings: Settings) -> None:
    """
    设置全局配置实例（类似 Redis 设置全局客户端）
    
    设置后，所有通过 get_settings() 获取的配置都会使用这个实例。
    如果不调用此函数，get_settings() 会从环境变量/配置文件读取。
    
    用法：
        ```python
        # 创建并设置全局配置
        custom_settings = create_settings(
            MILVUS_HOST="prod-milvus.example.com",
            TOP_K=20,
        )
        set_settings(custom_settings)
        
        # 后续所有 Service 调用都会使用这个配置
        from milvus_service import RetrieverService
        results = RetrieverService.semantic_search(...)
        ```
    
    Args:
        settings: Settings 实例
        
    注意：
        - 此函数会修改全局状态，在多线程环境下请谨慎使用
        - 如果需要不同配置，建议使用上下文管理器 with_settings()
    """
    global _global_settings
    _global_settings = settings


def reset_settings() -> None:
    """
    重置全局配置为 None，后续 get_settings() 会从环境变量/配置文件读取
    
    用法：
        ```python
        # 临时使用自定义配置
        custom_settings = create_settings(MILVUS_HOST="temp-host")
        set_settings(custom_settings)
        
        # ... 使用自定义配置 ...
        
        # 恢复默认配置
        reset_settings()
        ```
    """
    global _global_settings
    _global_settings = None


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例
    
    优先级：
    1. 如果通过 set_settings() 设置了全局配置，返回设置的配置
    2. 否则从环境变量 / .env / config.yaml 读取（单例模式）
    
    用法：
        ```python
        # 方式1：使用默认配置（从环境变量/配置文件读取）
        settings = get_settings()
        
        # 方式2：先设置自定义配置，再获取
        custom_settings = create_settings(MILVUS_HOST="custom-host")
        set_settings(custom_settings)
        settings = get_settings()  # 返回 custom_settings
        
        # 方式3：使用上下文管理器（推荐，线程安全）
        with with_settings(create_settings(MILVUS_HOST="temp-host")):
            settings = get_settings()  # 返回临时配置
        # 退出上下文后自动恢复
        ```
    
    Returns:
        Settings 实例
    """
    global _global_settings
    if _global_settings is not None:
        return _global_settings
    return Settings()


@contextmanager
def with_settings(settings: Settings) -> Generator[Settings, None, None]:
    """
    配置上下文管理器（推荐使用，线程安全）
    
    在上下文内使用指定的配置，退出上下文后自动恢复原配置。
    适合临时切换配置的场景。
    
    用法：
        ```python
        # 临时使用自定义配置
        custom_settings = create_settings(
            MILVUS_HOST="temp-host",
            TOP_K=50,
        )
        
        with with_settings(custom_settings):
            # 在这个代码块内，所有 get_settings() 调用都返回 custom_settings
            from milvus_service import RetrieverService
            results = RetrieverService.semantic_search(...)
        
        # 退出上下文后，自动恢复为默认配置
        ```
    
    Args:
        settings: 要使用的 Settings 实例
        
    Yields:
        Settings 实例
        
    示例：
        ```python
        # 为不同租户使用不同配置
        tenant_a_settings = create_settings(
            MILVUS_DB_NAME="tenant_a_db",
            TOP_K=10,
        )
        tenant_b_settings = create_settings(
            MILVUS_DB_NAME="tenant_b_db",
            TOP_K=20,
        )
        
        # 处理租户 A 的请求
        with with_settings(tenant_a_settings):
            results_a = RetrieverService.semantic_search(...)
        
        # 处理租户 B 的请求
        with with_settings(tenant_b_settings):
            results_b = RetrieverService.semantic_search(...)
        ```
    """
    global _global_settings
    old_settings = _global_settings
    try:
        _global_settings = settings
        yield settings
    finally:
        _global_settings = old_settings
