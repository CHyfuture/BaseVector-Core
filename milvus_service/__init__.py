"""
milvus_service 包

统一对外暴露所有核心接口，支持通过 pip install git+ 方式安装使用。

主要模块：
- 配置管理：Settings / get_settings
- 算子工厂：ParserFactory / ChunkerFactory / RetrieverFactory / StorageFactory
- Service 层：ParserService / ChunkerService / RetrieverService / StorageService
- 插件注册：RegistryService / PluginRegistry
- 存储客户端：milvus_client
"""

# 配置管理
from ability.config import (
    Settings,
    get_settings,
    create_settings,
    set_settings,
    reset_settings,
    with_settings,
)

# 算子工厂
from ability.operators.chunkers.chunker_factory import ChunkerFactory
from ability.operators.parsers.parser_factory import ParserFactory
from ability.operators.retrievers.retriever_factory import RetrieverFactory
from ability.operators.storage.storage_factory import StorageFactory

# Service 层（统一入口）
from milvus_service.service import (
    # Parser
    ParserService,
    ParseRequest,
    ParseResponse,
    # Chunker
    ChunkerService,
    ChunkRequest,
    ChunkDTO,
    # Retriever
    RetrieverService,
    BaseSearchRequest,
    SemanticSearchRequest,
    KeywordSearchRequest,
    HybridSearchRequest,
    FulltextSearchRequest,
    TextMatchSearchRequest,
    PhraseMatchSearchRequest,
    RetrievalResultDTO,
    # Storage
    StorageService,
    CreateCollectionRequest,
    DeleteCollectionRequest,
    GetCollectionRequest,
    ExistsCollectionRequest,
    ListCollectionsRequest,
    InsertRequest,
    # Registry
    RegistryService,
    PluginInfo,
    LoadPluginRequest,
    RegisterParserRequest,
    RegisterChunkerRequest,
    RegisterRetrieverRequest,
    RegisterStorageOperatorRequest,
    UnregisterRequest,
    UnregisterResponse,
    ClearAllResponse,
    RegistrySnapshot,
    # Descriptor
    AbilityDescriptor,
)

# 存储客户端
from ability.storage.milvus_client import milvus_client, MilvusClient

# 插件注册（底层）
from ability.operators.plugin_registry import PluginRegistry

__all__ = [
    # 配置
    "Settings",
    "get_settings",
    "create_settings",
    "set_settings",
    "reset_settings",
    "with_settings",
    # 算子工厂
    "ParserFactory",
    "ChunkerFactory",
    "RetrieverFactory",
    "StorageFactory",
    # Service 层 - Parser
    "ParserService",
    "ParseRequest",
    "ParseResponse",
    # Service 层 - Chunker
    "ChunkerService",
    "ChunkRequest",
    "ChunkDTO",
    # Service 层 - Retriever
    "RetrieverService",
    "BaseSearchRequest",
    "SemanticSearchRequest",
    "KeywordSearchRequest",
    "HybridSearchRequest",
    "FulltextSearchRequest",
    "TextMatchSearchRequest",
    "PhraseMatchSearchRequest",
    "RetrievalResultDTO",
    # Service 层 - Storage
    "StorageService",
    "CreateCollectionRequest",
    "DeleteCollectionRequest",
    "GetCollectionRequest",
    "ExistsCollectionRequest",
    "ListCollectionsRequest",
    "InsertRequest",
    # Service 层 - Registry
    "RegistryService",
    "PluginInfo",
    "LoadPluginRequest",
    "RegisterParserRequest",
    "RegisterChunkerRequest",
    "RegisterRetrieverRequest",
    "RegisterStorageOperatorRequest",
    "UnregisterRequest",
    "UnregisterResponse",
    "ClearAllResponse",
    "RegistrySnapshot",
    # Service 层 - Descriptor
    "AbilityDescriptor",
    # 存储客户端
    "milvus_client",
    "MilvusClient",
    # 插件注册（底层）
    "PluginRegistry",
]

