"""
service 层统一入口

这里主要提供：
- ParserService：文档解析能力
- ChunkerService：文本切片能力
- StorageService：集合管理与数据写入能力
- RetrieverService：多种检索能力
- AbilityDescriptor：能力索引与能力自描述
"""

from .parser_service import ParserService, ParseRequest, ParseResponse
from .chunker_service import ChunkerService, ChunkRequest, ChunkDTO
from .storage_service import (
    StorageService,
    CreateCollectionRequest,
    DeleteCollectionRequest,
    GetCollectionRequest,
    ExistsCollectionRequest,
    ListCollectionsRequest,
    InsertRequest,
)
from .retriever_service import (
    RetrieverService,
    BaseSearchRequest,
    SemanticSearchRequest,
    KeywordSearchRequest,
    HybridSearchRequest,
    FulltextSearchRequest,
    TextMatchSearchRequest,
    PhraseMatchSearchRequest,
    RetrievalResultDTO,
)
from .ability_descriptor import AbilityDescriptor
from .registry_service import (
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
)

__all__ = [
    # parser
    "ParserService",
    "ParseRequest",
    "ParseResponse",
    # chunker
    "ChunkerService",
    "ChunkRequest",
    "ChunkDTO",
    # storage
    "StorageService",
    "CreateCollectionRequest",
    "DeleteCollectionRequest",
    "GetCollectionRequest",
    "ExistsCollectionRequest",
    "ListCollectionsRequest",
    "InsertRequest",
    # retriever
    "RetrieverService",
    "BaseSearchRequest",
    "SemanticSearchRequest",
    "KeywordSearchRequest",
    "HybridSearchRequest",
    "FulltextSearchRequest",
    "TextMatchSearchRequest",
    "PhraseMatchSearchRequest",
    "RetrievalResultDTO",
    # descriptor
    "AbilityDescriptor",
    # registry
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
]

