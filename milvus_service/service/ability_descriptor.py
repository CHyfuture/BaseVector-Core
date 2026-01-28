"""
AbilityDescriptor

提供对各类算子能力的自描述能力，便于：
- 上层服务/网关根据该信息生成文档或前端配置
- 快速查看当前系统支持哪些能力
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from ability.operators.chunkers.chunker_factory import ChunkerFactory
from ability.operators.parsers.parser_factory import ParserFactory
from ability.operators.retrievers.retriever_factory import RetrieverFactory
from ability.operators.storage.storage_factory import StorageFactory


@dataclass
class AbilityInfo:
    """单个能力的信息描述"""

    name: str
    category: str
    description: str
    request_model: str
    extra: Dict[str, Any] | None = None


class AbilityDescriptor:
    """汇总各类能力元信息"""

    @staticmethod
    def list_parser_abilities() -> List[Dict[str, Any]]:
        """返回解析能力描述列表"""
        exts = ParserFactory.get_supported_extensions()
        info = AbilityInfo(
            name="document_parser",
            category="parser",
            description=f"文档解析能力，支持扩展名：{', '.join(sorted(exts))}",
            request_model="ParseRequest",
            extra={"extensions": sorted(exts)},
        )
        return [asdict(info)]

    @staticmethod
    def list_chunker_abilities() -> List[Dict[str, Any]]:
        """返回切片能力描述列表"""
        strategies = ChunkerFactory.get_supported_strategies()
        info = AbilityInfo(
            name="text_chunker",
            category="chunker",
            description=f"文本切片能力，支持策略：{', '.join(sorted(strategies))}",
            request_model="ChunkRequest",
            extra={"strategies": sorted(strategies)},
        )
        return [asdict(info)]

    @staticmethod
    def list_storage_abilities() -> List[Dict[str, Any]]:
        """返回存储能力描述列表"""
        operations = StorageFactory.get_supported_operations()
        abilities: List[AbilityInfo] = []

        abilities.append(
            AbilityInfo(
                name="collection_operations",
                category="storage",
                description="集合管理能力（create/delete/list/get/exists）",
                request_model=(
                    "CreateCollectionRequest / DeleteCollectionRequest / "
                    "GetCollectionRequest / ExistsCollectionRequest / ListCollectionsRequest"
                ),
                extra={"operations": operations},
            )
        )

        if "insert" in operations:
            abilities.append(
                AbilityInfo(
                    name="insert",
                    category="storage",
                    description="数据写入能力（批量插入向量与文本等字段）",
                    request_model="InsertRequest",
                    extra={"operation": "insert"},
                )
            )

        return [asdict(a) for a in abilities]

    @staticmethod
    def list_retriever_abilities() -> List[Dict[str, Any]]:
        """返回检索能力描述列表"""
        modes = RetrieverFactory.get_supported_modes()
        abilities: List[AbilityInfo] = []

        if "semantic" in modes:
            abilities.append(
                AbilityInfo(
                    name="semantic_search",
                    category="retriever",
                    description="语义检索（向量相似度），需要外部提供查询向量",
                    request_model="SemanticSearchRequest",
                    extra={"mode": "semantic"},
                )
            )
        if "keyword" in modes:
            abilities.append(
                AbilityInfo(
                    name="keyword_search",
                    category="retriever",
                    description="关键词检索（BM25 风格）",
                    request_model="KeywordSearchRequest",
                    extra={"mode": "keyword"},
                )
            )
        if "hybrid" in modes:
            abilities.append(
                AbilityInfo(
                    name="hybrid_search",
                    category="retriever",
                    description="混合检索（语义 + 关键词融合）",
                    request_model="HybridSearchRequest",
                    extra={"mode": "hybrid"},
                )
            )
        if "fulltext" in modes:
            abilities.append(
                AbilityInfo(
                    name="fulltext_search",
                    category="retriever",
                    description="全文检索（LIKE 等方式）",
                    request_model="FulltextSearchRequest",
                    extra={"mode": "fulltext"},
                )
            )
        if "text_match" in modes:
            abilities.append(
                AbilityInfo(
                    name="text_match_search",
                    category="retriever",
                    description="文本匹配检索（支持精确/模糊匹配）",
                    request_model="TextMatchSearchRequest",
                    extra={"mode": "text_match"},
                )
            )
        if "phrase_match" in modes:
            abilities.append(
                AbilityInfo(
                    name="phrase_match_search",
                    category="retriever",
                    description="短语匹配检索",
                    request_model="PhraseMatchSearchRequest",
                    extra={"mode": "phrase_match"},
                )
            )

        return [asdict(a) for a in abilities]

    @staticmethod
    def list_all_abilities() -> Dict[str, List[Dict[str, Any]]]:
        """返回所有能力的汇总信息，按类别组织"""
        return {
            "parser": AbilityDescriptor.list_parser_abilities(),
            "chunker": AbilityDescriptor.list_chunker_abilities(),
            "storage": AbilityDescriptor.list_storage_abilities(),
            "retriever": AbilityDescriptor.list_retriever_abilities(),
        }

