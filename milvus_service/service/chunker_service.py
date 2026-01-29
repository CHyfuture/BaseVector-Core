"""
ChunkerService

对 ability 中的切片算子做封装，统一暴露：
- 文本切片能力
- 所需入参模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ability.config import get_settings
from ability.operators.chunkers.base_chunker import Chunk
from ability.operators.chunkers.chunker_factory import ChunkerFactory


settings = get_settings()


class ChunkRequest(BaseModel):
    """
    文本切片请求模型（入参）

    字段说明：
    - text: 待切分的原始文本内容，不能为空字符串
    - strategy: 切片策略，默认使用全局配置 CHUNK_STRATEGY；
                可通过 ChunkerService.get_supported_strategies() 查看支持的值
    - chunk_size: 切片大小（字符数），不填则使用 Settings.CHUNK_SIZE
    - chunk_overlap: 相邻切片重叠字符数，不填则使用 Settings.CHUNK_OVERLAP
    - extra_config: 额外配置，将与上述默认配置合并后传递给具体 Chunker
                    （例如语义切片可能需要 sentence_delimiters 等参数）
    """

    text: str = Field(..., description="待切分的原始文本内容，必须为非空字符串")
    strategy: Optional[str] = Field(
        default=None,
        description=(
            "切片策略，默认使用全局配置 Settings.CHUNK_STRATEGY（默认值：'fixed'）；"
            "支持的策略名称可通过 ChunkerService.get_supported_strategies() 获取"
        ),
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="切片大小（字符数），不填则使用全局配置 Settings.CHUNK_SIZE（默认值：512）",
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="相邻切片重叠字符数，不填则使用全局配置 Settings.CHUNK_OVERLAP（默认值：50）",
    )
    extra_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "额外配置，将与 chunk_size/chunk_overlap 合并后传给具体 Chunker；"
            "不同策略可能使用不同的键，例如 whitespace_pattern/sentence_delimiters 等"
        ),
    )


class ChunkDTO(BaseModel):
    """
    文档块数据模型（返参）

    字段说明：
    - content: 当前块的文本内容
    - chunk_index: 当前块在整体切片结果中的索引（从 0 或 1 开始，取决于具体 Chunker 实现）
    - start_index: 当前块在原始文本中的起始字符位置（闭区间起点）
    - end_index: 当前块在原始文本中的结束字符位置（闭区间终点）
    - metadata: 与当前块相关的任意元数据字典（例如所在段落标题、层级信息等）
    - parent_chunk_id: 父块 ID，仅在父子切片策略下使用，否则通常为 None
    """

    content: str = Field(..., description="当前块的文本内容")
    chunk_index: int = Field(..., description="块索引（由具体切片器定义起始值）")
    start_index: int = Field(..., description="块在原始文本中的起始字符位置")
    end_index: int = Field(..., description="块在原始文本中的结束字符位置")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="块级元数据，例如所属标题、页码、层级结构等信息",
    )
    parent_chunk_id: Optional[int] = Field(
        default=None,
        description="父块 ID，仅在 parent_child 等策略下有值，其他策略通常为 None",
    )


class ChunkerService:
    """文本切片 service，封装 ChunkerFactory"""

    @staticmethod
    def chunk(request: ChunkRequest) -> List[ChunkDTO]:
        """
        切片文本

        Args:
            request: 文本切片请求模型 ChunkRequest

        Returns:
            List[ChunkDTO]:
                每个元素对应一个切片块，包含：
                - content / chunk_index / start_index / end_index / metadata / parent_chunk_id
        """
        # 构造 chunker 配置（对非法入参做兜底修正，避免在请求模型阶段直接抛错）
        chunk_size = request.chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = settings.CHUNK_SIZE

        chunk_overlap = request.chunk_overlap
        if chunk_overlap is None or chunk_overlap < 0:
            chunk_overlap = settings.CHUNK_OVERLAP

        config: Dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        if request.extra_config:
            config.update(request.extra_config)

        chunker = ChunkerFactory.create_chunker(strategy=request.strategy, config=config)
        chunks = chunker.process(request.text)

        result: List[ChunkDTO] = []
        for c in chunks:
            if isinstance(c, Chunk):
                data = c.to_dict()
            elif isinstance(c, dict):
                data = c
            else:
                # 最小兜底：仅包含 content
                data = {
                    "content": str(c),
                    "chunk_index": 0,
                    "start_index": 0,
                    "end_index": len(str(c)),
                    "metadata": {},
                    "parent_chunk_id": None,
                }

            result.append(
                ChunkDTO(
                    content=str(data.get("content", "")),
                    chunk_index=int(data.get("chunk_index", 0)),
                    start_index=int(data.get("start_index", 0)),
                    end_index=int(data.get("end_index", 0)),
                    metadata=data.get("metadata") or {},
                    parent_chunk_id=data.get("parent_chunk_id"),
                )
            )

        return result

    @staticmethod
    def get_supported_strategies() -> list[str]:
        """返回当前系统支持的切片策略列表"""
        return ChunkerFactory.get_supported_strategies()

