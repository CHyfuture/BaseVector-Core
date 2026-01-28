"""
ParserService

对 ability 中的文档解析算子做封装，统一暴露：
- 解析能力
- 所需入参模型
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ability.operators.parsers.parser_factory import ParserFactory


class ParseRequest(BaseModel):
    """
    文档解析请求模型（入参）

    字段说明：
    - file_path: 待解析文件的本地路径，支持相对或绝对路径；必须是 ParserFactory 支持的扩展名
    - parser_config: 解析器配置，将原样透传给具体 Parser 的 config 参数，用于控制解析细节
    """

    file_path: str = Field(
        ...,
        description=(
            "待解析文件路径，支持相对或绝对路径；扩展名必须在 "
            "ParserService.get_supported_extensions() 返回列表中"
        ),
    )
    parser_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "解析器配置，将透传给具体 Parser 的 config 参数；不同 Parser "
            "可能支持的键不同（例如是否提取元数据、是否保留换行等）"
        ),
    )


class ParseResponse(BaseModel):
    """
    文档解析响应模型（返参）

    字段说明：
    - content: 解析后的完整文本内容，类型为 str
    - metadata: 与文档相关的元数据字典，例如文件名、页码信息、标题结构等
    - raw: 如果底层 Parser 返回了除 content/metadata 之外的额外字段，这些字段会放在 raw 中
    """

    content: str = Field(..., description="解析后的文档纯文本内容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="解析得到的元数据，例如 source/file_name/page_ranges/title_hierarchy 等键",
    )
    raw: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "底层 Parser 原始返回的完整字典（如果存在），用于保留除 content/metadata 之外的所有字段；"
            "如果底层只返回了 content/metadata，则该字段为 None"
        ),
    )


class ParserService:
    """文档解析 service，封装 ParserFactory"""

    @staticmethod
    def parse(request: ParseRequest) -> ParseResponse:
        """
        解析文档

        Args:
            request: 文档解析请求模型 ParseRequest

        Returns:
            ParseResponse:
                - content: 文本内容（str）
                - metadata: 元数据（dict[str, Any]）
                - raw: 原始结果字典（可选）
        """
        file_path = Path(request.file_path)
        parser = ParserFactory.create_parser(file_path=file_path, config=request.parser_config)
        result = parser.process(file_path)

        # 统一转换为标准结构
        if isinstance(result, dict):
            content = str(result.get("content", ""))
            metadata = result.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            return ParseResponse(content=content, metadata=metadata, raw=result)

        # 兼容性处理：如果 Parser 返回的是自定义对象，尝试转换为 dict
        if hasattr(result, "to_dict"):
            raw_dict = result.to_dict()
            content = str(raw_dict.get("content", ""))
            metadata = raw_dict.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            return ParseResponse(content=content, metadata=metadata, raw=raw_dict)

        # 最小兜底：仅有 content
        return ParseResponse(content=str(result), metadata={}, raw=None)

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """返回当前系统支持解析的文件扩展名列表"""
        return ParserFactory.get_supported_extensions()

