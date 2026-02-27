"""
文档解析器工厂
"""
from pathlib import Path
from typing import Dict, Optional

from ability.config import get_settings
from ability.operators.parsers.base_parser import BaseParser
from ability.operators.parsers.docx_parser import DocxParser
from ability.operators.parsers.html_parser import HTMLParser
from ability.operators.parsers.markdown_parser import MarkdownParser
from ability.operators.parsers.mineru_parser import MinerUParser
from ability.operators.parsers.pdf_parser import PDFParser
from ability.operators.parsers.pptx_parser import PPTXParser
from ability.operators.parsers.txt_parser import TXTParser
from ability.operators.plugin_registry import PluginRegistry
from ability.utils.logger import logger

# 可由 MinerU 解析的格式（启用时优先使用 MinerUParser）
MINERU_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}

# 内置解析器映射（MinerU 未启用时使用）
BUILTIN_PARSER_REGISTRY: Dict[str, type[BaseParser]] = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".doc": DocxParser,
    ".pptx": PPTXParser,
    ".ppt": PPTXParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": TXTParser,
}


def _get_builtin_parsers() -> Dict[str, type[BaseParser]]:
    """根据 MINERU_ENABLED 返回实际使用的内置解析器映射。"""
    settings = get_settings()
    if getattr(settings, "MINERU_ENABLED", True):
        reg = dict(BUILTIN_PARSER_REGISTRY)
        for ext in MINERU_EXTENSIONS:
            reg[ext] = MinerUParser
        return reg
    return BUILTIN_PARSER_REGISTRY


class ParserFactory:
    """文档解析器工厂类"""

    @staticmethod
    def create_parser(file_path: Path | str, config: Optional[Dict] = None) -> BaseParser:
        """
        根据文件扩展名创建对应的解析器

        Args:
            file_path: 文件路径
            config: 解析器配置

        Returns:
            解析器实例

        Raises:
            ValueError: 如果文件格式不支持
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # 合并内置和插件注册表
        all_parsers = {**_get_builtin_parsers(), **PluginRegistry.get_parsers()}

        if extension not in all_parsers:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(all_parsers.keys())}"
            )

        parser_class = all_parsers[extension]
        parser = parser_class(config=config)
        parser.initialize()

        logger.info(f"Created parser {parser_class.__name__} for {file_path}")
        return parser

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """
        获取所有支持的文件扩展名（包括插件）

        Returns:
            扩展名列表
        """
        all_parsers = {**_get_builtin_parsers(), **PluginRegistry.get_parsers()}
        return list(all_parsers.keys())

    @staticmethod
    def is_supported(file_path: Path | str) -> bool:
        """
        检查文件格式是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        file_path = Path(file_path)
        all_parsers = {**_get_builtin_parsers(), **PluginRegistry.get_parsers()}
        return file_path.suffix.lower() in all_parsers
