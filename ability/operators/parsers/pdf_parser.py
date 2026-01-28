"""
PDF文档解析器
"""
from pathlib import Path
from typing import Any, Dict

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ability.operators.parsers.base_parser import BaseParser


class PDFParser(BaseParser):
    """PDF文档解析器，使用PyMuPDF"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化PDF解析器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.supported_extensions = [".pdf"]
        self.extract_images = self.get_config("extract_images", False)
        self.extract_tables = self.get_config("extract_tables", True)

        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install it with: pip install PyMuPDF"
            )

    def _initialize(self) -> None:
        """初始化PDF解析器"""
        if fitz is None:
            raise ImportError("PyMuPDF is not installed")
        self.logger.info("PDF parser initialized")

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        解析PDF文档

        Args:
            file_path: PDF文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典
        """
        doc = fitz.open(str(file_path))
        content_parts = []
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "page_count": len(doc),
        }

        structure = {
            "pages": [],
            "tables": [],
            "images": [],
        }

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                # 提取页面结构
                page_structure = {
                    "page_number": page_num + 1,
                    "blocks": [],
                }

                # 提取文本块
                blocks = page.get_text("blocks")
                for block in blocks:
                    if block[6] == 0:  # 文本块
                        page_structure["blocks"].append(
                            {
                                "type": "text",
                                "bbox": block[:4],
                                "text": block[4],
                            }
                        )

                structure["pages"].append(page_structure)

                # 添加页面内容
                if page_text.strip():
                    content_parts.append(f"## Page {page_num + 1}\n\n{page_text}\n")

                # 提取表格（如果启用）
                if self.extract_tables:
                    try:
                        tables = page.find_tables()
                        for table_idx, table in enumerate(tables):
                            table_data = table.extract()
                            if table_data:
                                # 转换为Markdown表格
                                markdown_table = self._table_to_markdown(table_data)
                                content_parts.append(f"\n### Table {table_idx + 1} (Page {page_num + 1})\n\n{markdown_table}\n\n")
                                structure["tables"].append(
                                    {
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "rows": len(table_data),
                                        "cols": len(table_data[0]) if table_data else 0,
                                    }
                                )
                    except Exception as e:
                        self.logger.warning(f"Failed to extract table from page {page_num + 1}: {str(e)}")

                # 提取图片（如果启用）
                if self.extract_images:
                    try:
                        image_list = page.get_images()
                        for img_idx, img in enumerate(image_list):
                            structure["images"].append(
                                {
                                    "page": page_num + 1,
                                    "image_index": img_idx,
                                    "xref": img[0],
                                }
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to extract images from page {page_num + 1}: {str(e)}")

        finally:
            doc.close()

        content = "\n".join(content_parts)

        return {
            "content": content,
            "metadata": metadata,
            "structure": structure,
        }

    def _table_to_markdown(self, table_data: list) -> str:
        """
        将表格数据转换为Markdown格式

        Args:
            table_data: 表格数据（二维列表）

        Returns:
            Markdown表格字符串
        """
        if not table_data:
            return ""

        markdown_lines = []

        # 表头
        if table_data:
            header = table_data[0]
            markdown_lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
            markdown_lines.append("| " + " | ".join("---" for _ in header) + " |")

        # 表体
        for row in table_data[1:]:
            markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(markdown_lines)
