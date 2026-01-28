"""
HTML文档解析器
"""
from pathlib import Path
from typing import Any, Dict

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from ability.operators.parsers.base_parser import BaseParser


class HTMLParser(BaseParser):
    """HTML文档解析器，使用BeautifulSoup"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化HTML解析器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.supported_extensions = [".html", ".htm"]

        if BeautifulSoup is None:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install it with: pip install beautifulsoup4"
            )

    def _initialize(self) -> None:
        """初始化HTML解析器"""
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is not installed")
        self.logger.info("HTML parser initialized")

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        解析HTML文档

        Args:
            file_path: HTML文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典
        """
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # 提取元数据
        metadata = {}
        if soup.title:
            metadata["title"] = soup.title.string or ""
        if soup.find("meta", {"name": "author"}):
            metadata["author"] = soup.find("meta", {"name": "author"}).get("content", "")
        if soup.find("meta", {"name": "description"}):
            metadata["description"] = soup.find("meta", {"name": "description"}).get("content", "")

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        # 提取文本内容并转换为Markdown
        content_parts = []
        structure = {
            "headings": [],
            "paragraphs": [],
            "links": [],
        }

        # 提取标题
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])
            text = heading.get_text().strip()
            if text:
                content_parts.append(f"{'#' * level} {text}\n\n")
                structure["headings"].append(
                    {
                        "level": level,
                        "text": text,
                    }
                )

        # 提取段落
        for para in soup.find_all("p"):
            text = para.get_text().strip()
            if text:
                content_parts.append(f"{text}\n\n")
                structure["paragraphs"].append({"text": text})

        # 提取链接
        for link in soup.find_all("a", href=True):
            text = link.get_text().strip()
            href = link.get("href", "")
            if text and href:
                content_parts.append(f"[{text}]({href})\n\n")
                structure["links"].append(
                    {
                        "text": text,
                        "url": href,
                    }
                )

        # 提取列表
        for ul in soup.find_all(["ul", "ol"]):
            items = ul.find_all("li", recursive=False)
            if items:
                for item in items:
                    text = item.get_text().strip()
                    if text:
                        content_parts.append(f"- {text}\n")
                content_parts.append("\n")

        # 提取表格
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if rows:
                table_data = []
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_data = [cell.get_text().strip() for cell in cells]
                    if row_data:
                        table_data.append(row_data)

                if table_data:
                    markdown_table = self._table_to_markdown(table_data)
                    content_parts.append(f"\n{markdown_table}\n\n")

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
