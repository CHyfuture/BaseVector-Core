"""
文档解析器基类
"""
from pathlib import Path
from typing import Any, Dict, Optional

from ability.operators.base import BaseOperator


class BaseParser(BaseOperator):
    """
    文档解析器基类
    所有格式的解析器都应继承此类
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化解析器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.supported_extensions = self.get_config("supported_extensions", [])

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据（文件路径）

        Args:
            input_data: 文件路径（Path或str）

        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            return False

        file_path = Path(input_data)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        if not file_path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            return False

        # 检查文件扩展名（支持带点和不带点的格式）
        extension_with_dot = file_path.suffix.lower()
        extension_without_dot = extension_with_dot.lstrip(".")
        if self.supported_extensions:
            if extension_with_dot not in self.supported_extensions and extension_without_dot not in self.supported_extensions:
                self.logger.error(f"Unsupported file extension: {extension_with_dot}")
                return False

        return True

    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        解析文档（抽象方法，子类必须实现）

        Args:
            input_data: 文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典，包含：
            - content: 文档内容（Markdown格式）
            - metadata: 元数据（标题、作者等）
            - structure: 文档结构信息
        """
        file_path = Path(input_data)
        self.logger.info(f"Parsing document: {file_path}")

        result = self._parse(file_path, **kwargs)

        # 验证输出
        if "content" not in result:
            raise ValueError("Parser result must contain 'content' field")

        self.logger.info(f"Document parsed successfully: {file_path}")
        return result

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        内部解析方法，子类必须实现

        Args:
            file_path: 文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典
        """
        raise NotImplementedError("Subclass must implement _parse method")

    def extract_text(self, file_path: Path) -> str:
        """
        提取纯文本（子类可以重写以优化性能）

        Args:
            file_path: 文件路径

        Returns:
            文本内容
        """
        result = self.process(file_path)
        return result.get("content", "")

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        提取元数据（子类可以重写）

        Args:
            file_path: 文件路径

        Returns:
            元数据字典
        """
        result = self.process(file_path)
        return result.get("metadata", {})
