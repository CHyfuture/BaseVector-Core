"""
BaseOperator - 所有算子的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ability.utils.logger import logger


class BaseOperator(ABC):
    """
    算子基类，所有解析器、切片器、向量化器、检索器都应继承此类

    设计原则：
    1. 所有算子都通过继承BaseOperator实现标准接口
    2. 支持配置驱动的初始化
    3. 提供统一的错误处理和日志记录
    4. 支持插件化扩展
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化算子

        Args:
            config: 配置字典，用于自定义算子行为
        """
        self.config = config or {}
        self.logger = logger.bind(operator=self.__class__.__name__)
        self._initialized = False

    def initialize(self) -> None:
        """
        初始化算子（延迟初始化）
        子类可以重写此方法实现自定义初始化逻辑
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")

    def _initialize(self) -> None:
        """
        内部初始化方法，子类可以重写
        """
        pass

    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        处理输入数据（抽象方法，子类必须实现）

        Args:
            input_data: 输入数据
            **kwargs: 额外的处理参数

        Returns:
            处理后的数据
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据（可选重写）

        Args:
            input_data: 输入数据

        Returns:
            验证是否通过
        """
        return input_data is not None

    def validate_output(self, output_data: Any) -> bool:
        """
        验证输出数据（可选重写）

        Args:
            output_data: 输出数据

        Returns:
            验证是否通过
        """
        return output_data is not None

    def __call__(self, input_data: Any, **kwargs) -> Any:
        """
        使算子可调用

        Args:
            input_data: 输入数据
            **kwargs: 额外的处理参数

        Returns:
            处理后的数据
        """
        if not self._initialized:
            self.initialize()

        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input data for {self.__class__.__name__}")

        try:
            result = self.process(input_data, **kwargs)

            if not self.validate_output(result):
                raise ValueError(f"Invalid output data from {self.__class__.__name__}")

            return result
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def update_config(self, **kwargs) -> None:
        """
        更新配置

        Args:
            **kwargs: 要更新的配置项
        """
        self.config.update(kwargs)
        self.logger.debug(f"Config updated: {kwargs}")

    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(config={self.config})"
