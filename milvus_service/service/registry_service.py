"""
RegistryService

将 ability 中的插件注册器（PluginRegistry）以 service 形式暴露出来，统一管理：
- 自定义解析器（parsers）
- 自定义切片器（chunkers）
- 自定义检索器（retrievers）
- 自定义存储算子（storage operators）

主要能力：
- 显式注册 / 取消注册某个算子类
- 从指定 Python 文件加载插件（自动扫描并注册符合约定的算子类）
- 查询当前所有已注册的算子信息（包括类全限定名）
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ability.operators.plugin_registry import PluginRegistry


PluginCategory = Literal["parser", "chunker", "retriever", "storage"]


class PluginInfo(BaseModel):
    """
    已注册插件算子的信息（返参）

    字段说明：
    - key: 注册键
        * 对解析器：文件扩展名（例如 ".pdf"、".md"）
        * 对切片器：切片策略名（例如 "fixed"、"semantic"、"custom_strategy"）
        * 对检索器：检索模式名（例如 "semantic"、"hybrid"、"fulltext"）
        * 对存储算子：操作名（例如 "collection"、"insert"）
    - category: 插件类别（parser / chunker / retriever / storage）
    - qualified_name: 类的全限定名，形如 "package.module.ClassName"
    """

    key: str = Field(..., description="插件注册键（扩展名/策略名/模式名/操作名）")
    category: PluginCategory = Field(..., description="插件类别：parser/chunker/retriever/storage")
    qualified_name: str = Field(..., description='算子类的全限定名，例如 "mypkg.mymodule.MyParser"')


class LoadPluginRequest(BaseModel):
    """
    从文件加载插件（入参）

    字段说明：
    - plugin_path: 插件 Python 文件路径，支持相对或绝对路径；
                   文件内部的 BaseOperator 子类需按约定定义 _extension/_strategy/_mode/_operation 属性
    """

    plugin_path: str = Field(
        ...,
        description=(
            "插件 Python 文件路径，支持相对或绝对路径。"
            "文件内部的 BaseOperator 子类需定义 _extension/_strategy/_mode/_operation 中的一个，"
            "PluginRegistry 会据此自动完成注册。"
        ),
    )


class _BaseRegisterRequest(BaseModel):
    """
    注册请求基类（入参）

    所有显式注册类请求都包含以下字段：
    - module: 模块名，例如 "mypkg.mymodule"
    - class_name: 类名，例如 "MyCustomParser"

    service 会通过 importlib.import_module(module) 并 getattr 获取类对象，然后调用 PluginRegistry 注册。
    """

    module: str = Field(..., description='Python 模块名，例如 "mypkg.mymodule"')
    class_name: str = Field(..., description='类名，例如 "MyCustomParser"')


class RegisterParserRequest(_BaseRegisterRequest):
    """
    注册解析器请求（入参）

    字段说明：
    - extension: 文件扩展名，例如 ".pdf"、".custom"
    - module: 模块名（继承 BaseParser 的类所在模块）
    - class_name: 类名（继承 BaseParser 的类名）
    """

    extension: str = Field(..., description='文件扩展名，例如 ".pdf"、".md"、".custom"')


class RegisterChunkerRequest(_BaseRegisterRequest):
    """
    注册切片器请求（入参）

    字段说明：
    - strategy: 切片策略名称，例如 "fixed"、"semantic"、"my_custom_strategy"
    - module: 模块名（继承 BaseChunker 的类所在模块）
    - class_name: 类名（继承 BaseChunker 的类名）
    """

    strategy: str = Field(..., description='切片策略名称，例如 "fixed" 或 "my_custom_strategy"')


class RegisterRetrieverRequest(_BaseRegisterRequest):
    """
    注册检索器请求（入参）

    字段说明：
    - mode: 检索模式名称，例如 "semantic"、"hybrid"、"my_custom_mode"
    - module: 模块名（继承 BaseRetriever 的类所在模块）
    - class_name: 类名（继承 BaseRetriever 的类名）
    """

    mode: str = Field(..., description='检索模式名称，例如 "semantic" 或 "custom_mode"')


class RegisterStorageOperatorRequest(_BaseRegisterRequest):
    """
    注册存储算子请求（入参）

    字段说明：
    - operation: 操作类型名称，例如 "collection"、"insert"、"custom_op"
    - module: 模块名（继承 BaseStorageOperator 的类所在模块）
    - class_name: 类名（继承 BaseStorageOperator 的类名）
    """

    operation: str = Field(..., description='操作类型名称，例如 "collection" 或 "my_custom_op"')


class UnregisterRequest(BaseModel):
    """
    取消注册请求（入参）

    字段说明：
    - category: 插件类别：parser/chunker/retriever/storage
    - key: 注册键：
        * parser: 扩展名（如 ".md"）
        * chunker: 策略名（如 "fixed"）
        * retriever: 模式名（如 "semantic"）
        * storage: 操作名（如 "insert"）
    """

    category: PluginCategory = Field(..., description="要取消注册的插件类别")
    key: str = Field(..., description="要取消注册的键（扩展名/策略名/模式名/操作名）")


class UnregisterResponse(BaseModel):
    """
    取消注册响应（返参）

    字段说明：
    - success: 是否成功取消注册
    - category: 插件类别
    - key: 取消注册的键
    """

    success: bool = Field(..., description="是否成功取消注册")
    category: PluginCategory = Field(..., description="插件类别")
    key: str = Field(..., description="取消注册的键")


class ClearAllResponse(BaseModel):
    """
    清空所有插件响应（返参）

    字段说明：
    - cleared: 是否已执行清空操作（总为 True）
    - remaining_counts: 清空后各类别剩余的注册数量（正常情况下均为 0）
    """

    cleared: bool = Field(..., description="是否已执行清空操作")
    remaining_counts: Dict[PluginCategory, int] = Field(
        ...,
        description="清空后各类别剩余的注册数量，键为 parser/chunker/retriever/storage",
    )


class RegistrySnapshot(BaseModel):
    """
    当前注册表快照（返参）

    字段说明：
    - parsers: 所有解析器插件信息列表
    - chunkers: 所有切片器插件信息列表
    - retrievers: 所有检索器插件信息列表
    - storage_operators: 所有存储算子插件信息列表
    """

    parsers: List[PluginInfo] = Field(..., description="所有已注册解析器插件信息列表")
    chunkers: List[PluginInfo] = Field(..., description="所有已注册切片器插件信息列表")
    retrievers: List[PluginInfo] = Field(..., description="所有已注册检索器插件信息列表")
    storage_operators: List[PluginInfo] = Field(..., description="所有已注册存储算子插件信息列表")


class RegistryService:
    """插件注册相关 service，封装 PluginRegistry"""

    # ========== 查询能力 ==========

    @staticmethod
    def list_parsers() -> List[PluginInfo]:
        """列出所有已注册的解析器插件"""
        result: List[PluginInfo] = []
        for ext, cls in PluginRegistry.get_parsers().items():
            result.append(
                PluginInfo(
                    key=ext,
                    category="parser",
                    qualified_name=f"{cls.__module__}.{cls.__name__}",
                )
            )
        return result

    @staticmethod
    def list_chunkers() -> List[PluginInfo]:
        """列出所有已注册的切片器插件"""
        result: List[PluginInfo] = []
        for strategy, cls in PluginRegistry.get_chunkers().items():
            result.append(
                PluginInfo(
                    key=strategy,
                    category="chunker",
                    qualified_name=f"{cls.__module__}.{cls.__name__}",
                )
            )
        return result

    @staticmethod
    def list_retrievers() -> List[PluginInfo]:
        """列出所有已注册的检索器插件"""
        result: List[PluginInfo] = []
        for mode, cls in PluginRegistry.get_retrievers().items():
            result.append(
                PluginInfo(
                    key=mode,
                    category="retriever",
                    qualified_name=f"{cls.__module__}.{cls.__name__}",
                )
            )
        return result

    @staticmethod
    def list_storage_operators() -> List[PluginInfo]:
        """列出所有已注册的存储算子插件"""
        result: List[PluginInfo] = []
        for op, cls in PluginRegistry.get_storage_operators().items():
            result.append(
                PluginInfo(
                    key=op,
                    category="storage",
                    qualified_name=f"{cls.__module__}.{cls.__name__}",
                )
            )
        return result

    @staticmethod
    def snapshot() -> RegistrySnapshot:
        """获取当前所有插件注册信息的整体快照"""
        return RegistrySnapshot(
            parsers=RegistryService.list_parsers(),
            chunkers=RegistryService.list_chunkers(),
            retrievers=RegistryService.list_retrievers(),
            storage_operators=RegistryService.list_storage_operators(),
        )

    # ========== 显式注册能力 ==========

    @staticmethod
    def register_parser(request: RegisterParserRequest) -> PluginInfo:
        """显式注册解析器"""
        module = importlib.import_module(request.module)
        cls = getattr(module, request.class_name)
        PluginRegistry.register_parser(request.extension, cls)
        return PluginInfo(
            key=request.extension.lower(),
            category="parser",
            qualified_name=f"{cls.__module__}.{cls.__name__}",
        )

    @staticmethod
    def register_chunker(request: RegisterChunkerRequest) -> PluginInfo:
        """显式注册切片器"""
        module = importlib.import_module(request.module)
        cls = getattr(module, request.class_name)
        PluginRegistry.register_chunker(request.strategy, cls)
        return PluginInfo(
            key=request.strategy,
            category="chunker",
            qualified_name=f"{cls.__module__}.{cls.__name__}",
        )

    @staticmethod
    def register_retriever(request: RegisterRetrieverRequest) -> PluginInfo:
        """显式注册检索器"""
        module = importlib.import_module(request.module)
        cls = getattr(module, request.class_name)
        PluginRegistry.register_retriever(request.mode, cls)
        return PluginInfo(
            key=request.mode,
            category="retriever",
            qualified_name=f"{cls.__module__}.{cls.__name__}",
        )

    @staticmethod
    def register_storage_operator(request: RegisterStorageOperatorRequest) -> PluginInfo:
        """显式注册存储算子"""
        module = importlib.import_module(request.module)
        cls = getattr(module, request.class_name)
        PluginRegistry.register_storage_operator(request.operation, cls)
        return PluginInfo(
            key=request.operation,
            category="storage",
            qualified_name=f"{cls.__module__}.{cls.__name__}",
        )

    # ========== 取消注册 / 清空 ==========

    @staticmethod
    def unregister(request: UnregisterRequest) -> UnregisterResponse:
        """根据类别和键取消注册某个插件"""
        if request.category == "parser":
            success = PluginRegistry.unregister_parser(request.key)
        elif request.category == "chunker":
            success = PluginRegistry.unregister_chunker(request.key)
        elif request.category == "retriever":
            success = PluginRegistry.unregister_retriever(request.key)
        else:
            success = PluginRegistry.unregister_storage_operator(request.key)

        return UnregisterResponse(success=success, category=request.category, key=request.key)

    @staticmethod
    def clear_all() -> ClearAllResponse:
        """清空所有插件注册表（请谨慎调用）"""
        PluginRegistry.clear_all()
        snapshot = RegistryService.snapshot()
        remaining = {
            "parser": len(snapshot.parsers),
            "chunker": len(snapshot.chunkers),
            "retriever": len(snapshot.retrievers),
            "storage": len(snapshot.storage_operators),
        }
        return ClearAllResponse(cleared=True, remaining_counts=remaining)

    # ========== 从插件文件加载 ==========

    @staticmethod
    def load_plugin(request: LoadPluginRequest) -> RegistrySnapshot:
        """
        从指定 Python 文件加载插件，并返回加载完成后的注册表快照。
        """
        plugin_path = Path(request.plugin_path)
        PluginRegistry.load_plugin(plugin_path)
        return RegistryService.snapshot()

