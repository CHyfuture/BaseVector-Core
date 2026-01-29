"""
StorageService

对 ability 中的存储算子做封装，统一暴露：
- 集合管理能力（create/delete/list/get/exists）
- 数据写入能力（insert）
- 所需入参模型
"""

from typing import Any, Dict, List, Optional

try:
    # Pydantic v2
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    # Pydantic v1 fallback
    from pydantic import BaseModel, Field

    ConfigDict = None  # type: ignore[assignment,misc]
from pymilvus import Collection, DataType, FieldSchema

from ability.config import get_settings
from ability.operators.storage.storage_factory import StorageFactory
from ability.storage.milvus_client import milvus_client


settings = get_settings()


class MilvusConnectionParams(BaseModel):
    """
    Milvus 连接配置（可选入参）

    说明：
    - 如果不提供，则使用全局 Settings 中的 MILVUS_HOST / MILVUS_PORT / MILVUS_USER / MILVUS_PASSWORD / MILVUS_DB_NAME；
    - 如果提供任意字段，则会在本次调用前覆盖全局客户端的对应字段。
    """

    host: Optional[str] = Field(
        default=None,
        description="Milvus 服务主机名或 IP，例如 'localhost' 或 '10.0.0.2'",
    )
    port: Optional[int] = Field(
        default=None,
        description="Milvus 服务端口号，例如 19530",
    )
    user: Optional[str] = Field(
        default=None,
        description="Milvus 用户名，如果未启用鉴权可以留空",
    )
    password: Optional[str] = Field(
        default=None,
        description="Milvus 密码，如果未启用鉴权可以留空",
    )
    db_name: Optional[str] = Field(
        default=None,
        description="Milvus 数据库名称，默认通常为 'default'",
    )


class CreateCollectionRequest(BaseModel):
    """
    创建集合请求

    字段说明（与 CollectionOperator._create_collection 保持一致）：
    - collection_name: 集合名称，不填时由底层根据配置决定名称
    - dimension: 稠密向量维度（必填），即向量字段中每个向量的长度
    - description: 集合描述信息
    - auto_id: 是否由 Milvus 自动生成主键 ID
    - primary_field: 主键字段名
    - dense_vector_field: 稠密向量字段名，例如 'vector'
    - sparse_vector_field: 稀疏向量字段名，例如 'sparse_vector'
    - primary_dtype: 主键字段类型
    - metadata_fields: 额外的元数据字段定义列表（FieldSchema 数组）
    - varchar_max_length: VARCHAR 类型字段默认最大长度
    - dense_index_params: 稠密向量索引参数
    - sparse_index_params: 稀疏向量索引参数
    - analyzer_params: 文本分词器参数（用于 BM25 Function）
    - milvus_connection: 本次操作使用的 Milvus 连接配置（可选）
    """

    collection_name: Optional[str] = Field(
        default=None,
        description="集合名称，不填则由底层根据 MILVUS_COLLECTION_TEMPLATE 等配置生成",
    )
    dimension: int = Field(..., ge=1, description="稠密向量维度，必填")
    description: str = Field(
        default="",
        description="集合描述信息",
    )
    auto_id: bool = Field(
        default=True,
        description="是否自动生成主键 ID，示例中通常为 False 以便自定义主键",
    )
    primary_field: str = Field(
        default="id",
        description="主键字段名，默认 'id'",
    )
    dense_vector_field: Optional[str] = Field(
        default="vector",
        description="稠密向量字段名，默认 'vector'",
    )
    sparse_vector_field: Optional[str] = Field(
        default=None,
        description="稀疏向量字段名，如需 BM25 Function，可设置例如 'sparse_vector'",
    )
    primary_dtype: DataType = Field(
        default=DataType.INT64,
        description="主键字段类型，默认 INT64",
    )
    metadata_fields: Optional[List[FieldSchema]] = Field(
        default=None,
        description="额外的元数据字段定义列表，例如 document_id/content/metadata 等",
    )
    varchar_max_length: int = Field(
        default=255,
        description="VARCHAR 字段默认最大长度",
    )
    dense_index_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="稠密向量索引参数，未填则由客户端按全局配置创建默认索引",
    )
    sparse_index_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="稀疏向量索引参数",
    )
    analyzer_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="分词器参数，用于 BM25 Function，如 {'tokenizer': 'english'} 或 {'tokenizer': 'chinese'}",
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次创建集合要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )

    # Pydantic v2 会尝试为 FieldSchema 生成 schema；这里允许第三方类型以避免导入阶段失败
    if ConfigDict is not None:  # Pydantic v2
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:  # Pydantic v1
        class Config:  # noqa: D106
            arbitrary_types_allowed = True


class DeleteCollectionRequest(BaseModel):
    """删除集合请求"""

    collection_name: Optional[str] = Field(
        default=None,
        description="集合名称，不填则由底层根据配置生成",
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次删除集合要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )


class GetCollectionRequest(BaseModel):
    """获取集合请求"""

    collection_name: Optional[str] = Field(
        default=None,
        description="集合名称，不填则由底层根据配置生成",
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次获取集合要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )


class ExistsCollectionRequest(BaseModel):
    """检查集合是否存在请求"""

    collection_name: Optional[str] = Field(
        default=None,
        description="集合名称，不填则由底层根据配置生成",
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次存在性检查要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )


class ListCollectionsRequest(BaseModel):
    """列出集合请求（当前实现无过滤）"""

    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次列出集合要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )


class InsertRequest(BaseModel):
    """
    插入数据请求

    字段说明：
    - records: 待插入的数据记录列表，每条记录必须与集合 schema 完全对齐；
               例如示例脚本中常用字段包括：
               * id: 主键 ID（int）
               * vector: 稠密向量（List[float]）
               * document_id: 文档 ID
               * content: 文本内容
               * metadata: 元数据（JSON 字符串或字典）
               * chunk_index: 块序号
               * parent_chunk_id: 父块 ID
               实际字段应与创建集合时传入的 metadata_fields 保持一致
    - collection_name: 目标集合名称，不填时由底层根据默认配置决定
    """

    records: List[Dict[str, Any]] = Field(
        ...,
        description="待插入的数据列表，每条记录是一个字段字典，必须符合集合 schema 定义",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="集合名称，不填则由底层根据配置生成",
    )
    milvus_connection: Optional[MilvusConnectionParams] = Field(
        default=None,
        description="本次插入操作要使用的 Milvus 连接配置，不填则使用全局默认配置",
    )


class StorageService:
    """存储相关 service，封装 CollectionOperator / InsertOperator 等"""

    @staticmethod
    def _apply_milvus_connection(params: Optional[MilvusConnectionParams]) -> None:
        """
        根据请求中提供的 Milvus 连接配置，动态更新全局 Milvus 客户端。

        如果 params 为 None，则保持现有全局配置不变。
        """
        if params is None:
            return

        milvus_client.update_connection_config(
            host=params.host,
            port=params.port,
            user=params.user,
            password=params.password,
            db_name=params.db_name,
        )

    @staticmethod
    def create_collection(request: CreateCollectionRequest) -> Collection:
        """创建集合"""
        StorageService._apply_milvus_connection(request.milvus_connection)
        collection_op = StorageFactory.create_operator("collection", config={})
        # 注意：CreateCollectionRequest 中包含 FieldSchema（第三方类型）。
        # 在 Pydantic v2 的某些环境下，对包含 arbitrary type 的模型调用 model_dump 可能触发序列化异常。
        # 这里直接使用实例字典，避免序列化阶段失败。
        kwargs = dict(request.__dict__)
        kwargs.pop("milvus_connection", None)
        # process 的第一个参数为 operation 类型
        return collection_op.process("create", **kwargs)

    @staticmethod
    def delete_collection(request: DeleteCollectionRequest) -> None:
        """删除集合"""
        StorageService._apply_milvus_connection(request.milvus_connection)
        collection_op = StorageFactory.create_operator("collection", config={})
        kwargs = request.model_dump(exclude={"milvus_connection"})
        collection_op.process("delete", **kwargs)

    @staticmethod
    def drop_collection(collection_name: str) -> None:
        """删除集合（兼容别名：测试/示例中常用 drop_collection）"""
        StorageService.delete_collection(DeleteCollectionRequest(collection_name=collection_name))

    @staticmethod
    def list_collections(request: Optional[ListCollectionsRequest] = None) -> List[str]:
        """列出所有集合（当前不区分租户）"""
        if request is not None:
            StorageService._apply_milvus_connection(request.milvus_connection)
        collection_op = StorageFactory.create_operator("collection", config={})
        return collection_op.process("list")

    @staticmethod
    def get_collection(request: GetCollectionRequest | str) -> Collection:
        """获取集合对象"""
        if isinstance(request, str):
            request = GetCollectionRequest(collection_name=request)
        StorageService._apply_milvus_connection(request.milvus_connection)
        collection_op = StorageFactory.create_operator("collection", config={})
        kwargs = request.model_dump(exclude={"milvus_connection"})
        return collection_op.process("get", **kwargs)

    @staticmethod
    def exists_collection(request: ExistsCollectionRequest) -> bool:
        """检查集合是否存在"""
        StorageService._apply_milvus_connection(request.milvus_connection)
        collection_op = StorageFactory.create_operator("collection", config={})
        kwargs = request.model_dump(exclude={"milvus_connection"})
        return bool(collection_op.process("exists", **kwargs))

    @staticmethod
    def collection_exists(collection_name: str) -> bool:
        """检查集合是否存在（兼容别名：测试/示例中常用 collection_exists）"""
        return StorageService.exists_collection(ExistsCollectionRequest(collection_name=collection_name))

    @staticmethod
    def insert(request: InsertRequest) -> List[int]:
        """插入数据到集合"""
        StorageService._apply_milvus_connection(request.milvus_connection)
        insert_op = StorageFactory.create_operator("insert", config={})
        kwargs = {"collection_name": request.collection_name}
        return insert_op.process(request.records, **kwargs)

