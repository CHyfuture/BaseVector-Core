"""
Milvus客户端封装
"""
from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

try:
    from pymilvus import Function, FunctionType
    HAS_FUNCTION_API = True
except ImportError:
    HAS_FUNCTION_API = False
    Function = None
    FunctionType = None

from ability.config import get_settings
from ability.utils.logger import logger

settings = get_settings()


class MilvusClient:
    """Milvus客户端封装类"""

    def __init__(self):
        """初始化Milvus客户端（默认从全局 Settings 读取连接配置）"""
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.user = settings.MILVUS_USER
        self.password = settings.MILVUS_PASSWORD
        self.db_name = settings.MILVUS_DB_NAME
        self.connected = False
        self._collections: Dict[str, Collection] = {}

    def update_connection_config(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """
        动态更新 Milvus 连接配置。

        说明：
        - 仅在传入参数非 None 时覆盖对应字段；
        - 更新配置后会将当前连接标记为未连接，下一次使用前会自动重新 connect。
        """
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if user is not None:
            self.user = user
        if password is not None:
            self.password = password
        if db_name is not None:
            self.db_name = db_name

        # 标记需要重新连接
        if self.connected:
            try:
                connections.disconnect("default")
            except Exception:
                # 忽略断开异常，直接重置状态
                pass
        self.connected = False
        self._collections.clear()

    def connect(self) -> None:
        """连接到Milvus服务器"""
        if self.connected:
            return

        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user if self.user else None,
                password=self.password if self.password else None,
            )
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def disconnect(self) -> None:
        """断开Milvus连接"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False

    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        description: str = "",
        auto_id: bool = True,
        primary_field: str = "id",
        dense_vector_field: Optional[str] = None,
        sparse_vector_field: Optional[str] = None,
        primary_dtype: DataType = DataType.INT64,
        metadata_fields: Optional[List[FieldSchema]] = None,
        varchar_max_length: int = 255,
        dense_index_params: Optional[Dict[str, Any]] = None,
        sparse_index_params: Optional[Dict[str, Any]] = None,
        analyzer_params: Optional[str] = None,
    ) -> Collection:
        """
        创建集合
        
        支持动态选择是否创建稠密向量字段和/或稀疏向量字段（BM25）。
        用户可以根据需求选择：
        - 只创建稠密向量字段（用于语义检索）
        - 只创建稀疏向量字段（用于BM25检索）
        - 同时创建两种向量字段（用于混合检索）
        - 都不创建（仅存储文本和元数据）
        
        字段创建说明（根据 Milvus Schema 文档）：
        - 稠密向量字段 (FLOAT_VECTOR)：根据 dense_vector_field 参数动态创建，必须指定 dim 维度
        - 稀疏向量字段 (SPARSE_FLOAT_VECTOR)：根据 sparse_vector_field 参数动态创建，用于存储 BM25 稀疏向量
        - 文本字段 (VARCHAR)：如果用于 BM25 Function，需要 enable_analyzer=True 和可选的 analyzer_params

        Args:
            collection_name: 集合名称
            dimension: 稠密向量维度（当 dense_vector_field 不为 None 时必需）
            description: 集合描述
            auto_id: 是否自动生成ID
            primary_field: 主键字段名
            dense_vector_field: 稠密向量字段名（FLOAT_VECTOR类型），如果为 None 则不创建稠密向量字段
            sparse_vector_field: 稀疏向量字段名（SPARSE_FLOAT_VECTOR类型，用于BM25），如果为 None 则不创建稀疏向量字段
            primary_dtype: 主键字段类型（默认INT64）
            metadata_fields: 额外的元数据字段列表
            varchar_max_length: VARCHAR字段的最大长度
            dense_index_params: 稠密向量索引参数字典，支持自定义索引类型和参数。如果为None，则使用全局配置。
                格式示例：
                - IVF_FLAT: {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
                - IVF_SQ8: {"metric_type": "L2", "index_type": "IVF_SQ8", "params": {"nlist": 1024}}
                - IVF_PQ: {"metric_type": "L2", "index_type": "IVF_PQ", "params": {"nlist": 1024, "m": 8, "nbits": 8}}
                - HNSW: {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
                - FLAT: {"metric_type": "L2", "index_type": "FLAT"}
            sparse_index_params: 稀疏向量索引参数字典，如果为None，则使用默认配置。
                稀疏向量通常使用 SPARSE_INVERTED_INDEX 索引类型。
                如果使用 BM25 Function，metric_type 必须为 "BM25"。
            analyzer_params: 分词器参数（可选），例如 "english" 或 "chinese"，用于配置 BM25 Function 的文本分词器

        Returns:
            Collection对象
            
        Raises:
            ValueError: 如果 dense_vector_field 不为 None 但 dimension 为 None
        """
        if not self.connected:
            self.connect()

        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            
            # 如果集合已存在，检查是否需要创建稀疏向量索引
            if sparse_vector_field is not None:
                # 检查稀疏向量字段是否存在
                field_names = [f.name for f in collection.schema.fields]
                if sparse_vector_field in field_names:
                    # 检查索引是否存在
                    try:
                        indexes = collection.indexes
                        has_sparse_index = any(
                            idx.field_name == sparse_vector_field 
                            for idx in indexes
                        )
                        if not has_sparse_index:
                            # 为稀疏向量字段创建索引
                            if sparse_index_params is None:
                                sparse_index_params = {
                                    "metric_type": "IP",
                                    "index_type": "SPARSE_INVERTED_INDEX",
                                }
                            if "metric_type" not in sparse_index_params:
                                sparse_index_params.setdefault("metric_type", "IP")
                            collection.create_index(
                                field_name=sparse_vector_field,
                                index_params=sparse_index_params,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to check/create sparse vector index: {str(e)}")
                        # 如果字段存在但索引创建失败，继续执行（可能索引已存在）
            
            return collection

        # 验证参数（允许两者都不创建：仅存储文本与元数据）
        if dense_vector_field is not None and dimension is None:
            raise ValueError(
                "dimension is required when dense_vector_field is specified"
            )

        # 定义字段
        fields = []

        # 主键字段
        if auto_id:
            fields.append(
                FieldSchema(
                    name=primary_field,
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                )
            )
        else:
            # 允许外部指定主键（如：Chunk.id），默认使用INT64；若用VARCHAR则需max_length
            if primary_dtype == DataType.VARCHAR:
                fields.append(
                    FieldSchema(
                        name=primary_field,
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        auto_id=False,
                        max_length=varchar_max_length,
                    )
                )
            else:
                fields.append(
                    FieldSchema(
                        name=primary_field,
                        dtype=primary_dtype,
                        is_primary=True,
                        auto_id=False,
                    )
                )

        # 稠密向量字段（可选）
        if dense_vector_field is not None:
            fields.append(
                FieldSchema(
                    name=dense_vector_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension,
                )
            )

        # 稀疏向量字段（可选，Milvus 2.4+ 支持）
        if sparse_vector_field is not None:
            # 使用 SPARSE_FLOAT_VECTOR 类型（Milvus 2.4+）
            if not hasattr(DataType, "SPARSE_FLOAT_VECTOR"):
                raise ValueError(
                    "SPARSE_FLOAT_VECTOR is not supported in this Milvus version. "
                    "Please upgrade to Milvus 2.4+ or set sparse_vector_field=None"
                )
            
            fields.append(
                FieldSchema(
                    name=sparse_vector_field,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                )
            )

        # 元数据字段
        # 如果创建稀疏向量字段，需要检查是否有 content 字段并启用 analyzer
        # 根据 Milvus Schema 文档：文本字段需要 enable_analyzer=True 以支持 BM25 Function
        content_field_name = None
        if metadata_fields:
            # 检查是否有 content 字段
            for field in metadata_fields:
                if field.name == "content" and field.dtype == DataType.VARCHAR:
                    # 如果创建稀疏向量字段，需要启用 analyzer
                    if sparse_vector_field is not None:
                        # 创建一个新的 FieldSchema，启用 analyzer 以支持 BM25 Function
                        # 根据 Milvus Schema 文档：enable_analyzer=True 允许 Milvus 进行分词处理
                        content_field_kwargs = {
                            "name": field.name,
                            "dtype": field.dtype,
                            "max_length": field.max_length if hasattr(field, 'max_length') else 65535,
                            "enable_analyzer": True,  # 启用 analyzer 以支持 BM25 Function
                        }
                        # 如果指定了 analyzer_params（如 "english" 或 "chinese"），添加到字段定义中
                        if analyzer_params:
                            content_field_kwargs["analyzer_params"] = analyzer_params
                        
                        content_field = FieldSchema(**content_field_kwargs)
                        # 替换原来的字段
                        fields.append(content_field)
                        content_field_name = field.name
                        continue
                fields.append(field)

        # 创建Schema
        schema = CollectionSchema(
            fields=fields,
            description=description,
        )

        # 创建集合
        collection = Collection(
            name=collection_name,
            schema=schema,
        )

        # 如果创建稀疏向量字段且存在 content 字段，创建 BM25 Function
        bm25_function_created = False
        if sparse_vector_field is not None and content_field_name:
            if HAS_FUNCTION_API:
                try:
                    # 创建 BM25 Function，将 content 字段映射到 sparse_vector 字段
                    bm25_function = Function(
                        function_type=FunctionType.BM25,
                        input_field_names=[content_field_name],
                        output_field_names=[sparse_vector_field],
                    )
                    collection.create_function(bm25_function)
                    bm25_function_created = True
                except Exception as e:
                    logger.warning(
                        f"Failed to create BM25 Function (Milvus may not support Function API): {e}"
                    )
                    bm25_function_created = False
            else:
                logger.warning(
                    "Function API not available. Please upgrade pymilvus to version 2.5+ "
                    "or manually create BM25 Function."
                )
                bm25_function_created = False

        # 为稠密向量字段创建索引（可选）
        if dense_vector_field is not None:
            if dense_index_params is None:
                # 使用全局配置作为默认值
                dense_index_params = {
                    "metric_type": settings.MILVUS_METRIC_TYPE,
                    "index_type": settings.MILVUS_INDEX_TYPE,
                }
                # 根据索引类型设置默认参数
                index_type = settings.MILVUS_INDEX_TYPE.upper()
                if index_type == "FLAT":
                    # FLAT索引不需要params
                    pass
                elif index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
                    dense_index_params["params"] = {"nlist": settings.MILVUS_NLIST}
                elif index_type == "HNSW":
                    # HNSW默认参数
                    dense_index_params["params"] = {"M": 16, "efConstruction": 200}
                else:
                    # 其他索引类型使用nlist作为默认参数
                    dense_index_params["params"] = {"nlist": settings.MILVUS_NLIST}
            
            # 验证索引参数格式
            if not isinstance(dense_index_params, dict):
                raise ValueError("dense_index_params must be a dictionary")
            if "metric_type" not in dense_index_params:
                raise ValueError("dense_index_params must contain 'metric_type'")
            if "index_type" not in dense_index_params:
                raise ValueError("dense_index_params must contain 'index_type'")
            
            collection.create_index(
                field_name=dense_vector_field,
                index_params=dense_index_params,
            )

        # 为稀疏向量字段创建索引（可选）
        if sparse_vector_field is not None:
            if sparse_index_params is None:
                # 稀疏向量默认使用 SPARSE_INVERTED_INDEX
                # 只有当 BM25 Function 成功创建时，才使用 "BM25" metric_type
                # 否则使用 IP（内积）作为度量类型
                sparse_index_params = {
                    "metric_type": "BM25" if bm25_function_created else "IP",
                    "index_type": "SPARSE_INVERTED_INDEX",
                }
            
            # 验证索引参数格式
            if not isinstance(sparse_index_params, dict):
                raise ValueError("sparse_index_params must be a dictionary")
            if "metric_type" not in sparse_index_params:
                # 如果没有指定 metric_type，默认使用 BM25（如果 BM25 Function 成功创建）或 IP
                sparse_index_params.setdefault("metric_type", "BM25" if bm25_function_created else "IP")
            if "index_type" not in sparse_index_params:
                raise ValueError("sparse_index_params must contain 'index_type'")
            
            # 如果 metric_type 是 "BM25" 但 BM25 Function 未成功创建，则改为 "IP"
            if sparse_index_params.get("metric_type") == "BM25" and not bm25_function_created:
                logger.warning(
                    f"Cannot use BM25 metric type without BM25 Function. "
                    f"Falling back to IP metric type for field '{sparse_vector_field}'"
                )
                sparse_index_params["metric_type"] = "IP"
            
            collection.create_index(
                field_name=sparse_vector_field,
                index_params=sparse_index_params,
            )

        self._collections[collection_name] = collection
        return collection

    def get_collection(self, collection_name: str) -> Collection:
        """
        获取集合对象

        Args:
            collection_name: 集合名称

        Returns:
            Collection对象
        """
        if not self.connected:
            self.connect()

        if collection_name in self._collections:
            return self._collections[collection_name]

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            self._collections[collection_name] = collection
            return collection

        raise ValueError(f"Collection {collection_name} does not exist")

    def insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> List[int]:
        """
        插入数据到Milvus集合
        
        根据 Milvus 插入文档：
        - 如果集合配置了 BM25 Function，插入时只需要提供文本字段（如 content）
          Milvus 会自动调用 BM25 Function 生成稀疏向量并填充到稀疏向量字段
        - 如果集合包含稀疏向量字段但数据中未提供，会自动填充空字典 {}，
          Milvus 会根据 BM25 Function（如果存在）自动处理
        
        数据格式说明：
        - 每个字典必须包含集合schema中定义的所有必需字段
        - 稠密向量字段：通常是 "vector"（List[float]），需要用户自行使用嵌入模型生成
        - 文本字段：通常是 "content"（str），如果配置了 BM25 Function，只需提供文本即可
        - 稀疏向量字段：通常由 BM25 Function 自动生成，如果未提供会自动填充空字典 {}
        - 其他元数据字段根据schema定义
        
        Args:
            collection_name: 集合名称
            data: 数据列表，每个元素是一个字典，包含：
                - vector（可选）: 稠密向量数据（List[float]），如果集合包含稠密向量字段则必需
                - content（必需）: 文本内容（str），如果配置了 BM25 Function，只需提供文本
                - sparse_vector（可选）: 稀疏向量数据（Dict[int, float]），如果集合包含稀疏向量字段且未配置 BM25 Function 则必需
                - 其他schema中定义的字段（如document_id、tenant_id、metadata等）

        Returns:
            插入的ID列表

        示例:
            # 示例1: 使用稠密向量（需要用户自行使用嵌入模型生成向量）
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
            
            texts = ["文档1", "文档2"]
            vectors = model.encode(texts).tolist()
            
            data = [
                {
                    "vector": vectors[0],
                    "content": texts[0],
                    "document_id": 1,
                    "tenant_id": "tenant001",
                    "metadata": '{"key": "value"}'
                },
                {
                    "vector": vectors[1],
                    "content": texts[1],
                    "document_id": 1,
                    "tenant_id": "tenant001",
                    "metadata": '{"key": "value"}'
                }
            ]
            ids = milvus_client.insert("collection_name", data)
            
            # 示例2: 使用 BM25 Function（只需提供文本，稀疏向量自动生成）
            # 集合必须配置 BM25 Function，将 content 字段映射到 sparse_vector 字段
            data = [
                {
                    "vector": vectors[0],  # 稠密向量（可选）
                    "content": texts[0],    # 文本字段，BM25 Function 会自动生成稀疏向量
                    # sparse_vector 字段会自动填充（由 BM25 Function 生成）
                    "document_id": 1,
                    "tenant_id": "tenant001",
                }
            ]
            ids = milvus_client.insert("collection_name", data)
            
            # 示例3: 手动提供稀疏向量（不使用 BM25 Function）
            # 如果集合包含稀疏向量字段但未配置 BM25 Function，需要手动提供
            data = [
                {
                    "vector": vectors[0],
                    "sparse_vector": {0: 0.5, 5: 0.3, 10: 0.8},  # 手动计算的稀疏向量
                    "content": texts[0],
                    "document_id": 1,
                }
            ]
            ids = milvus_client.insert("collection_name", data)
        """
        collection = self.get_collection(collection_name)
        collection.load()

        # 按schema字段顺序准备数据，避免字段顺序不一致导致插入错误
        schema_fields = [f.name for f in collection.schema.fields]
        
        # 检查是否有稀疏向量字段
        sparse_vector_field = None
        for field in collection.schema.fields:
            # 直接比较 DataType 枚举，更可靠
            if hasattr(field, 'dtype'):
                field_dtype_str = str(field.dtype)
                field_dtype_repr = repr(field.dtype)
                
                # 检查是否是 SPARSE_FLOAT_VECTOR 类型
                if hasattr(DataType, 'SPARSE_FLOAT_VECTOR'):
                    if field.dtype == DataType.SPARSE_FLOAT_VECTOR:
                        sparse_vector_field = field.name
                        break
                
                # 也检查字符串表示（向后兼容）
                if field_dtype_str.upper() in ('SPARSE_FLOAT_VECTOR', 'SPARSE_FLOAT_VECTOR_TYPE', '113'):
                    sparse_vector_field = field.name
                    break
                
                # 检查 dtype 的值（SPARSE_FLOAT_VECTOR 的值通常是 113）
                if hasattr(field.dtype, 'value') and field.dtype.value == 113:
                    sparse_vector_field = field.name
                    break
        
        insert_columns: Dict[str, List[Any]] = {name: [] for name in schema_fields}

        # 根据 Milvus 插入文档：如果配置了 BM25 Function，只需提供文本字段即可
        # 稀疏向量字段会自动填充（如果未提供，填充空字典 {}，由 BM25 Function 处理）
        for item_idx, item in enumerate(data):
            for name in schema_fields:
                if name not in item:
                    # 如果存在稀疏向量字段且缺少sparse_vector字段，自动添加空字典
                    # 根据 Milvus 文档：BM25 Function 会自动填充稀疏向量，插入时可以提供一个空字典
                    if sparse_vector_field and name == sparse_vector_field:
                        # BM25 Function会自动填充稀疏向量，插入时可以提供一个空字典 {}
                        insert_columns[name].append({})
                    else:
                        # 详细错误信息，包括检测到的稀疏向量字段
                        error_msg = (
                            f"Missing field '{name}' for Milvus insert into {collection_name}. "
                            f"Provided keys: {list(item.keys())}. "
                            f"Required fields: {schema_fields}"
                        )
                        if sparse_vector_field:
                            error_msg += f" (sparse_vector_field detected: '{sparse_vector_field}')"
                        raise ValueError(error_msg)
                else:
                    # 验证稀疏向量字段的数据格式
                    if sparse_vector_field and name == sparse_vector_field and item[name] is not None:
                        # 如果手动提供了稀疏向量，确保格式正确（Dict[int, float]）
                        if not isinstance(item[name], dict):
                            logger.warning(
                                f"Sparse vector field '{name}' should be a dict (Dict[int, float]), "
                                f"got {type(item[name])}. If using BM25 Function, you can omit this field."
                            )
                    insert_columns[name].append(item[name])

        result = collection.insert([insert_columns[name] for name in schema_fields])
        collection.flush()

        return result.primary_keys

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        top_k: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        anns_field: str = "vector",
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            collection_name: 集合名称
            vectors: 查询向量列表
            top_k: 返回Top-K结果
            expr: 过滤表达式（可选）
            output_fields: 返回的字段列表（可选）
            search_params: 搜索参数字典，支持自定义搜索参数。如果为None，则使用全局配置。
                格式示例：
                - IVF系列: {"metric_type": "L2", "params": {"nprobe": 10}}
                - HNSW: {"metric_type": "L2", "params": {"ef": 100}}
                - FLAT: {"metric_type": "L2"}  (FLAT不需要params)

        Returns:
            检索结果列表
        """
        collection = self.get_collection(collection_name)
        collection.load()

        # 使用自定义搜索参数，如果不提供则使用全局配置
        if search_params is None:
            search_params = {
                "metric_type": settings.MILVUS_METRIC_TYPE,
                "params": {"nprobe": settings.MILVUS_NPROBE},
            }
        else:
            # 验证搜索参数格式
            if not isinstance(search_params, dict):
                raise ValueError("search_params must be a dictionary")
            if "metric_type" not in search_params:
                # 如果没有指定metric_type，使用全局配置
                search_params["metric_type"] = settings.MILVUS_METRIC_TYPE

        results = collection.search(
            data=vectors,
            anns_field=anns_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields or [],
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                # 兼容不同pymilvus版本：entity可能是dict-like，也可能需要to_dict
                # 在pymilvus中，output_fields指定的字段可能通过hit.entity访问，也可能直接作为hit的属性
                entity_payload: Dict[str, Any] = {}
                try:
                    # 方法1: 尝试从 hit.entity 获取（优先方法）
                    if hasattr(hit, "entity") and hit.entity is not None:
                        entity = hit.entity
                        # 尝试多种方式获取字段
                        if hasattr(entity, "to_dict"):
                            try:
                                entity_payload = entity.to_dict()
                            except:
                                pass
                        if not entity_payload and isinstance(entity, dict):
                            entity_payload = entity.copy()
                        if not entity_payload and hasattr(entity, "__dict__"):
                            # 如果entity是对象，尝试从__dict__获取
                            entity_payload = entity.__dict__.copy()
                        # 尝试通过属性访问字段（即使entity_payload已有内容也要尝试，因为可能只有部分字段）
                        if output_fields:
                            for field_name in output_fields:
                                if field_name not in entity_payload or entity_payload.get(field_name) is None:
                                    if hasattr(entity, field_name):
                                        try:
                                            value = getattr(entity, field_name)
                                            entity_payload[field_name] = value  # 即使为None也添加
                                        except:
                                            pass
                        # 如果entity有get方法，尝试使用get方法
                        if hasattr(entity, "get"):
                            try:
                                if output_fields:
                                    for field_name in output_fields:
                                        if field_name not in entity_payload or entity_payload.get(field_name) is None:
                                            try:
                                                value = entity.get(field_name)
                                                entity_payload[field_name] = value  # 即使为None也添加
                                            except:
                                                pass
                            except:
                                pass
                    
                    # 方法2: 如果字段仍然缺失，尝试直接从hit对象获取字段（某些版本pymilvus可能直接将字段作为属性）
                    if output_fields:
                        for field_name in output_fields:
                            if field_name not in entity_payload or entity_payload.get(field_name) is None:
                                # 尝试从 hit 对象直接获取字段
                                if hasattr(hit, field_name):
                                    try:
                                        value = getattr(hit, field_name)
                                        entity_payload[field_name] = value  # 即使为None也添加
                                    except:
                                        pass
                                # 如果仍然没有，检查是否有 get 方法
                                if field_name not in entity_payload and hasattr(hit, "get"):
                                    try:
                                        value = hit.get(field_name)
                                        entity_payload[field_name] = value  # 即使为None也添加
                                    except:
                                        pass
                    
                    # 方法3: 尝试从 hit.__dict__ 获取所有字段（最后备选）
                    if output_fields:
                        hit_dict = getattr(hit, "__dict__", {})
                        for field_name in output_fields:
                            if field_name not in entity_payload or entity_payload.get(field_name) is None:
                                if field_name in hit_dict:
                                    entity_payload[field_name] = hit_dict[field_name]
                    else:
                        # 如果没有指定output_fields，尝试获取所有非标准字段
                        if hasattr(hit, "__dict__"):
                            hit_dict = hit.__dict__
                            for key, value in hit_dict.items():
                                if key not in ("id", "distance", "score", "entity") and key not in entity_payload:
                                    entity_payload[key] = value
                                    
                except Exception as e:
                    entity_payload = {}

                result_dict = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance),  # 转换为相似度分数
                    **entity_payload,
                }
                
                # 确保所有output_fields都存在（即使值为None）
                if output_fields:
                    for field_name in output_fields:
                        if field_name not in result_dict:
                            result_dict[field_name] = None
                
                formatted_results.append(result_dict)

        return formatted_results

    def delete_collection(self, collection_name: str) -> None:
        """
        删除集合

        Args:
            collection_name: 集合名称
        """
        if not self.connected:
            self.connect()

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]

    def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            集合名称列表
        """
        if not self.connected:
            self.connect()

        return utility.list_collections()

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


# 全局Milvus客户端实例
milvus_client = MilvusClient()
