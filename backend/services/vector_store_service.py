import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
import hashlib
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG  # Updated import
from pypinyin import lazy_pinyin, Style
from pymilvus import MilvusClient, exceptions
import chromadb
import re
from utils.paths import VECTOR_STORE_DIR, CHROMADB_DIR
import faiss
import numpy as np

chromadb_path = str(CHROMADB_DIR)
logger = logging.getLogger(__name__)


def clean_filename(file_name: str) -> str:
    """
    清理文件名，满足以下要求：
    1. 仅保留[a-zA-Z0-9._-]和中文字符，其余字符替换为下划线
    2. 确保文件名以[a-zA-Z0-9]开头和结尾
    3. 合并连续的下划线

    :param file_name: 原始文件名
    :return: 符合要求的清理后文件名
    """
    # 处理空字符串情况
    if not file_name:
        return "default_filename"

    # 步骤1：替换非法字符为下划线
    # 允许的字符：a-zA-Z0-9._- 和中文字符（\u4e00-\u9fa5）
    # 注意：将'-'放在字符集最后避免被解释为范围符号
    cleaned = re.sub(r"[^a-zA-Z0-9._\u4e00-\u9fa5-]", "_", file_name)

    # 步骤2：合并连续的下划线
    cleaned = re.sub(r"_+", "_", cleaned)

    # 步骤3：确保以字母或数字开头
    # 如果开头不是字母或数字，添加默认前缀"file_"
    if not re.match(r"^[a-zA-Z]", cleaned):
        cleaned = "file_" + cleaned
        # 再次合并可能产生的连续下划线
        cleaned = re.sub(r"_+", "_", cleaned)

    # 步骤4：确保以字母或数字结尾
    # 如果结尾不是字母或数字，添加默认后缀"_file"
    if not re.search(r"[a-zA-Z0-9]$", cleaned):
        cleaned += "_file"
        # 再次合并可能产生的连续下划线
        cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned


def build_collection_name(
    filename: str, embedding_provider: str, timestamp: str, max_length: int = 63
) -> str:
    """
    生成满足 Chroma 命名约束的集合名称。

    规则：
    1. 只保留字母、数字、下划线和连字符
    2. 总长度不超过 max_length
    3. 以字母数字开头和结尾
    4. 使用短哈希保留长文件名场景下的唯一性
    """
    raw_base_name = filename.replace(".pdf", "") if filename else "doc"
    pinyin_base_name = "".join(lazy_pinyin(raw_base_name, style=Style.NORMAL))
    normalized_base_name = clean_filename(pinyin_base_name).lower().replace(".", "_")
    normalized_provider = (
        clean_filename(embedding_provider or "unknown").lower().replace(".", "_")
    )

    normalized_base_name = re.sub(r"[^a-z0-9_-]", "_", normalized_base_name)
    normalized_provider = re.sub(r"[^a-z0-9_-]", "_", normalized_provider)

    normalized_base_name = normalized_base_name.strip("_-") or "doc"
    normalized_provider = normalized_provider.strip("_-") or "unknown"

    hash_suffix = hashlib.md5(
        f"{raw_base_name}_{normalized_provider}".encode("utf-8")
    ).hexdigest()[:8]
    fixed_suffix = f"_{normalized_provider}_{hash_suffix}_{timestamp}"

    available_base_length = max_length - len(fixed_suffix)
    if available_base_length < 3:
        raise ValueError("Collection name suffix is too long to build a valid name")

    trimmed_base_name = (
        normalized_base_name[:available_base_length].rstrip("_-") or "doc"
    )
    collection_name = f"{trimmed_base_name}{fixed_suffix}"
    collection_name = re.sub(r"^[^a-z0-9]+", "", collection_name)
    collection_name = re.sub(r"[^a-z0-9]+$", "", collection_name)

    if not collection_name:
        collection_name = f"doc_{hash_suffix}_{timestamp}"

    return collection_name[:max_length]


class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """

    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置

        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chromadb_uri = chromadb_path
        # 新增 FAISS 路径配置
        self.faiss_dir = VECTOR_STORE_DIR / "faiss"
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型

        参数:
            index_mode: 索引模式

        返回:
            对应的Milvus索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")

    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数

        参数:
            index_mode: 索引模式

        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

    def _get_chroma_index_type(self) -> str:
        """

        返回:
            对应的chroma索引类型
        """
        return self.index_mode


class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """

    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        # 确保存储目录存在
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # 连接到chroma
        self.client = chromadb.PersistentClient(chromadb_path)

    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)

    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)

    def index_embeddings(
        self, embedding_file: str, config: VectorDBConfig
    ) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库

        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        start_time = datetime.now()

        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)

        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)

        if config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)

        if config.provider == VectorDBProvider.FAISS:  # 新增 FAISS 分支
            result = self._index_to_faiss(embeddings_data, config)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A"),
        }

    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings

        参数:
            file_path: 嵌入向量文件路径

        返回:
            包含嵌入向量和元数据的字典
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")

                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError(
                        "Invalid embedding file format: missing 'embeddings' key"
                    )

                # 返回完整的数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data

        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise

    def _index_to_milvus(
        self, embeddings_data: Dict[str, Any], config: VectorDBConfig
    ) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus数据库

        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        try:
            filename = embeddings_data.get("filename", "")
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = build_collection_name(
                filename, embedding_provider, timestamp, max_length=63
            )

            # 连接到Milvus

            client = MilvusClient(
                uri="http://localhost:19530",
                token="root:Milvus",
                db_name=config.milvus_uri,
            )

            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")

            logger.info(f"Creating collection with dimension: {vector_dim}")

            # 定义字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 10000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": self._get_milvus_index_params(config),
                },
            ]

            # 准备数据为列表格式
            entities = []
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get(
                        "filename", ""
                    ),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    # "chunking_method": str(emb["metadata"].get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get(
                        "embedding_provider", ""
                    ),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get(
                        "embedding_model", ""
                    ),  # 从顶层配置获取
                    "embedding_timestamp": str(
                        emb["metadata"].get("embedding_timestamp", "")
                    ),
                    "vector": [float(x) for x in emb.get("embedding", [])],
                }
                entities.append(entity)

            logger.info(f"Creating Milvus collection: {collection_name}")

            # 创建collection
            # field_schemas = [
            #     FieldSchema(name=field["name"],
            #                dtype=getattr(DataType, field["dtype"]),
            #                is_primary="is_primary" in field and field["is_primary"],
            #                auto_id="auto_id" in field and field["auto_id"],
            #                max_length=field.get("max_length"),
            #                dim=field.get("dim"),
            #                params=field.get("params"))
            #     for field in fields
            # ]

            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get("max_length") is not None:
                    extra_params["max_length"] = field["max_length"]
                if field.get("dim") is not None:
                    extra_params["dim"] = field["dim"]
                if field.get("params") is not None:
                    extra_params["params"] = field["params"]
                field_schema = FieldSchema(
                    name=field["name"],
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params,
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(
                fields=field_schemas, description=f"Collection for {collection_name}"
            )
            #  collection = Collection(name=collection_name, schema=schema)
            collection = client.create_collection(
                collection_name=collection_name, schema=schema
            )

            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insert_result = client.insert(
                collection_name=collection_name, data=entities
            )

            # 创建索引
            index_params = client.prepare_index_params()

            index_params.add_index(
                field_name="vector",
                metric_type="IP",
                index_type="IVF_FLAT",
                params={"nlist": 1280},
                #       params=self._get_milvus_index_params(config)
            )
            logger.info(f"create index")
            client.create_index(
                collection_name=collection_name, index_params=index_params
            )
            client.load_collection(collection_name=collection_name)
            logger.info(f"after load \n {insert_result}")

            return {
                "index_size": len(insert_result["ids"]),
                "collection_name": collection_name,
            }

        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise

        finally:
            connections.disconnect("default")

    def _index_to_chroma(
        self, embeddings_data: Dict[str, Any], config: VectorDBConfig
    ) -> Dict[str, Any]:
        """
        将嵌入向量索引到chroma数据库

        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        try:
            filename = embeddings_data.get("filename", "")
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = build_collection_name(
                filename, embedding_provider, timestamp, max_length=63
            )
            logger.info(f"Filename: {collection_name}")

            # collection = self.client.create_collection(
            #     name=collection_name,
            #     metadata={"hnsw:space": "cosine"}
            # )
            # 先删除已有的同名Collection，确保重新创建
            try:
                self.client.delete_collection(collection_name)
            except:
                pass

            # 创建/获取Collection，指定余弦相似度作为度量方式
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # 关键：指定使用余弦相似度空间
            )

            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")

            logger.info(f"Creating collection with dimension: {vector_dim}")

            # 定义字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 10000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": config._get_chroma_index_type(),
                },
            ]

            # 准备数据为列表格式
            entities = []
            entity_num = 0
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "document_name": embeddings_data.get(
                        "filename", ""
                    ),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    # "chunking_method": str(emb["metadata"].get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get(
                        "embedding_provider", ""
                    ),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get(
                        "embedding_model", ""
                    ),  # 从顶层配置获取
                    "embedding_timestamp": str(
                        emb["metadata"].get("embedding_timestamp", "")
                    ),
                    "index_mode": str(config._get_chroma_index_type()),
                }
                entities.append(entity)
                collection.add(
                    documents=[str(emb["metadata"].get("content", ""))],  # 文本描述
                    metadatas=[entity],
                    embeddings=[[float(x) for x in emb.get("embedding", [])]],
                    ids=[str(int(emb["metadata"].get("chunk_id", 0)))],
                )
                entity_num += 1

            logger.info(f"Creating CHROMA collection: {collection_name}")

            return {"index_size": entity_num, "collection_name": collection_name}

        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise

        # finally:
        #     connections.disconnect("default")

    def _index_to_faiss(
            self, embeddings_data: Dict[str, Any], config: VectorDBConfig
    ) -> Dict[str, Any]:
        """
        将嵌入向量索引到本地 FAISS 数据库
        """
        try:
            filename = embeddings_data.get("filename", "")
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = build_collection_name(
                filename, embedding_provider, timestamp, max_length=63
            )

            # 准备向量数据
            embeddings_list = []
            metadata_list = []

            for emb in embeddings_data["embeddings"]:
                embeddings_list.append(emb.get("embedding", []))
                # 记录每一个向量对应的文本元数据
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get("filename", ""),
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                }
                metadata_list.append(entity)

            xb = np.array(embeddings_list).astype('float32')
            vector_dim = xb.shape[1]

            # 根据选中的索引模式创建不同的 FAISS 索引
            index_mode = config.index_mode.upper()
            if index_mode == "FLAT":
                index = faiss.IndexFlatL2(vector_dim)
            elif index_mode == "IVF_FLAT":
                nlist = min(100, len(xb))  # 聚类中心数量
                quantizer = faiss.IndexFlatL2(vector_dim)
                index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
                index.train(xb)  # IVF 索引需要训练
            elif index_mode == "HNSW":
                M = 32  # 邻居数量
                index = faiss.IndexHNSWFlat(vector_dim, M)
            else:
                logger.warning(f"Unknown FAISS index mode {index_mode}, fallback to FLAT")
                index = faiss.IndexFlatL2(vector_dim)

            # 添加向量到索引
            index.add(xb)

            # 创建本地存储路径
            faiss_dir = VECTOR_STORE_DIR / "faiss"
            faiss_dir.mkdir(parents=True, exist_ok=True)

            # 1. 写入物理索引文件
            index_file_path = faiss_dir / f"{collection_name}.index"
            faiss.write_index(index, str(index_file_path))

            # 2. 写入对应的元数据JSON文件
            metadata_file_path = faiss_dir / f"{collection_name}_metadata.json"
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)

            logger.info(f"Successfully created FAISS collection: {collection_name} with mode {index_mode}")
            return {"index_size": len(xb), "collection_name": collection_name}

        except Exception as e:
            logger.error(f"Error indexing to FAISS: {str(e)}")
            raise

    def list_collections(self, provider: str) -> List[str]:
        """
        列出指定提供商的所有集合

        参数:
            provider: 向量数据库提供商

        返回:
            集合名称列表
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collections = utility.list_collections()
                return collections
            finally:
                connections.disconnect("default")

        if provider == VectorDBProvider.CHROMA:
            collections = self.client.list_collections()
            return collections

        if provider == VectorDBProvider.FAISS:  # 新增 FAISS 扫描
            faiss_dir = VECTOR_STORE_DIR / "faiss"
            if not faiss_dir.exists():
                return []
            # 获取所有 .index 文件名
            return [f.stem for f in faiss_dir.glob("*.index")]

        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合

        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称

        返回:
            是否删除成功
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                return True
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                self.client.delete_collection(name=collection_name)
                return True
            finally:
                logger.info("after delete collection")
        elif provider == VectorDBProvider.FAISS:  # 新增 FAISS 删除
            try:
                faiss_dir = VECTOR_STORE_DIR / "faiss"
                index_file = faiss_dir / f"{collection_name}.index"
                meta_file = faiss_dir / f"{collection_name}_metadata.json"
                if index_file.exists():
                    index_file.unlink()
                if meta_file.exists():
                    meta_file.unlink()
                return True
            except Exception as e:
                logger.error(f"FAISS delete error: {e}")
                return False
        return False

    def get_collection_info(
        self, provider: str, collection_name: str
    ) -> Dict[str, Any]:
        """
        获取指定集合的信息

        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称

        返回:
            集合信息字典
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collection = Collection(collection_name)
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": collection.schema.to_dict(),
                }
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                collection = self.client.get_collection(name=collection_name)
                full_data = collection.get()
                return_info = {
                    "name": collection_name,
                    "num_entities": len(full_data["metadatas"]),
                    "schema": full_data["metadatas"][0],
                }
                logger.info(str(return_info))
                return {
                    "name": collection_name,
                    "num_entities": len(full_data["metadatas"]),
                    "schema": full_data["metadatas"][0],
                }
            finally:
                logger.info("after get collection info")
                # 【新增：FAISS 集合信息读取逻辑】
        elif provider == VectorDBProvider.FAISS or provider == "faiss":
            try:
                faiss_dir = VECTOR_STORE_DIR / "faiss"
                meta_file = faiss_dir / f"{collection_name}_metadata.json"

                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    return {
                        "name": collection_name,
                        "num_entities": len(meta_data),  # 元数据数组的长度就是向量总数
                        "schema": meta_data[0] if meta_data else {},  # 返回第一个向量的元数据作为 Schema 示例
                        "processing_time": 0.0  # 占位符
                    }
                else:
                    logger.error(f"FAISS metadata file not found: {meta_file}")
            except Exception as e:
                logger.error(f"Error getting FAISS collection info: {str(e)}")
                raise

        return {}
