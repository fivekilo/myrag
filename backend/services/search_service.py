from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import dotenv
from pymilvus import connections, Collection, utility
from openai import OpenAI
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG
import os
import json
from pymilvus import MilvusClient, exceptions
import chromadb
from utils.paths import CHROMADB_DIR, SEARCH_RESULTS_DIR, workspace_relative,VECTOR_STORE_DIR

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

chromadb_path = str(CHROMADB_DIR)

logger = logging.getLogger(__name__)


class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """

    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置Milvus连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.search_results_dir = SEARCH_RESULTS_DIR
        self.search_results_dir.mkdir(parents=True, exist_ok=True)
        self.client=chromadb.PersistentClient(chromadb_path)

    def _expand_query(self, original_query: str) -> List[str]:
        """Use an LLM to create query variants for better retrieval recall."""
        api_key = os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.info("No Aliyun API key found, using original query only")
            return [original_query]

        prompt = f"""你是一个AI助手。你的任务是将用户的问题改写成3个不同角度的搜索查询词，以便更好在向量数据库中检索。
请直接输出查询词，每行一个，不要包含序号、引号或其他说明。
用户原问题: {original_query}"""

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            variants = [line.strip() for line in content.splitlines() if line.strip()]

            queries = []
            seen_queries = set()
            for candidate in [original_query, *variants]:
                normalized = candidate.strip()
                if normalized and normalized not in seen_queries:
                    queries.append(normalized)
                    seen_queries.add(normalized)

            return queries or [original_query]
        except Exception as e:
            logger.error(f"Expand query failed: {e}")
            return [original_query]

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表

        Returns:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """
        return [
            #     {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"}
            {"id": VectorDBProvider.CHROMA.value, "name": "chroma"},
            {"id": "faiss", "name": "FAISS (Local)"}  # 增加本地 FAISS 服务
        ]

    def list_collections(self, provider: str = VectorDBProvider.CHROMA.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合

        Args:
            provider (str): 向量数据库提供商，默认为Milvus

        Returns:
            List[Dict[str, Any]]: 集合信息列表，包含id、名称和实体数量

        Raises:
            Exception: 连接或查询集合时发生错误
        """
        try:
            # client = MilvusClient(
            #     uri="http://localhost:19530",
            #     token="root:Milvus",
            #     db_name=self.milvus_uri
            # )
            logger.info(f"into list collection")

            # 【新增：对 FAISS 提供商进行本地磁盘扫描】
            if provider == "faiss":
                faiss_dir = VECTOR_STORE_DIR / "faiss"
                if not faiss_dir.exists():
                    return []
                collections = []
                for f in faiss_dir.glob("*.index"):
                    name = f.stem
                    meta_file = faiss_dir / f"{name}_metadata.json"
                    count = 0
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as meta_f:
                                meta_data = json.load(meta_f)
                                count = len(meta_data)
                        except Exception as e:
                            logger.error(f"Error reading FAISS metadata: {e}")
                    collections.append({
                        "id": name,
                        "name": name,
                        "count": count
                    })
                return collections

            collections = []
            collection_names = self.client.list_collections()
            print(collection_names)

            for sample in collection_names:
                name = sample if isinstance(sample, str) else getattr(sample, "name", None)
                if not name:
                    continue
                try:
                    collection = self.client.get_or_create_collection(name)
                    collections.append({
                        "id": name,
                        "name": name,
                        "count": collection.count()
                    })
                except Exception as e:
                    logger.error(f"Error getting info for collection {name}: {str(e)}")

            return collections

        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
        # finally:
        #     connections.disconnect("default")

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """
        保存搜索结果到JSON文件

        Args:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表

        Returns:
            str: 保存文件的路径

        Raises:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = self.search_results_dir / filename

            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }

            logger.info(f"Saving search results to: {filepath}")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved search results to: {filepath}")
            return workspace_relative(filepath)

        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise

    async def search(self,
                     query: str,
                     collection_id: str,
                     top_k: int = 3,
                     threshold: float = 0.7,
                     word_count_threshold: int = 20,
                     save_results: bool = False) -> Dict[str, Any]:
        """
        执行向量搜索

        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量，默认为3
            threshold (float): 相似度阈值，低于此值的结果将被过滤，默认为0.7
            word_count_threshold (int): 文本字数阈值，低于此值的结果将被过滤，默认为20
            save_results (bool): 是否保存搜索结果，默认为False

        Returns:
            Dict[str, Any]: 包含搜索结果的字典，如果保存结果则包含保存路径

        Raises:
            Exception: 搜索过程中发生错误
        """
        try:
            # 添加参数日志
            logger.info(f"Search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results} (type: {type(save_results)})")

            logger.info(
                f"Starting search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}")

            # 【新增：本地 FAISS 检索分支】
            faiss_index_path = VECTOR_STORE_DIR / "faiss" / f"{collection_id}.index"
            if faiss_index_path.exists():
                logger.info(f"FAISS index file found, executing native FAISS search in SearchService.")
                import faiss
                import numpy as np

                # 1. 载入本地 FAISS 集合的元数据文件
                meta_file_path = VECTOR_STORE_DIR / "faiss" / f"{collection_id}_metadata.json"
                if not meta_file_path.exists():
                    raise ValueError(f"FAISS metadata not found for {collection_id}")
                with open(meta_file_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

                # 自动提取该集合创建时的 Embedding 配置
                first_meta = meta_data[0] if meta_data else {}
                provider = first_meta.get("embedding_provider", "huggingface")
                model = first_meta.get("embedding_model", "BAAI/bge-small-zh-v1.5")

                # 2. 【核心优化：完美复用服务内已有的多查询扩展（Multi-query Expansion）逻辑】
                queries = self._expand_query(query)
                logger.info(f"FAISS Multi-query expansion variants: {queries}")

                # 3. 读取本地物理索引
                index = faiss.read_index(str(faiss_index_path))
                processed_results = []
                seen_chunk_ids = set()

                # 4. 对多路扩展后的 Query 分别检索、去重和合并
                for expanded_query in queries:
                    query_embedding = self.embedding_service.create_single_embedding(
                        expanded_query,
                        provider=provider,
                        model=model
                    )
                    xq = np.array([query_embedding]).astype('float32')
                    D, I = index.search(xq, top_k)

                    for dist, idx in zip(D[0], I[0]):
                        if idx == -1:
                            continue

                        # 还原余弦相似度（由于 BGE 向量本身已经 L2 归一化，公式为 1 - d/2）
                        hit_score = 1.0 - float(dist) / 2.0
                        if hit_score >= threshold and idx not in seen_chunk_ids:
                            seen_chunk_ids.add(idx)
                            item = meta_data[idx]
                            content = item.get("content", "")

                            processed_results.append({
                                "text": content,
                                "score": float(hit_score),
                                "metadata": {
                                    "source": item.get("document_name"),
                                    "page": item.get("page_number"),
                                    "chunk": int(idx),  # 【修改此处：强转为 Python 标准 int 即可修复】
                                    "total_chunks": item.get("total_chunks"),
                                    "page_range": item.get("page_range"),
                                    "embedding_provider": provider,
                                    "embedding_model": model,
                                    "embedding_timestamp": item.get("embedding_timestamp")
                                }
                            })

                # 5. 全局相似度重排，并截取最相关的 Top K
                processed_results.sort(key=lambda item: item["score"], reverse=True)
                processed_results = processed_results[:top_k]

                response_data = {"results": processed_results}

                # 6. 完美复用结果保存逻辑
                if save_results:
                    logger.info("Save results is True, saving FAISS search results...")
                    if processed_results:
                        try:
                            filepath = self.save_search_results(query, collection_id, processed_results)
                            response_data["saved_filepath"] = filepath
                        except Exception as e:
                            logger.error(f"Error saving FAISS search results: {str(e)}")
                            response_data["save_error"] = str(e)

                return response_data

            # 连接到 Chroma
            # 获取collection
            logger.info(f"Loading collection: {collection_id}")

            collection = self.client.get_collection(collection_id)
            # 记录collection的基本信息
            num_entities=collection.count()
            logger.info(f"Collection info - Entities: {num_entities}")

            logger.info(f"query: {query}")

            sample_entity = collection.peek(limit=1)
            if not sample_entity or not sample_entity.get('metadatas'):
                raise ValueError(f"Collection {collection_id} is empty")

            sample_metadata = sample_entity['metadatas'][0]

            # 使用collection中存储的配置创建查询向量
            queries = self._expand_query(query)
            logger.info(f"Expanded queries: {queries}")

            # 处理结果
            # 连接到 Milvus
            #logger.info(f"Connecting to Milvus at {self.milvus_uri}")
            #connections.connect(
            #    alias="default",
            #    uri=self.milvus_uri
            #)



            # 获取collection
            # logger.info(f"Loading collection: {collection_id}")
            #collection = Collection(collection_id)
            #collection.load()

            # 记录collection的基本信息
            # logger.info(f"Collection info - Entities: {collection.num_entities}")

            # 执行搜索
            # logger.info("Querying sample entity")
            # sample_entity = collection.query(
            #    expr="id >= 0",
            #    output_fields=["embedding_provider", "embedding_model"],
            #    limit=1
            # )

            #
            # if not sample_entity:
            #     logger.error(f"Collection {collection_id} is empty")
            #     raise ValueError(f"Collection {collection_id} is empty")
            #
            # logger.info(f"Sample entity configuration: {sample_entity[0]}")
            #
            # # 使用collection中存储的配置创建查询向量
            # logger.info("Creating query embedding")
            # query_embedding = self.embedding_service.create_single_embedding(
            #     query,
            #     provider=sample_entity[0]["embedding_provider"],
            #     model=sample_entity[0]["embedding_model"]
            # )
            # logger.info(f"Query embedding created with dimension: {len(query_embedding)}")
            #
            # # 执行搜索
            # search_params = {
            #     "metric_type": "COSINE",
            #     "params": {"nprobe": 10}
            # }
            # logger.info(f"Executing search with params: {search_params}")
            # logger.info(f"Word count threshold filter: word_count >= {word_count_threshold}")
            #
            # results = collection.search(
            #     data=[query_embedding],
            #     anns_field="vector",
            #     param=search_params,
            #     limit=top_k,
            #     expr=f"word_count >= {word_count_threshold}",
            #     output_fields=[
            #         "content",
            #         "document_name",
            #         "chunk_id",
            #         "total_chunks",
            #         "word_count",
            #         "page_number",
            #         "page_range",
            #         "embedding_provider",
            #         "embedding_model",
            #         "embedding_timestamp"
            #     ]
            # )

            # 处理结果
            # processed_results = []
            # logger.info(f"Raw search results count: {len(results[0])}")
            #
            # for hits in results:
            #     for hit in hits:
            #         logger.info(f"Processing hit - Score: {hit.score}, Word Count: {hit.entity.get('word_count')}")
            #         if hit.score >= threshold:
            #             processed_results.append({
            #                 "text": hit.entity.content,
            #                 "score": float(hit.score),
            #                 "metadata": {
            #                     "source": hit.entity.document_name,
            #                     "page": hit.entity.page_number,
            #                     "chunk": hit.entity.chunk_id,
            #                     "total_chunks": hit.entity.total_chunks,
            #                     "page_range": hit.entity.page_range,
            #                     "embedding_provider": hit.entity.embedding_provider,
            #                     "embedding_model": hit.entity.embedding_model,
            #                     "embedding_timestamp": hit.entity.embedding_timestamp
            #                 }
            #             })

            processed_results = []
            seen_chunk_ids = set()

            for expanded_query in queries:
                logger.info(f"Running multi-query retrieval for: {expanded_query}")
                query_embedding = self.embedding_service.create_single_embedding(
                    expanded_query,
                    provider=sample_metadata.get('embedding_provider'),
                    model=sample_metadata.get('embedding_model')
                )
                results =collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                )

                results_count=len(results['ids'][0]) if results.get('ids') else 0
                logger.info(f"Raw search results count for expanded query: {results_count}")

                for hit in range(results_count):
                    chunk_id = results.get('ids')[0][hit]
                    hit_score=1-results['distances'][0][hit]
                    metadata = results['metadatas'][0][hit]
                    logger.info(f"Processing hit - Score: {hit_score}, Word Count: {metadata.get('word_count')}")
                    if hit_score >= threshold and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        processed_results.append({
                            "text": results.get('documents')[0][hit],
                            "score": float(hit_score),
                            "metadata": {
                                "source": metadata.get('document_name'),
                                "page": metadata.get('page_number'),
                                "chunk": chunk_id,
                                "total_chunks": metadata.get('total_chunks'),
                                "page_range": metadata.get('page_range'),
                                "embedding_provider": metadata.get('embedding_provider'),
                                "embedding_model": metadata.get('embedding_model'),
                                "embedding_timestamp": metadata.get('embedding_timestamp')
                            }
                        })

            processed_results.sort(key=lambda item: item["score"], reverse=True)
            processed_results = processed_results[:top_k]

            response_data = {"results": processed_results}

            # 添加详细的保存逻辑日志
            logger.info(f"Preparing to handle save_results (flag: {save_results})")
            if save_results:
                logger.info("Save results is True, attempting to save...")
                if processed_results:
                    try:
                        filepath = self.save_search_results(query, collection_id, processed_results)
                        logger.info(f"Successfully saved results to: {filepath}")
                        response_data["saved_filepath"] = filepath
                    except Exception as e:
                        logger.error(f"Error saving results: {str(e)}")
                        response_data["save_error"] = str(e)
                        raise  # 添加这行来查看完整的错误堆栈
                else:
                    logger.info("No results to save")
            else:
                logger.info("Save results is False, skipping save")

            return response_data

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
        finally:
            connections.disconnect("default")
