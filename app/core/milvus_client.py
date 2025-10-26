"""
Milvus 클라이언트
벡터 저장소 연결 및 CRUD 작업
"""
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from app.config import settings
from app.utils.logger import setup_logger
from app.schemas.milvus_schema import create_collection_schema, get_index_params, get_search_params

logger = setup_logger(__name__)


class MilvusClient:
    """Milvus 벡터 데이터베이스 클라이언트"""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self._connection = None
    
    async def connect(self):
        """Milvus 연결"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD
            )
            logger.info(f"✅ Milvus 연결 성공: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Milvus 연결 실패: {str(e)}")
            raise
    
    async def disconnect(self):
        """Milvus 연결 해제"""
        try:
            connections.disconnect("default")
            logger.info("Milvus 연결 해제")
        except Exception as e:
            logger.error(f"Milvus 연결 해제 실패: {str(e)}")
    
    def create_collection(self, account_name: str, dimension: int = 1536):
        """
        계정별 컬렉션 생성
        
        Args:
            account_name: 계정명
            dimension: 벡터 차원
        
        Raises:
            Exception: 컬렉션이 이미 존재하는 경우 "already exists" 포함된 예외 발생
        
        Note:
            계정당 1개 컬렉션 (collection_chatty, collection_enterprise)
            각 컬렉션 내에서 봇별로 파티션 분리
        """
        try:
            collection_name = settings.get_collection_name(account_name)  # "collection_chatty"
            
            # 이미 존재하는지 확인
            from pymilvus import utility
            if utility.has_collection(collection_name):
                raise Exception(f"Collection '{collection_name}' already exists")
            
            # 스키마 파일에서 올바른 스키마 가져오기
            schema = create_collection_schema(
                dimension=dimension,
                use_sparse=settings.USE_SPARSE_EMBEDDING
            )
            
            collection = Collection(name=collection_name, schema=schema)
            
            # 인덱스 생성
            # chat_bot_id 스칼라 인덱스 (파티션 필터링용)
            collection.create_index(
                field_name="chat_bot_id",
                index_name="idx_chat_bot_id"
            )
            
            # doc_id 스칼라 인덱스
            collection.create_index(
                field_name="doc_id",
                index_name="idx_doc_id"
            )
            
            # content_name 스칼라 인덱스 (삭제 API용)
            collection.create_index(
                field_name="content_name",
                index_name="idx_content_name"
            )
            
            # Dense 벡터 인덱스 (스키마 파일에서 파라미터 가져오기)
            index_params = get_index_params()
            collection.create_index(
                field_name="embedding_dense",
                index_params=index_params
            )
            
            # metadata는 인덱스 불필요 (expr로 직접 필터링)
            
            logger.info(f"✅ 통합 컬렉션 생성 완료: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 생성 실패: {str(e)}")
            raise
    
    def create_partition(self, account_name: str, partition_name: str):
        """
        Milvus 파티션 생성
        
        Args:
            account_name: 계정명
            partition_name: 파티션 이름 (예: news_bot, law_bot)
        
        Note:
            collection_{account_name} 컬렉션 내에 봇별 파티션 생성
        """
        try:
            collection_name = settings.get_collection_name(account_name)
            collection = Collection(name=collection_name)
            
            # 파티션 생성
            collection.create_partition(partition_name=partition_name)
            
            logger.info(f"✅ Milvus 파티션 생성 완료: {collection_name}/{partition_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Milvus 파티션 생성 실패: {str(e)}")
            raise
    
    async def insert_vectors(
        self,
        account_name: str,
        chat_bot_id: str,
        partition_name: str,
        doc_id: int,
        content_name: str,
        chunks: List[Dict[str, Any]],
        metadata: dict = None
    ) -> List[int]:
        """
        벡터 삽입 (특정 파티션에) - 실제 구현
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID
            partition_name: 파티션 이름 (예: bot_550e8400...)
            doc_id: 문서 ID
            chunks: 청크 데이터 (embedding, chunk_index, text 포함)
            metadata: 필터링용 메타데이터 (content_type, tags 등)
        
        Returns:
            삽입된 벡터 ID 리스트
        
        Note:
            collection_{account_name}의 특정 파티션에 삽입
        """
        try:
            collection_name = settings.get_collection_name(account_name)
            collection = Collection(name=collection_name)
            
            # 메타데이터 기본값
            if metadata is None:
                metadata = {}
            
            # 엔티티 데이터 준비
            entities = [
                [doc_id] * len(chunks),  # doc_id
                [chat_bot_id] * len(chunks),  # chat_bot_id
                [content_name] * len(chunks),  # content_name
                [chunk["chunk_index"] for chunk in chunks],  # chunk_index
                [chunk["embedding"] for chunk in chunks],  # embedding_dense
                [metadata] * len(chunks)  # metadata JSON
            ]
            
            # Sparse 임베딩 필드 (향후 고도화용, 기본값 NULL)
            if settings.USE_SPARSE_EMBEDDING:
                # sparse_embedding이 있으면 사용, 없으면 빈 리스트 (NULL)
                sparse_embeddings = []
                for chunk in chunks:
                    if "sparse_embedding" in chunk and chunk["sparse_embedding"]:
                        sparse_embeddings.append(chunk["sparse_embedding"])
                    else:
                        sparse_embeddings.append([])  # 기본값 NULL
                entities.append(sparse_embeddings)
            
            # 벡터 삽입
            insert_result = collection.insert(
                entities, 
                partition_name=partition_name
            )
            
            logger.info(f"✅ Milvus 벡터 삽입 완료: collection={collection_name}, partition={partition_name}, vectors={len(chunks)}")
            return insert_result.primary_keys
            
        except Exception as e:
            logger.error(f"❌ Milvus 벡터 삽입 실패: {str(e)}")
            raise
    
    async def insert_vectors_with_retry(
        self,
        account_name: str,
        chat_bot_id: str,
        partition_name: str,
        doc_id: int,
        content_name: str,
        chunks: List[Dict[str, Any]],
        metadata: dict = None,
        max_retries: int = 3,
        backoff: float = 2.0
    ) -> List[int]:
        """
        재시도 로직이 포함된 벡터 삽입
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID
            partition_name: 파티션 이름
            doc_id: 문서 ID
            chunks: 청크 데이터
            metadata: 메타데이터
            max_retries: 최대 재시도 횟수
            backoff: 재시도 간격 (초)
        
        Returns:
            삽입된 벡터 ID 리스트
        """
        import asyncio
        
        for attempt in range(max_retries + 1):
            try:
                return await self.insert_vectors(
                    account_name=account_name,
                    chat_bot_id=chat_bot_id,
                    partition_name=partition_name,
                    doc_id=doc_id,
                    content_name=content_name,
                    chunks=chunks,
                    metadata=metadata
                )
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ Milvus 삽입 최종 실패 (시도 {max_retries + 1}회): {str(e)}")
                    raise
                else:
                    wait_time = backoff * (2 ** attempt)  # 지수 백오프
                    logger.warning(f"⚠️ Milvus 삽입 실패 (시도 {attempt + 1}/{max_retries + 1}), {wait_time}초 후 재시도: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def batch_insert_vectors(
        self,
        account_name: str,
        documents_data: List[Dict[str, Any]],
        metadata: dict = None
    ) -> List[List[int]]:
        """
        여러 문서의 벡터를 배치로 삽입
        
        Args:
            account_name: 계정명
            documents_data: 문서 데이터 리스트 (각 문서는 chat_bot_id, doc_id, chunks 포함)
            metadata: 공통 메타데이터
        
        Returns:
            각 문서별 삽입된 벡터 ID 리스트
        """
        try:
            collection_name = settings.get_collection_name(account_name)
            collection = Collection(name=collection_name)
            
            all_vector_ids = []
            
            for doc_data in documents_data:
                chat_bot_id = doc_data["chat_bot_id"]
                doc_id = doc_data["doc_id"]
                chunks = doc_data["chunks"]
                partition_name = f"bot_{chat_bot_id.replace('-', '')}"
                
                # 개별 문서 메타데이터 사용 (우선순위: 개별 > 공통)
                doc_metadata = doc_data.get("metadata", {}) or metadata or {}
                
                # 엔티티 데이터 준비
                content_name = doc_data.get("content_name", "")
                entities = [
                    [doc_id] * len(chunks),  # doc_id
                    [chat_bot_id] * len(chunks),  # chat_bot_id
                    [content_name] * len(chunks),  # content_name
                    [chunk["chunk_index"] for chunk in chunks],  # chunk_index
                    [chunk["embedding"] for chunk in chunks],  # embedding_dense
                    [doc_metadata] * len(chunks)  # metadata JSON
                ]
                
                # Sparse 임베딩 필드 (향후 고도화용, 기본값 NULL)
                if settings.USE_SPARSE_EMBEDDING:
                    # sparse_embedding이 있으면 사용, 없으면 빈 리스트 (NULL)
                    sparse_embeddings = []
                    for chunk in chunks:
                        if "sparse_embedding" in chunk and chunk["sparse_embedding"]:
                            sparse_embeddings.append(chunk["sparse_embedding"])
                        else:
                            sparse_embeddings.append([])  # 기본값 NULL
                    entities.append(sparse_embeddings)
                
                # 벡터 삽입
                insert_result = collection.insert(
                    entities, 
                    partition_name=partition_name
                )
                
                all_vector_ids.append(insert_result.primary_keys)
            
            logger.info(f"✅ Milvus 배치 벡터 삽입 완료: {len(documents_data)}개 문서")
            return all_vector_ids
            
        except Exception as e:
            logger.error(f"❌ Milvus 배치 벡터 삽입 실패: {str(e)}")
            raise
    
    async def batch_insert_vectors_with_retry(
        self,
        account_name: str,
        documents_data: List[Dict[str, Any]],
        metadata: dict = None,
        max_retries: int = 3,
        backoff: float = 2.0
    ) -> List[List[int]]:
        """
        재시도 로직이 포함된 배치 벡터 삽입
        
        Args:
            account_name: 계정명
            documents_data: 문서 데이터 리스트
            metadata: 공통 메타데이터
            max_retries: 최대 재시도 횟수
            backoff: 재시도 간격 (초)
        
        Returns:
            각 문서별 삽입된 벡터 ID 리스트
        """
        import asyncio
        
        for attempt in range(max_retries + 1):
            try:
                return await self.batch_insert_vectors(
                    account_name=account_name,
                    documents_data=documents_data,
                    metadata=metadata
                )
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ Milvus 배치 삽입 최종 실패 (시도 {max_retries + 1}회): {str(e)}")
                    raise
                else:
                    wait_time = backoff * (2 ** attempt)  # 지수 백오프
                    logger.warning(f"⚠️ Milvus 배치 삽입 실패 (시도 {attempt + 1}/{max_retries + 1}), {wait_time}초 후 재시도: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def search(
        self,
        account_name: str,
        chat_bot_id: str,
        partition_name: str,
        query_vector: List[float],
        limit: int = 5,
        filter_expr: str = None
    ):
        """
        벡터 유사도 검색 (특정 파티션에서만 + 메타데이터 필터링)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (필터용)
            partition_name: 파티션 이름 (예: news_bot)
            query_vector: 쿼리 벡터
            limit: 반환할 결과 수
            filter_expr: 메타데이터 필터 표현식 (옵션)
        
        Returns:
            검색 결과 (doc_id, chunk_index, score, metadata)
        
        Note:
            partition_names 지정으로 해당 파티션만 검색 (10~100배 빠름!)
        
        Examples:
            # PDF 파일만 검색
            filter_expr='metadata["content_type"] == "pdf"'
            
            # AI 태그 포함 문서만
            filter_expr='"ai" in metadata["tags"]'
            
            # 특정 소스 타입 + PDF
            filter_expr='metadata["source_type"] == "file" and metadata["content_type"] == "pdf"'
        """
        # TODO: 구현
        # collection_name = settings.get_collection_name(account_name)
        # collection = Collection(name=collection_name)
        # collection.load(partition_names=[partition_name])  # 해당 파티션만 로드
        # 
        # # expr 구성 (chat_bot_id + 사용자 정의 필터)
        # base_expr = f"chat_bot_id == '{chat_bot_id}'"
        # if filter_expr:
        #     expr = f"{base_expr} and {filter_expr}"  # ⭐ 메타데이터 필터 추가!
        # else:
        #     expr = base_expr
        # 
        # search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        # results = collection.search(
        #     data=[query_vector],
        #     anns_field="embedding_dense",
        #     param=search_params,
        #     limit=limit,
        #     partition_names=[partition_name],  # 파티션 지정
        #     expr=expr,  # ⭐ 메타데이터 필터링!
        #     output_fields=["doc_id", "chunk_index", "metadata"]  # metadata도 반환
        # )
        pass
    
    async def delete_by_doc_id(self, collection_name: str, doc_id: int) -> int:
        """
        특정 문서의 모든 벡터 삭제
        
        Args:
            collection_name: 컬렉션 이름
            doc_id: 문서 ID
        
        Returns:
            삭제된 벡터 수
        """
        try:
            from pymilvus import Collection
            
            collection = Collection(name=collection_name)
            collection.load()
            
            # doc_id로 필터링하여 삭제
            expr = f"doc_id == {doc_id}"
            result = collection.delete(expr=expr)
            
            deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0
            logger.info(f"Milvus 문서 삭제 완료: doc_id={doc_id}, 삭제된 벡터={deleted_count}개")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Milvus 문서 삭제 실패: {str(e)}")
            raise
    
    async def delete_by_content_name(self, collection_name: str, chat_bot_id: str, content_name: str) -> int:
        """
        content_name 기준으로 모든 벡터 삭제
        
        Args:
            collection_name: 컬렉션 이름
            chat_bot_id: 챗봇 ID (파티션 필터링용)
            content_name: 문서 고유 식별자
        
        Returns:
            삭제된 벡터 수
        """
        try:
            from pymilvus import Collection
            
            collection = Collection(name=collection_name)
            collection.load()
            
            # content_name과 chat_bot_id로 필터링하여 삭제
            expr = f'content_name == "{content_name}" and chat_bot_id == "{chat_bot_id}"'
            result = collection.delete(expr=expr)
            
            deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0
            logger.info(f"Milvus content_name 삭제 완료: content_name={content_name}, chat_bot_id={chat_bot_id}, 삭제된 벡터={deleted_count}개")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Milvus content_name 삭제 실패: {str(e)}")
            raise
    
    async def delete_partition(self, collection_name: str, partition_name: str) -> int:
        """
        파티션 전체 삭제
        
        Args:
            collection_name: 컬렉션 이름
            partition_name: 파티션 이름
        
        Returns:
            삭제된 벡터 수
        """
        try:
            from pymilvus import Collection
            
            collection = Collection(name=collection_name)
            
            # 파티션 존재 확인
            if not collection.has_partition(partition_name):
                logger.warning(f"파티션이 존재하지 않음: {partition_name}")
                return 0
            
            # 파티션 삭제 전 벡터 수 확인
            collection.load(partition_names=[partition_name])
            stats = collection.get_stats()
            vector_count = stats.get('row_count', 0)
            
            # 파티션 삭제
            collection.drop_partition(partition_name)
            
            logger.info(f"Milvus 파티션 삭제 완료: {partition_name}, 삭제된 벡터={vector_count}개")
            
            return vector_count
            
        except Exception as e:
            logger.error(f"Milvus 파티션 삭제 실패: {str(e)}")
            raise

    async def delete_by_content_names(self, collection_name: str, chat_bot_id: str, content_names: List[str]) -> int:
        """
        여러 content_name으로 벡터 일괄 삭제
        
        Args:
            collection_name: 컬렉션명
            chat_bot_id: 챗봇 ID
            content_names: 삭제할 문서의 content_name 리스트
            
        Returns:
            삭제된 벡터 수
        """
        try:
            collection = Collection(name=collection_name)
            partition_name = f"bot_{chat_bot_id.replace('-', '')}"
            
            logger.info(f"Milvus 일괄 삭제 시작: {len(content_names)}개 문서")
            logger.info(f"   - Collection: {collection_name}")
            logger.info(f"   - Partition: {partition_name}")
            logger.info(f"   - Content Names: {content_names}")
            
            # 컬렉션 로드
            collection.load()
            
            # 파티션 존재 확인
            if not collection.has_partition(partition_name):
                logger.warning(f"파티션이 존재하지 않음: {partition_name}")
                return 0
            
            # 삭제 전 벡터 수 확인
            query_expr = f"chat_bot_id == '{chat_bot_id}'"
            if len(content_names) == 1:
                query_expr += f" and content_name == '{content_names[0]}'"
            else:
                content_names_str = "', '".join(content_names)
                query_expr += f" and content_name in ['{content_names_str}']"
            
            # 삭제 전 벡터 수 조회
            query_result = collection.query(
                expr=query_expr,
                partition_names=[partition_name],
                output_fields=["count(*)"]
            )
            vector_count_before = len(query_result) if query_result else 0
            
            if vector_count_before == 0:
                logger.warning(f"삭제할 벡터가 없음: {content_names}")
                return 0
            
            # 벡터 삭제
            delete_result = collection.delete(
                expr=query_expr,
                partition_name=partition_name
            )
            
            deleted_count = delete_result.delete_count if delete_result else 0
            
            logger.info(f"Milvus 일괄 삭제 완료: {deleted_count}개 벡터 삭제")
            logger.info(f"   - 삭제 전: {vector_count_before}개")
            logger.info(f"   - 삭제 후: {deleted_count}개")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Milvus 일괄 삭제 실패: {str(e)}")
            raise


# 전역 클라이언트 인스턴스
milvus_client = MilvusClient()

