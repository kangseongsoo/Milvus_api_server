"""
PostgreSQL 클라이언트
메타데이터 저장소 연결 및 CRUD 작업 (파티셔닝 기반)
"""
from typing import List, Optional, Dict, Any
import asyncpg
import json
from app.config import settings
from app.utils.logger import setup_logger
from app.schemas.postgres_schema import get_init_sql

logger = setup_logger(__name__)


class PostgresClient:
    """
    PostgreSQL 데이터베이스 클라이언트 (파티셔닝 지원)
    
    단일 DB(rag_db_chatty)에서 bot_id로 파티셔닝된 테이블 관리
    """
    
    def __init__(self):
        # 계정별 연결 풀 캐싱 {account_name: pool}
        self.pools: Dict[str, asyncpg.Pool] = {}
    
    async def get_pool(self, account_name: str) -> asyncpg.Pool:
        """
        계정별 연결 풀 가져오기 (없으면 생성)
        
        Args:
            account_name: 계정명
        
        Returns:
            해당 계정의 연결 풀
        """
        if account_name not in self.pools:
            await self._create_pool(account_name)
        
        return self.pools[account_name]
    
    async def _create_pool(self, account_name: str):
        """
        계정별 PostgreSQL 연결 풀 생성
        
        Args:
            account_name: 계정명 (예: chatty, enterprise)
        """
        try:
            db_name = settings.get_db_name(account_name)
            
            pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=db_name,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                min_size=1,
                max_size=settings.CONNECTION_POOL_SIZE
            )
            
            self.pools[account_name] = pool
            logger.info(f"✅ PostgreSQL 연결 풀 생성: account={account_name} → DB={db_name}")
            
        except ValueError as e:
            logger.error(f"❌ 유효하지 않은 계정명: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ PostgreSQL 연결 실패 (account: {account_name}): {str(e)}")
            raise
    
    async def disconnect(self):
        """모든 PostgreSQL 연결 풀 해제"""
        for account_name, pool in self.pools.items():
            await pool.close()
            logger.info(f"PostgreSQL 연결 풀 해제: {account_name}")
        
        self.pools.clear()
    
    async def create_database(self, account_name: str):
        """
        계정별 PostgreSQL 데이터베이스 생성
        
        Args:
            account_name: 계정명
        
        Returns:
            성공 여부
        
        Note:
            postgres 데이터베이스에 연결해서 새 DB 생성
        """
        try:
            db_name = settings.get_db_name(account_name)
            
            # postgres DB에 연결 (DB 생성용)
            conn = await asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database='postgres',  # 기본 DB
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
            
            # 데이터베이스 존재 확인
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                db_name
            )
            
            if exists:
                logger.info(f"⚠️ 데이터베이스가 이미 존재합니다: {db_name}")
                await conn.close()
                return True
            
            # 데이터베이스 생성
            await conn.execute(f'CREATE DATABASE {db_name}')
            logger.info(f"✅ PostgreSQL 데이터베이스 생성 완료: {db_name}")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL 데이터베이스 생성 실패: {str(e)}")
            raise
    
    async def init_account_tables(self, account_name: str):
        """
        계정별 PostgreSQL 테이블 초기화
        
        Args:
            account_name: 계정명
        
        Note:
            bot_registry, documents, document_chunks 테이블 및 파티션 생성 함수 등 생성
        """
        pool = await self.get_pool(account_name)
        
        # postgres_schema.py에서 올바른 스키마 가져오기
        init_sql = get_init_sql() + """
        
        -- 봇 레지스트리 테이블 (파티셔닝용)
        CREATE TABLE IF NOT EXISTS bot_registry (
            bot_id VARCHAR(100) PRIMARY KEY,
            bot_name VARCHAR(255) NOT NULL,
            partition_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_bot_registry_name ON bot_registry(bot_name);
        CREATE INDEX IF NOT EXISTS idx_bot_registry_partition ON bot_registry(partition_name);
        
        -- 청크 테이블 외래키 먼저 삭제
        ALTER TABLE document_chunks DROP CONSTRAINT IF EXISTS document_chunks_doc_id_fkey;
        ALTER TABLE document_chunks DROP CONSTRAINT IF EXISTS document_chunks_pkey;
        
        -- 기존 테이블 삭제 후 파티셔닝 테이블로 재생성
        DROP TABLE IF EXISTS document_chunks CASCADE;
        DROP TABLE IF EXISTS documents CASCADE;
        
        -- 문서 테이블 (파티셔닝)
        CREATE TABLE documents (
            doc_id BIGSERIAL,
            chat_bot_id VARCHAR(100) NOT NULL,
            content_name VARCHAR(500) NOT NULL,
            chunk_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB,
            PRIMARY KEY (doc_id, chat_bot_id),
            FOREIGN KEY (chat_bot_id) REFERENCES bot_registry(bot_id) ON DELETE CASCADE,
            UNIQUE(chat_bot_id, content_name)
        ) PARTITION BY LIST (chat_bot_id);
        
        -- 청크 테이블 (파티셔닝)
        CREATE TABLE document_chunks (
            chunk_id BIGSERIAL,
            doc_id BIGINT NOT NULL,
            chat_bot_id VARCHAR(100) NOT NULL,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            page_number INT,
            PRIMARY KEY (chunk_id, chat_bot_id),
            FOREIGN KEY (doc_id, chat_bot_id) REFERENCES documents(doc_id, chat_bot_id) ON DELETE CASCADE,
            UNIQUE(doc_id, chat_bot_id, chunk_index)
        ) PARTITION BY LIST (chat_bot_id);
        
        -- 4. 파티션 생성 함수
        CREATE OR REPLACE FUNCTION create_bot_partitions(p_bot_id VARCHAR)
        RETURNS VOID AS $$
        DECLARE
            partition_suffix VARCHAR;
            documents_partition_name VARCHAR;
            chunks_partition_name VARCHAR;
        BEGIN
            partition_suffix := REPLACE(REPLACE(p_bot_id, '-', ''), '_', '');
            documents_partition_name := 'documents_' || partition_suffix;
            chunks_partition_name := 'document_chunks_' || partition_suffix;
            
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I PARTITION OF documents
                FOR VALUES IN (%L)
            ', documents_partition_name, p_bot_id);
            
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%I_created_at ON %I(created_at)
            ', documents_partition_name, documents_partition_name);
            
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%I_content_name ON %I(content_name)
            ', documents_partition_name, documents_partition_name);
            
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%I_metadata ON %I USING GIN(metadata)
            ', documents_partition_name, documents_partition_name);
            
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I PARTITION OF document_chunks
                FOR VALUES IN (%L)
            ', chunks_partition_name, p_bot_id);
            
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%I_doc_id ON %I(doc_id)
            ', chunks_partition_name, chunks_partition_name);
            
            RAISE NOTICE '파티션 생성 완료: % (bot_id: %)', documents_partition_name, p_bot_id;
        END;
        $$ LANGUAGE plpgsql;
        
        -- 5. 봇 등록 시 자동 파티션 생성 트리거
        CREATE OR REPLACE FUNCTION auto_create_bot_partitions()
        RETURNS TRIGGER AS $$
        BEGIN
            PERFORM create_bot_partitions(NEW.bot_id);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS trigger_auto_create_partitions ON bot_registry;
        CREATE TRIGGER trigger_auto_create_partitions
        AFTER INSERT ON bot_registry
        FOR EACH ROW EXECUTE FUNCTION auto_create_bot_partitions();
        """
        
        async with pool.acquire() as conn:
            await conn.execute(init_sql)
        
        logger.info(f"✅ PostgreSQL 테이블 초기화 완료: account={account_name}")
        return True
    
    async def register_bot(
        self, 
        account_name: str, 
        bot_id: str, 
        bot_name: str, 
        partition_name: str, 
        description: str = None,
        metadata: dict = None
    ) -> bool:
        """
        봇 등록 (자동으로 PostgreSQL + Milvus 파티션 생성)
        
        Args:
            account_name: 계정명
            bot_id: 봇 ID (UUID)
            bot_name: 봇 이름
            partition_name: Milvus 파티션명 (예: bot_550e8400...)
            description: 봇 설명 (선택)
            metadata: 추가 메타데이터 JSONB (선택)
        
        Returns:
            성공 여부
        
        Note:
            - PostgreSQL: 트리거로 documents, document_chunks 파티션 자동 생성
            - Milvus: collection_{account_name}의 파티션으로 생성
        """
        pool = await self.get_pool(account_name)
        
        query = """
        INSERT INTO bot_registry (bot_id, bot_name, partition_name, description, metadata)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT (bot_id) DO NOTHING
        """
        async with pool.acquire() as conn:
            result = await conn.execute(
                query, 
                bot_id, 
                bot_name, 
                partition_name, 
                description,
                json.dumps(metadata or {})  # JSON 문자열로 변환
            )
            logger.info(f"봇 등록 완료 (account: {account_name}): bot_id={bot_id}, name={bot_name}, partition={partition_name}")
            return True
    
    async def insert_document(self, account_name: str, document_data: Dict[str, Any], chunk_count: int = 0) -> int:
        """
        문서 메타데이터 삽입 (자동으로 해당 파티션에 저장)
        
        Args:
            account_name: 계정명
            document_data: 문서 메타데이터 (chat_bot_id, metadata 포함)
            chunk_count: 총 청크 개수
        
        Returns:
            생성된 doc_id
        
        Note:
            모든 메타데이터는 metadata JSONB 컬럼에 저장
            예: {"title": "...", "content_type": "pdf", "tags": [...]}
        """
        pool = await self.get_pool(account_name)
        
        query = """
        INSERT INTO documents (chat_bot_id, content, chunk_count, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING doc_id
        """
        async with pool.acquire() as conn:
            doc_id = await conn.fetchval(
                query,
                document_data.get("chat_bot_id"),
                document_data.get("content"),  # 원문 전체 (선택)
                chunk_count,
                json.dumps(document_data.get("metadata", {}))  # JSON 문자열로 변환
            )
        logger.info(f"문서 삽입 완료 (account: {account_name}, bot: {document_data.get('chat_bot_id')}): doc_id={doc_id}, chunks={chunk_count}")
        return doc_id
    
    async def insert_document_with_chunks_transaction(
        self, 
        account_name: str, 
        document_data: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        문서 + 청크를 단일 트랜잭션으로 삽입 (Saga Pattern용)
        
        Args:
            account_name: 계정명
            document_data: 문서 메타데이터
            chunks: 청크 데이터 리스트
        
        Returns:
            생성된 doc_id
        
        Note:
            PostgreSQL 트랜잭션으로 문서와 청크 삽입의 원자성 보장
        """
        pool = await self.get_pool(account_name)
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. 문서 삽입
                doc_id = await conn.fetchval("""
                    INSERT INTO documents (chat_bot_id, content_name, chunk_count, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    RETURNING doc_id
                """, 
                    document_data.get("chat_bot_id"),
                    document_data.get("content_name"),
                    len(chunks),
                    json.dumps(document_data.get("metadata", {}))
                )
                
                # 2. 청크 삽입
                await conn.executemany("""
                    INSERT INTO document_chunks (doc_id, chat_bot_id, chunk_index, chunk_text, page_number)
                    VALUES ($1, $2, $3, $4, $5)
                """, [
                    (doc_id, document_data.get("chat_bot_id"), chunk["chunk_index"], chunk["text"], chunk.get("page_number"))
                    for chunk in chunks
                ])
                
        # 트랜잭션 커밋 (이 시점에서 doc_id 확정)
        logger.info(f"✅ PostgreSQL 트랜잭션 완료: doc_id={doc_id}, chunks={len(chunks)}")
        return doc_id
    
    async def batch_insert_documents_with_chunks_transaction(
        self,
        account_name: str,
        documents: List[Dict[str, Any]]
    ) -> List[int]:
        """
        여러 문서 + 청크를 단일 트랜잭션으로 삽입 (배치 Saga Pattern용)
        
        Args:
            account_name: 계정명
            documents: 문서 데이터 리스트 (각 문서는 document_data, chunks 포함)
        
        Returns:
            생성된 doc_id 리스트
        
        Note:
            PostgreSQL 트랜잭션으로 모든 문서와 청크 삽입의 원자성 보장
            하나라도 실패하면 전체 롤백
        """
        pool = await self.get_pool(account_name)
        doc_ids = []
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                for doc_data in documents:
                    # 1. 문서 삽입
                    doc_id = await conn.fetchval("""
                        INSERT INTO documents (chat_bot_id, content_name, chunk_count, metadata)
                        VALUES ($1, $2, $3, $4::jsonb)
                        RETURNING doc_id
                    """, 
                        doc_data["document_data"].get("chat_bot_id"),
                        doc_data["document_data"].get("content_name"),
                        len(doc_data["chunks"]),
                        json.dumps(doc_data["document_data"].get("metadata", {}))
                    )
                    doc_ids.append(doc_id)
                    
                    # 2. 청크 삽입
                    await conn.executemany("""
                        INSERT INTO document_chunks (doc_id, chat_bot_id, chunk_index, chunk_text, page_number)
                        VALUES ($1, $2, $3, $4, $5)
                    """, [
                        (doc_id, doc_data["document_data"].get("chat_bot_id"), chunk["chunk_index"], chunk["text"], chunk.get("page_number"))
                        for chunk in doc_data["chunks"]
                    ])
                
                # 트랜잭션 커밋 (이 시점에서 모든 doc_id 확정)
                logger.info(f"✅ PostgreSQL 배치 트랜잭션 완료: {len(doc_ids)}개 문서")
                return doc_ids
    
    async def insert_chunks(self, account_name: str, chat_bot_id: str, doc_id: int, chunks: List[Dict[str, Any]]):
        """
        청크 데이터 삽입 (자동으로 해당 파티션에 저장)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID
            doc_id: 문서 ID
            chunks: 청크 데이터 리스트
        """
        pool = await self.get_pool(account_name)
        
        query = """
        INSERT INTO document_chunks (doc_id, chat_bot_id, chunk_index, chunk_text, page_number)
        VALUES ($1, $2, $3, $4, $5)
        """
        async with pool.acquire() as conn:
            await conn.executemany(
                query,
                [(doc_id, chat_bot_id, chunk["chunk_index"], chunk["text"], chunk.get("page_number")) 
                 for chunk in chunks]
            )
        logger.info(f"청크 삽입 완료 (account: {account_name}, bot: {chat_bot_id}): doc_id={doc_id}, count={len(chunks)}")
    
    async def get_document(self, account_name: str, chat_bot_id: str, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        문서 조회 (자동으로 해당 파티션만 스캔)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_id: 문서 ID
        
        Returns:
            문서 데이터
        """
        pool = await self.get_pool(account_name)
        
        query = "SELECT * FROM documents WHERE chat_bot_id = $1 AND doc_id = $2"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, chat_bot_id, doc_id)
            return dict(row) if row else None
    
    async def get_doc_id_by_content_name(self, account_name: str, chat_bot_id: str, content_name: str) -> Optional[int]:
        """
        content_name으로 doc_id 조회
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            content_name: 문서 고유 식별자
        
        Returns:
            doc_id (없으면 None)
        """
        pool = await self.get_pool(account_name)
        
        query = "SELECT doc_id FROM documents WHERE chat_bot_id = $1 AND content_name = $2"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, chat_bot_id, content_name)
            return row['doc_id'] if row else None
    
    async def get_documents_by_ids(self, account_name: str, chat_bot_id: str, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """
        여러 문서 일괄 조회 (자동으로 해당 파티션만 스캔)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_ids: 문서 ID 리스트
        
        Returns:
            문서 데이터 리스트
        """
        pool = await self.get_pool(account_name)
        
        query = "SELECT * FROM documents WHERE chat_bot_id = $1 AND doc_id = ANY($2)"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, chat_bot_id, doc_ids)
            return [dict(row) for row in rows]
    
    async def get_documents_with_chunks_by_ids(self, account_name: str, chat_bot_id: str, doc_ids: List[int], chunk_indices: List[int] = None) -> List[Dict[str, Any]]:
        """
        문서와 해당 청크들을 함께 조회 (검색 결과용)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_ids: 문서 ID 리스트
            chunk_indices: 특정 청크 인덱스들 (옵션)
        
        Returns:
            문서 데이터 리스트 (chunks 필드 포함)
        """
        pool = await self.get_pool(account_name)
        
        # 단일 트랜잭션으로 문서와 청크를 함께 조회
        async with pool.acquire() as conn:
            # 문서 조회
            doc_query = "SELECT * FROM documents WHERE chat_bot_id = $1 AND doc_id = ANY($2)"
            doc_rows = await conn.fetch(doc_query, chat_bot_id, doc_ids)
            documents = [dict(row) for row in doc_rows]
            
            # 각 문서의 청크들 조회 (같은 연결 사용)
            for doc in documents:
                doc_id = doc['doc_id']
                
                # 특정 청크 인덱스가 지정된 경우
                if chunk_indices:
                    chunk_query = """
                    SELECT chunk_index, chunk_text, page_number 
                    FROM document_chunks 
                    WHERE chat_bot_id = $1 AND doc_id = $2 AND chunk_index = ANY($3)
                    ORDER BY chunk_index
                    """
                    chunk_rows = await conn.fetch(chunk_query, chat_bot_id, doc_id, chunk_indices)
                else:
                    # 모든 청크 조회
                    chunk_query = """
                    SELECT chunk_index, chunk_text, page_number 
                    FROM document_chunks 
                    WHERE chat_bot_id = $1 AND doc_id = $2
                    ORDER BY chunk_index
                    """
                    chunk_rows = await conn.fetch(chunk_query, chat_bot_id, doc_id)
                
                # 청크 데이터를 딕셔너리로 변환
                doc['chunks'] = {row['chunk_index']: dict(row) for row in chunk_rows}
        
        return documents
    
    async def delete_document(self, account_name: str, chat_bot_id: str, doc_id: int):
        """
        문서 삭제 (CASCADE로 청크도 자동 삭제, 해당 파티션에서만 삭제)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_id: 문서 ID
        """
        pool = await self.get_pool(account_name)
        
        query = "DELETE FROM documents WHERE chat_bot_id = $1 AND doc_id = $2"
        async with pool.acquire() as conn:
            await conn.execute(query, chat_bot_id, doc_id)
        logger.info(f"문서 삭제 완료 (account: {account_name}, bot: {chat_bot_id}): doc_id={doc_id}")
    
    async def update_document(self, account_name: str, chat_bot_id: str, doc_id: int, document_data: Dict[str, Any], chunk_count: int = None):
        """
        문서 업데이트 (메타데이터 + chunk_count)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_id: 문서 ID
            document_data: 업데이트할 문서 데이터 (metadata 포함)
            chunk_count: 총 청크 개수 (변경 시)
        
        Note:
            모든 메타데이터는 metadata JSONB에 저장
        """
        pool = await self.get_pool(account_name)
        
        if chunk_count is not None:
            query = """
            UPDATE documents 
            SET content = $1, chunk_count = $2, metadata = $3::jsonb, updated_at = NOW()
            WHERE chat_bot_id = $4 AND doc_id = $5
            """
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    document_data.get("content"),
                    chunk_count,
                    json.dumps(document_data.get("metadata", {})),
                    chat_bot_id,
                    doc_id
                )
        else:
            query = """
            UPDATE documents 
            SET content = $1, metadata = $2::jsonb, updated_at = NOW()
            WHERE chat_bot_id = $3 AND doc_id = $4
            """
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    document_data.get("content"),
                    json.dumps(document_data.get("metadata", {})),
                    chat_bot_id,
                    doc_id
                )
        
        logger.info(f"문서 업데이트 완료 (account: {account_name}, bot: {chat_bot_id}): doc_id={doc_id}")
    
    async def update_metadata(self, account_name: str, chat_bot_id: str, doc_id: int, metadata_updates: Dict[str, Any]):
        """
        메타데이터만 업데이트 (해당 파티션에서만 업데이트)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            doc_id: 문서 ID
            metadata_updates: 업데이트할 메타데이터 (JSONB 병합)
        
        Note:
            기존 metadata와 병합하여 업데이트
        """
        pool = await self.get_pool(account_name)
        
        query = """
        UPDATE documents 
        SET metadata = metadata || $1::jsonb, updated_at = NOW()
        WHERE chat_bot_id = $2 AND doc_id = $3
        """
        async with pool.acquire() as conn:
            await conn.execute(query, json.dumps(metadata_updates), chat_bot_id, doc_id)
        
        logger.info(f"메타데이터 업데이트 완료 (account: {account_name}, bot: {chat_bot_id}): doc_id={doc_id}")
    
    async def get_bot_stats(self, account_name: str, chat_bot_id: str) -> Dict[str, Any]:
        """
        봇별 통계 조회 (해당 파티션만 스캔)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID
        
        Returns:
            통계 정보 (문서 수, 청크 수 등)
        """
        pool = await self.get_pool(account_name)
        
        query = """
        SELECT 
            COUNT(*) as doc_count,
            MAX(created_at) as last_updated
        FROM documents
        WHERE chat_bot_id = $1
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, chat_bot_id)
            return dict(row) if row else {}
    
    async def delete_document_by_content_name(self, account_name: str, chat_bot_id: str, content_name: str) -> tuple:
        """
        content_name 기준으로 문서 삭제 (해당 파티션에서만)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            content_name: 문서 고유 식별자
        
        Returns:
            (삭제된 문서 수, 삭제된 청크 수)
        """
        pool = await self.get_pool(account_name)
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. 삭제 전 통계 조회
                stats_query = """
                SELECT 
                    COUNT(DISTINCT d.doc_id) as doc_count,
                    COUNT(c.chunk_id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.doc_id = c.doc_id AND d.chat_bot_id = c.chat_bot_id
                WHERE d.chat_bot_id = $1 AND d.content_name = $2
                """
                stats_row = await conn.fetchrow(stats_query, chat_bot_id, content_name)
                doc_count = stats_row['doc_count'] if stats_row else 0
                chunk_count = stats_row['chunk_count'] if stats_row else 0
                
                if doc_count == 0:
                    logger.warning(f"삭제할 문서가 없음 (account: {account_name}, bot: {chat_bot_id}, content_name: {content_name})")
                    return 0, 0
                
                # 2. 청크 삭제 (CASCADE로 자동 삭제되지만 명시적으로)
                await conn.execute("""
                    DELETE FROM document_chunks 
                    WHERE chat_bot_id = $1 AND doc_id IN (
                        SELECT doc_id FROM documents 
                        WHERE chat_bot_id = $1 AND content_name = $2
                    )
                """, chat_bot_id, content_name)
                
                # 3. 문서 삭제
                await conn.execute("""
                    DELETE FROM documents 
                    WHERE chat_bot_id = $1 AND content_name = $2
                """, chat_bot_id, content_name)
                
                logger.info(f"문서 삭제 완료 (account: {account_name}, bot: {chat_bot_id}, content_name: {content_name}): {doc_count}개 문서, {chunk_count}개 청크")
                return doc_count, chunk_count

    async def delete_documents_by_content_names(self, account_name: str, chat_bot_id: str, content_names: List[str]) -> tuple:
        """
        여러 content_name 기준으로 문서 일괄 삭제 (해당 파티션에서만)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            content_names: 문서 고유 식별자 리스트
        
        Returns:
            (삭제된 문서 수, 삭제된 청크 수)
        """
        pool = await self.get_pool(account_name)
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. 삭제 전 통계 조회
                if len(content_names) == 1:
                    # 단일 문서인 경우
                    stats_query = """
                    SELECT 
                        COUNT(DISTINCT d.doc_id) as doc_count,
                        COUNT(c.chunk_id) as chunk_count
                    FROM documents d
                    LEFT JOIN document_chunks c ON d.doc_id = c.doc_id AND d.chat_bot_id = c.chat_bot_id
                    WHERE d.chat_bot_id = $1 AND d.content_name = $2
                    """
                    stats_row = await conn.fetchrow(stats_query, chat_bot_id, content_names[0])
                else:
                    # 여러 문서인 경우
                    placeholders = ','.join([f'${i+2}' for i in range(len(content_names))])
                    stats_query = f"""
                    SELECT 
                        COUNT(DISTINCT d.doc_id) as doc_count,
                        COUNT(c.chunk_id) as chunk_count
                    FROM documents d
                    LEFT JOIN document_chunks c ON d.doc_id = c.doc_id AND d.chat_bot_id = c.chat_bot_id
                    WHERE d.chat_bot_id = $1 AND d.content_name IN ({placeholders})
                    """
                    stats_row = await conn.fetchrow(stats_query, chat_bot_id, *content_names)
                
                doc_count = stats_row['doc_count'] if stats_row else 0
                chunk_count = stats_row['chunk_count'] if stats_row else 0
                
                if doc_count == 0:
                    logger.warning(f"삭제할 문서가 없음 (account: {account_name}, bot: {chat_bot_id}, content_names: {content_names})")
                    return 0, 0
                
                # 2. 청크 삭제 (CASCADE로 자동 삭제되지만 명시적으로)
                if len(content_names) == 1:
                    # 단일 문서인 경우
                    await conn.execute("""
                        DELETE FROM document_chunks 
                        WHERE chat_bot_id = $1 AND doc_id IN (
                            SELECT doc_id FROM documents 
                            WHERE chat_bot_id = $1 AND content_name = $2
                        )
                    """, chat_bot_id, content_names[0])
                else:
                    # 여러 문서인 경우
                    placeholders = ','.join([f'${i+2}' for i in range(len(content_names))])
                    await conn.execute(f"""
                        DELETE FROM document_chunks 
                        WHERE chat_bot_id = $1 AND doc_id IN (
                            SELECT doc_id FROM documents 
                            WHERE chat_bot_id = $1 AND content_name IN ({placeholders})
                        )
                    """, chat_bot_id, *content_names)
                
                # 3. 문서 삭제
                if len(content_names) == 1:
                    # 단일 문서인 경우
                    await conn.execute("""
                        DELETE FROM documents 
                        WHERE chat_bot_id = $1 AND content_name = $2
                    """, chat_bot_id, content_names[0])
                else:
                    # 여러 문서인 경우
                    placeholders = ','.join([f'${i+2}' for i in range(len(content_names))])
                    await conn.execute(f"""
                        DELETE FROM documents 
                        WHERE chat_bot_id = $1 AND content_name IN ({placeholders})
                    """, chat_bot_id, *content_names)
                
                logger.info(f"문서 일괄 삭제 완료 (account: {account_name}, bot: {chat_bot_id}, content_names: {len(content_names)}개): {doc_count}개 문서, {chunk_count}개 청크")
                return doc_count, chunk_count

    async def get_existing_content_names(self, account_name: str, chat_bot_id: str, content_names: List[str]) -> List[str]:
        """
        존재하는 content_name들만 반환
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID (파티션 키)
            content_names: 확인할 문서 고유 식별자 리스트
        
        Returns:
            존재하는 content_name 리스트
        """
        from app.utils.logger import setup_logger
        logger = setup_logger(__name__)
        
        pool = await self.get_pool(account_name)
        
        logger.info(f"🔍 PostgreSQL에서 content_name 존재 확인 시작")
        logger.info(f"   - Account: {account_name}")
        logger.info(f"   - Bot ID: {chat_bot_id}")
        logger.info(f"   - 요청한 content_names: {len(content_names)}개")
        logger.info(f"   - 요청한 content_names 값: {content_names}")
        
        async with pool.acquire() as conn:
            if len(content_names) == 1:
                # 단일 문서인 경우
                query = """
                SELECT content_name FROM documents 
                WHERE chat_bot_id = $1 AND content_name = $2
                """
                logger.debug(f"SQL 쿼리: {query}, chat_bot_id={chat_bot_id}, content_name={content_names[0]}")
                result = await conn.fetchrow(query, chat_bot_id, content_names[0])
                
                if result:
                    logger.info(f"✅ 문서 발견: content_name='{result['content_name']}'")
                    return [content_names[0]]
                else:
                    logger.warning(f"❌ 문서를 찾지 못함: content_name='{content_names[0]}'")
                    
                    # URL 형식인 경우에만 http/https 차이 자동 매칭 시도
                    requested_name = content_names[0]
                    if requested_name.startswith('http://') or requested_name.startswith('https://'):
                        # http ↔ https 교체
                        if requested_name.startswith('http://'):
                            alternative_name = requested_name.replace('http://', 'https://', 1)
                        else:
                            alternative_name = requested_name.replace('https://', 'http://', 1)
                        
                        logger.info(f"🔄 URL 형식 감지, http/https 차이로 재검색: '{alternative_name}'")
                        alt_query = """
                        SELECT content_name FROM documents 
                        WHERE chat_bot_id = $1 AND content_name = $2
                        """
                        alt_result = await conn.fetchrow(alt_query, chat_bot_id, alternative_name)
                        
                        if alt_result:
                            matched_name = alt_result['content_name']
                            logger.info(f"✅ 자동 매칭 (http/https): '{requested_name}' → '{matched_name}'")
                            return [matched_name]
                        else:
                            logger.warning(f"❌ http/https 교체 버전도 찾지 못함: '{alternative_name}'")
                
                return []
            else:
                # 여러 문서인 경우
                placeholders = ','.join([f'${i+2}' for i in range(len(content_names))])
                query = f"""
                SELECT content_name FROM documents 
                WHERE chat_bot_id = $1 AND content_name IN ({placeholders})
                """
                logger.debug(f"SQL 쿼리: {query}, chat_bot_id={chat_bot_id}, content_names={content_names}")
                results = await conn.fetch(query, chat_bot_id, *content_names)
                found_names = [row['content_name'] for row in results]
                
                logger.info(f"✅ 발견된 문서: {len(found_names)}개 / {len(content_names)}개")
                
                # 찾지 못한 content_names에 대해 URL 형식인 경우 http/https 차이만 자동 매칭
                missing = set(content_names) - set(found_names)
                if missing:
                    logger.warning(f"❌ 찾지 못한 content_names: {missing}")
                    
                    # 각 누락된 content_name에 대해 URL 형식인 경우만 http/https 매칭
                    for missing_name in missing:
                        # URL 형식인 경우에만 http/https 차이 자동 매칭 시도
                        if missing_name.startswith('http://') or missing_name.startswith('https://'):
                            # http ↔ https 교체
                            if missing_name.startswith('http://'):
                                alternative_name = missing_name.replace('http://', 'https://', 1)
                            else:
                                alternative_name = missing_name.replace('https://', 'http://', 1)
                            
                            logger.info(f"🔄 URL 형식 감지, http/https 차이로 재검색: '{alternative_name}'")
                            alt_query = """
                            SELECT content_name FROM documents 
                            WHERE chat_bot_id = $1 AND content_name = $2
                            """
                            alt_result = await conn.fetchrow(alt_query, chat_bot_id, alternative_name)
                            
                            if alt_result:
                                matched_name = alt_result['content_name']
                                # 이미 찾은 목록에 없는 경우만 추가
                                if matched_name not in found_names:
                                    logger.info(f"✅ 자동 매칭 (http/https): '{missing_name}' → '{matched_name}'")
                                    found_names.append(matched_name)
                                else:
                                    logger.warning(f"⚠️ 매칭된 content_name '{matched_name}'는 이미 다른 요청과 매칭됨")
                            else:
                                logger.warning(f"❌ http/https 교체 버전도 찾지 못함: '{alternative_name}'")
                
                return found_names

    async def delete_bot_data(self, account_name: str, chat_bot_id: str) -> tuple:
        """
        봇 전체 데이터 삭제 (해당 파티션의 모든 문서와 청크)
        
        Args:
            account_name: 계정명
            chat_bot_id: 챗봇 ID
        
        Returns:
            (삭제된 문서 수, 삭제된 청크 수)
        """
        pool = await self.get_pool(account_name)
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. 삭제 전 통계 조회
                stats_query = """
                SELECT 
                    COUNT(DISTINCT d.doc_id) as doc_count,
                    COUNT(c.chunk_id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.doc_id = c.doc_id
                WHERE d.chat_bot_id = $1
                """
                stats_row = await conn.fetchrow(stats_query, chat_bot_id)
                doc_count = stats_row['doc_count'] if stats_row else 0
                chunk_count = stats_row['chunk_count'] if stats_row else 0
                
                # 2. 청크 삭제 (CASCADE로 자동 삭제되지만 명시적으로)
                await conn.execute("""
                    DELETE FROM document_chunks 
                    WHERE chat_bot_id = $1
                """, chat_bot_id)
                
                # 3. 문서 삭제
                await conn.execute("""
                    DELETE FROM documents 
                    WHERE chat_bot_id = $1
                """, chat_bot_id)
                
                logger.info(f"봇 데이터 삭제 완료 (account: {account_name}, bot: {chat_bot_id}): {doc_count}개 문서, {chunk_count}개 청크")
                return doc_count, chunk_count


# 전역 클라이언트 인스턴스
postgres_client = PostgresClient()

