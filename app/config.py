"""
애플리케이션 설정 관리
환경변수 로드 및 전역 설정
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Milvus 설정
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    
    # Milvus 컬렉션 네이밍 (계정별 컬렉션 + 봇별 파티셔닝)
    MILVUS_COLLECTION_PREFIX: str = "collection_"  # 예: collection_chatty, collection_enterprise
    
    # PostgreSQL 설정
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    
    # DB 네이밍 (계정별 데이터베이스)
    POSTGRES_DB_PREFIX: str = "rag_db_"  # 예: rag_db_chatty, rag_db_enterprise
    
    # Redis 설정
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 10
    
    # 봇별 테이블 네이밍
    # bot_id (UUID)를 테이블명으로 변환: bot_{uuid_without_hyphens}
    BOT_TABLE_PREFIX: str = "bot_"  # 예: bot_550e8400e29b41d4a716446655440000
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = "openai"  # "openai", "bge-m3", "sentence-bert"
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_DIMENSION: int = 1536
    
    # 하이브리드 검색 설정 (향후 고도화용)
    USE_SPARSE_EMBEDDING: bool  # .env에서 설정 (USE_SPARSE_EMBEDDING=true/false)
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    
    # 성능 설정
    MAX_BATCH_SIZE: int = 100  # 임베딩 배치 처리 최대 크기
    
    # Milvus 메타데이터 필터링 필드 설정
    MILVUS_METADATA_FIELDS: list = [
        "content_type", "source_type", "language", "tags", "category", "source_url",
    ]
    CONNECTION_POOL_SIZE: int = 10  # PostgreSQL 연결 풀 크기
    
    # 파티션 메모리 관리 설정
    PARTITION_TTL_MINUTES: int = 30  # 파티션 자동 언로드 시간 (분)
    MEMORY_THRESHOLD_PERCENT: float = 80.0  # 메모리 임계값 (%)
    MAX_CONCURRENT_LOADS: int = 10  # 최대 동시 로드 개수
    CLEANUP_INTERVAL_SECONDS: int = 300  # 자동 정리 주기 (초, 5분)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_collection_name(self, account_name: str) -> str:
        """
        계정명으로부터 Milvus 컬렉션명 생성
        
        Args:
            account_name: 계정명 (예: chatty, enterprise, company_abc)
        
        Returns:
            Milvus 컬렉션명 (예: collection_chatty, collection_enterprise)
        """
        import re
        # 입력 검증: 영문, 숫자, 언더스코어만 허용
        if not re.match(r'^[a-zA-Z0-9_]+$', account_name):
            raise ValueError(f"Invalid account_name: {account_name}")
        
        return f"{self.MILVUS_COLLECTION_PREFIX}{account_name}"
    
    def get_db_name(self, account_name: str) -> str:
        """
        계정명으로부터 PostgreSQL DB명 생성
        
        Args:
            account_name: 계정명 (예: chatty, enterprise)
        
        Returns:
            PostgreSQL DB명 (예: rag_db_chatty, rag_db_enterprise)
        """
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', account_name):
            raise ValueError(f"Invalid account_name: {account_name}")
        
        return f"{self.POSTGRES_DB_PREFIX}{account_name}"
    
    def get_bot_table_name(self, bot_id: str) -> str:
        """
        봇 ID(UUID)로부터 PostgreSQL 테이블명 생성
        
        Args:
            bot_id: 봇 ID (UUID 형식, 예: 550e8400-e29b-41d4-a716-446655440000)
        
        Returns:
            PostgreSQL 테이블명 (예: bot_550e8400e29b41d4a716446655440000)
        
        Note:
            - UUID의 하이픈(-)을 제거하여 유효한 테이블명 생성
            - 접두사 'bot_'를 추가하여 숫자 시작 방지
        """
        # UUID 검증 및 정규화 (하이픈 제거, 소문자 변환)
        import re
        
        # 하이픈 제거
        sanitized_id = bot_id.replace("-", "").replace("_", "").lower()
        
        # 영문, 숫자만 허용 (보안)
        if not re.match(r'^[a-z0-9]+$', sanitized_id):
            raise ValueError(f"Invalid bot_id: {bot_id}. Must be UUID format.")
        
        # 길이 제한 (PostgreSQL은 63자까지, bot_ + 32자 UUID = 36자)
        if len(sanitized_id) > 50:
            raise ValueError(f"bot_id too long: {bot_id}")
        
        return f"{self.BOT_TABLE_PREFIX}{sanitized_id}"
    
    def get_postgres_url(self, account_name: str) -> str:
        """
        계정별 PostgreSQL 연결 URL 생성
        
        Args:
            account_name: 계정명
        
        Returns:
            PostgreSQL 연결 URL
        """
        db_name = self.get_db_name(account_name)
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{db_name}"
    
    def get_async_postgres_url(self, account_name: str) -> str:
        """
        계정별 비동기 PostgreSQL 연결 URL 생성
        
        Args:
            account_name: 계정명
        
        Returns:
            비동기 PostgreSQL 연결 URL
        """
        db_name = self.get_db_name(account_name)
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{db_name}"


# 전역 설정 인스턴스
settings = Settings()

