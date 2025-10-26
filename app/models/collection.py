"""
컬렉션 관련 Pydantic 모델
"""
from pydantic import BaseModel, Field


# ========== 1. 컬렉션 생성 (최초 1회) ==========
class CollectionInitRequest(BaseModel):
    """
    컬렉션 생성 요청
    
    계정별 Milvus 컬렉션 생성 (예: collection_chatty)
    이미 존재하면 "already exists" 메시지 반환 (데이터는 그대로 유지)
    
    Note: 벡터 차원은 시스템 설정값(config.EMBEDDING_DIMENSION) 사용
    """
    account_name: str = Field(..., description="계정명", example="chatty")


class CollectionInitResponse(BaseModel):
    """컬렉션 생성 응답"""
    status: str = Field(..., description="상태", example="success")
    message: str = Field(..., description="메시지", example="Collection created successfully.")
    collection_name: str = Field(..., description="컬렉션명", example="collection_chatty")


# ========== 2. 봇 등록 (파티션 생성) ==========
class BotRegisterRequest(BaseModel):
    """
    봇 등록 요청
    
    Note: 
    - 파티션명은 chat_bot_id로 자동 생성 (예: bot_550e8400e29b41d4a716446655440000)
    - PostgreSQL bot_registry에 등록 + 파티션 자동 생성
    - Milvus 파티션 생성
    """
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    bot_name: str = Field(..., description="봇 이름", example="뉴스봇")
    description: str | None = Field(None, description="봇 설명")
    metadata: dict | None = Field(None, description="추가 메타데이터 (JSONB)")


class BotRegisterResponse(BaseModel):
    """봇 등록 응답"""
    status: str = Field(..., description="상태", example="success")
    message: str = Field(..., description="메시지", example="Bot registered successfully.")
    partition_name: str = Field(..., description="생성된 파티션명", example="bot_550e8400e29b41d4a716446655440000")
    collection_name: str = Field(..., description="컬렉션명", example="collection_chatty")

