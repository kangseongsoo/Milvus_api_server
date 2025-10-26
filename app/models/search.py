"""
검색 관련 Pydantic 모델
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    """
    검색 요청
    
    Note: 
    - account_name으로 컬렉션 선택 (collection_chatty)
    - chat_bot_id로 파티션 자동 선택
    - filter_expr로 메타데이터 필터링 (옵션)
    """
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    query_text: str = Field(..., description="검색 쿼리", example="인공지능 학습 방법")
    limit: int = Field(5, description="반환할 결과 수", example=5)
    filter_expr: Optional[str] = Field(None, description="메타데이터 필터 표현식", example='metadata["file_type"] == "pdf"')


class SearchResultItem(BaseModel):
    """검색 결과 아이템"""
    doc_id: int = Field(..., description="문서 ID")
    chunk_index: int = Field(..., description="청크 순서")
    score: float = Field(..., description="유사도 점수")
    chunk_text: str = Field(..., description="청크 텍스트")
    document: Dict[str, Any] = Field(..., description="문서 메타데이터")


class SearchResponse(BaseModel):
    """검색 응답"""
    status: str = Field(..., description="상태", example="success")
    vector_search_time_ms: float = Field(..., description="벡터 검색 시간 (ms)")
    postgres_query_time_ms: float = Field(..., description="PostgreSQL 조회 시간 (ms)")
    total_time_ms: float = Field(..., description="총 처리 시간 (ms)")
    results: List[SearchResultItem] = Field(..., description="검색 결과 리스트")

