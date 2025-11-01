"""
문서 관련 Pydantic 모델
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChunkData(BaseModel):
    """청크 데이터 (간소화)"""
    chunk_index: int = Field(..., description="청크 순서 (0부터 시작)", example=0)
    text: str = Field(..., description="청크 텍스트", example="인공지능은 데이터를 기반으로...")


class ChunkDataWithEmbedding(BaseModel):
    """임베딩 벡터를 포함한 청크 데이터 (마이그레이션용)"""
    chunk_index: int = Field(..., description="청크 순서 (0부터 시작)", example=0)
    text: str = Field(..., description="청크 텍스트", example="인공지능은 데이터를 기반으로...")
    embedding: List[float] = Field(..., description="임베딩 벡터 (기존 PostgreSQL에서 가져옴)", example=[0.1, 0.2, 0.3])


class DocumentInsertRequest(BaseModel):
    """
    문서 삽입 요청 (최종 간소화 버전)
    
    Note: 
    - account_name으로 컬렉션 선택 (collection_chatty)
    - chat_bot_id로 파티션 자동 선택
    - content_name으로 문서 고유 식별 (파일명, URL, 제목 등)
    - metadata는 자유 형식 JSON (PostgreSQL JSONB, Milvus JSON 타입)
    """
    # 필수 필드
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    content_name: str = Field(..., description="문서 고유 식별자 (파일명, URL, 제목 등)", example="https://example.com/article1")
    chunks: List[ChunkData] = Field(..., description="텍스트 청크 리스트", min_items=1)
    
    # 선택 필드 (자유 형식 메타데이터)
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="""문서 메타데이터 (자유 형식 JSON)
        
        ⚠️ 메타데이터 저장 규칙:
        - Milvus 필터링용: content_type, source_type, language, tags, category, author, department, created_date, page_count, file_size, status, priority, is_public, has_attachments
        - PostgreSQL 전체: 모든 메타데이터 필드 (전체 저장)
        
        예시:
        {
            "title": "인공지능 입문서",           // → PostgreSQL (전체)
            "content_type": "pdf",              // → Milvus + PostgreSQL
            "source_type": "file",              // → Milvus + PostgreSQL  
            "tags": ["ai", "ml"],               // → Milvus + PostgreSQL
            "author": "강성수",                  // → Milvus + PostgreSQL
            "file_path": "/uploads/ai.pdf",     // → PostgreSQL (전체)
            "detailed_info": {...}              // → PostgreSQL (전체)
        }""",
        example={
            "title": "인공지능 입문서",
            "content_type": "pdf",
            "source_type": "file", 
            "tags": ["ai", "ml"],
            "author": "강성수",
            "department": "AI연구소",
            "language": "ko",
            "created_date": "2024-01-15",
            "page_count": 120,
            "file_path": "/uploads/ai.pdf"
        }
    )


class DocumentInsertResponse(BaseModel):
    """문서 삽입 응답"""
    status: str = Field(..., description="상태", example="success")
    doc_id: int = Field(..., description="생성된 문서 ID", example=1234)
    total_chunks: int = Field(..., description="삽입된 총 청크 수", example=120)
    postgres_insert_time_ms: float = Field(..., description="PostgreSQL 삽입 시간 (ms)", example=5.2)
    embedding_time_ms: float = Field(..., description="임베딩 처리 시간 (ms)", example=2450.8)
    milvus_insert_time_ms: float = Field(..., description="Milvus 삽입 시간 (ms)", example=180.5)
    total_time_ms: float = Field(..., description="총 처리 시간 (ms)", example=2636.5)


class DocumentWithChunks(BaseModel):
    """문서와 청크 묶음 (배치용) - 최종 간소화 버전"""
    # 필수 필드
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    content_name: str = Field(..., description="문서 고유 식별자 (파일명, URL, 제목 등)", example="https://example.com/article1")
    chunks: List[ChunkData] = Field(..., description="텍스트 청크 리스트", min_items=1)
    
    # 선택 필드 (자유 형식 메타데이터)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="문서 메타데이터 (자유 형식 JSON)")


class BatchInsertRequest(BaseModel):
    """배치 문서 삽입 요청"""
    account_name: str = Field(..., description="계정명", example="chatty")
    documents: List[DocumentWithChunks] = Field(..., description="문서 리스트", min_items=1)


class DocumentWithChunksAndEmbeddings(BaseModel):
    """임베딩을 포함한 문서와 청크 묶음 (마이그레이션용)"""
    # 필수 필드
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    content_name: str = Field(..., description="문서 고유 식별자 (파일명, URL, 제목 등)", example="https://example.com/article1")
    chunks: List[ChunkDataWithEmbedding] = Field(..., description="임베딩 포함 텍스트 청크 리스트", min_items=1)
    
    # 선택 필드 (자유 형식 메타데이터)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="문서 메타데이터 (자유 형식 JSON)")


class BatchInsertWithEmbeddingsRequest(BaseModel):
    """임베딩을 포함한 배치 문서 삽입 요청 (마이그레이션용)"""
    account_name: str = Field(..., description="계정명", example="chatty")
    documents: List[DocumentWithChunksAndEmbeddings] = Field(..., description="임베딩 포함 문서 리스트", min_items=1)


class BatchInsertResult(BaseModel):
    """배치 삽입 개별 결과"""
    doc_id: int = Field(..., description="문서 ID")
    title: str = Field(..., description="문서 제목")
    total_chunks: int = Field(..., description="청크 수")
    success: bool = Field(..., description="성공 여부")
    error: Optional[str] = Field(None, description="에러 메시지 (실패 시)")


class BatchInsertResponse(BaseModel):
    """배치 문서 삽입 응답"""
    status: str = Field(..., description="상태", example="success")
    total_documents: int = Field(..., description="처리된 총 문서 수", example=100)
    total_chunks: int = Field(..., description="삽입된 총 청크 수", example=12000)
    success_count: int = Field(..., description="성공한 문서 수", example=98)
    failure_count: int = Field(..., description="실패한 문서 수", example=2)
    results: List[BatchInsertResult] = Field(..., description="개별 결과 리스트")
    postgres_insert_time_ms: float = Field(..., description="PostgreSQL 삽입 시간 (ms)")
    embedding_time_ms: float = Field(..., description="임베딩 처리 시간 (ms)")
    milvus_insert_time_ms: float = Field(..., description="Milvus 삽입 시간 (ms)")
    total_time_ms: float = Field(..., description="총 처리 시간 (ms)")


class DocumentResponse(BaseModel):
    """문서 조회 응답"""
    status: str = Field(..., description="상태")
    doc_id: int = Field(..., description="문서 ID")
    chunk_count: int = Field(..., description="청크 수")
    chunks: List[Dict[str, Any]] = Field(..., description="청크 리스트")


class DocumentUpdateRequest(BaseModel):
    """문서 업데이트 요청 (최종 간소화 버전)"""
    # 필수 필드
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    chunks: List[ChunkData] = Field(..., description="텍스트 청크 리스트", min_items=1)
    
    # 선택 필드 (자유 형식 메타데이터)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="문서 메타데이터 (자유 형식 JSON)")


class DocumentUpdateResponse(BaseModel):
    """문서 업데이트 응답"""
    status: str = Field(..., description="상태")
    message: str = Field(..., description="메시지")
    doc_id: int = Field(..., description="문서 ID")
    deleted_chunks: int = Field(..., description="삭제된 청크 수")
    inserted_chunks: int = Field(..., description="삽입된 청크 수")
    postgres_time_ms: float = Field(..., description="PostgreSQL 처리 시간 (ms)")
    embedding_time_ms: float = Field(..., description="임베딩 처리 시간 (ms)")
    milvus_time_ms: float = Field(..., description="Milvus 처리 시간 (ms)")
    total_time_ms: float = Field(..., description="총 처리 시간 (ms)")


class MetadataUpdateRequest(BaseModel):
    """메타데이터 업데이트 요청"""
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")
    metadata_updates: Dict[str, Any] = Field(..., description="업데이트할 메타데이터")


class MetadataUpdateResponse(BaseModel):
    """메타데이터 업데이트 응답"""
    status: str = Field(..., description="상태")
    message: str = Field(..., description="메시지")
    doc_id: int = Field(..., description="문서 ID")
    updated_at: datetime = Field(..., description="업데이트 시간")
    postgres_time_ms: float = Field(..., description="PostgreSQL 처리 시간 (ms)")


class DocumentDeleteRequest(BaseModel):
    """문서 삭제 요청 (여러 문서 일괄 삭제)"""
    content_names: List[str] = Field(..., description="삭제할 문서들의 고유 식별자 리스트", example=["test_document_001", "test_document_002"])
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")


class DocumentDeleteResponse(BaseModel):
    """문서 삭제 응답 (여러 문서 일괄 삭제)"""
    status: str = Field(..., description="상태", example="partial_success")
    message: str = Field(..., description="메시지", example="Deleted 2 out of 3 requested documents")
    total_requested: int = Field(..., description="요청된 문서 수", example=3)
    total_success: int = Field(..., description="성공한 문서 수", example=2)
    total_failed: int = Field(..., description="실패한 문서 수", example=1)
    successful_content_names: List[str] = Field(..., description="삭제 성공한 문서 리스트", example=["doc1", "doc2"])
    failed_content_names: List[str] = Field(..., description="삭제 실패한 문서 리스트", example=["doc3"])
    deleted_documents: int = Field(..., description="삭제된 문서 수", example=2)
    deleted_chunks: int = Field(..., description="삭제된 청크 수", example=30)
    deleted_vectors: int = Field(..., description="삭제된 벡터 수", example=30)
    postgres_delete_time_ms: float = Field(..., description="PostgreSQL 삭제 시간 (ms)", example=45.2)
    milvus_delete_time_ms: float = Field(..., description="Milvus 삭제 시간 (ms)", example=123.8)
    total_time_ms: float = Field(..., description="총 삭제 시간 (ms)", example=169.0)






class BotDeleteRequest(BaseModel):
    """봇 전체 삭제 요청 (chat_bot_id 기준)"""
    account_name: str = Field(..., description="계정명", example="chatty")
    chat_bot_id: str = Field(..., description="챗봇 ID (UUID)", example="550e8400-e29b-41d4-a716-446655440000")


class BotDeleteResponse(BaseModel):
    """봇 전체 삭제 응답"""
    status: str = Field(..., description="상태", example="success")
    chat_bot_id: str = Field(..., description="삭제된 봇 ID")
    deleted_documents: int = Field(..., description="삭제된 문서 수")
    deleted_chunks: int = Field(..., description="삭제된 청크 수")
    deleted_vectors: int = Field(..., description="삭제된 벡터 수")
    postgres_delete_time_ms: float = Field(..., description="PostgreSQL 삭제 시간 (ms)")
    milvus_delete_time_ms: float = Field(..., description="Milvus 삭제 시간 (ms)")
    total_time_ms: float = Field(..., description="총 삭제 시간 (ms)")

