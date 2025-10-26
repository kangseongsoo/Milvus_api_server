"""
Milvus 컬렉션 스키마 정의
"""
from pymilvus import FieldSchema, CollectionSchema, DataType


def create_collection_schema(dimension: int = 1536, use_sparse: bool = True) -> CollectionSchema:
    """
    Milvus 컬렉션 스키마 생성
    
    Args:
        dimension: Dense 벡터 차원
        use_sparse: Sparse 임베딩 사용 여부 (기본값: True, 향후 고도화용)
    
    Returns:
        CollectionSchema 객체
    
    Note:
        - use_sparse=True: embedding_sparse 필드를 항상 생성 (기본값 NULL)
        - 향후 하이브리드 검색 고도화 시 sparse 벡터 사용 가능
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="청크 고유 ID (자동 생성)"
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.INT64,
            description="문서 ID (PostgreSQL FK)"
        ),
        FieldSchema(
            name="chat_bot_id",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="챗봇 ID (파티션 키)"
        ),
        FieldSchema(
            name="content_name",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="문서 고유 식별자 (URL, 파일명, 제목 등)"
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
            description="청크 순서 (0부터 시작)"
        ),
        FieldSchema(
            name="embedding_dense",
            dtype=DataType.FLOAT_VECTOR,
            dim=dimension,
            description="Dense 임베딩 벡터"
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="메타데이터 (JSON 형태, expr 필터링용)"
        )
    ]
    
    # Sparse 임베딩 필드 추가 (하이브리드 검색용, 향후 고도화)
    if use_sparse:
        fields.append(
            FieldSchema(
                name="embedding_sparse",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="Sparse 임베딩 벡터 (하이브리드 검색, 향후 고도화용)"
            )
        )
    
    schema = CollectionSchema(
        fields=fields,
        description="RAG 문서 벡터 컬렉션 (파티셔닝 + JSON 메타데이터 + Sparse 임베딩)"
    )
    
    return schema


def get_index_params():
    """
    벡터 인덱스 파라미터 반환
    
    Returns:
        인덱스 설정 딕셔너리
    """
    return {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 8,              # 그래프 연결 수 (높을수록 정확하지만 느림)
            "efConstruction": 64  # 인덱스 구축 시 탐색 범위
        }
    }


def get_search_params():
    """
    검색 파라미터 반환
    
    Returns:
        검색 설정 딕셔너리
    """
    return {
        "metric_type": "COSINE",
        "params": {
            "ef": 64  # 검색 시 탐색 범위 (높을수록 정확하지만 느림)
        }
    }

