"""
Milvus 필터링용 메타데이터 스키마 정의

Milvus의 metadata 필드에 저장할 수 있는 필터링용 메타데이터만 정의합니다.
상세한 메타데이터는 PostgreSQL에 저장됩니다.
"""
from typing import Optional
from app.config import settings


# MilvusMetadata 클래스는 설정 기반으로 동작하므로 제거
# 대신 filter_milvus_metadata() 함수를 사용하여 동적으로 필터링


def filter_milvus_metadata(all_metadata: dict) -> dict:
    """
    전체 메타데이터에서 Milvus 필터링용 메타데이터만 추출
    
    Args:
        all_metadata: 전체 메타데이터 딕셔너리
        
    Returns:
        Milvus 필터링용 메타데이터 딕셔너리
    """
    # 설정에서 Milvus 필터링 필드 목록 가져오기
    milvus_fields = set(settings.MILVUS_METADATA_FIELDS)
    
    # Milvus 필터링용 필드만 추출
    milvus_metadata = {}
    for key, value in all_metadata.items():
        if key in milvus_fields and value is not None:
            milvus_metadata[key] = value
    
    return milvus_metadata


def get_postgresql_metadata(all_metadata: dict) -> dict:
    """
    전체 메타데이터에서 PostgreSQL용 상세 메타데이터 추출
    
    Args:
        all_metadata: 전체 메타데이터 딕셔너리
        
    Returns:
        PostgreSQL용 상세 메타데이터 딕셔너리
    """
    # 설정에서 Milvus 필터링 필드 목록 가져오기
    milvus_fields = set(settings.MILVUS_METADATA_FIELDS)
    
    # Milvus 필드가 아닌 모든 필드를 PostgreSQL용으로 분류
    postgresql_metadata = {}
    for key, value in all_metadata.items():
        if key not in milvus_fields and value is not None:
            postgresql_metadata[key] = value
    
    return postgresql_metadata


def get_milvus_metadata_fields() -> set:
    """
    설정에서 Milvus 메타데이터 필드 목록을 가져옵니다.
    
    Returns:
        Milvus 필터링용 필드 집합
    """
    return set(settings.MILVUS_METADATA_FIELDS)


def is_milvus_metadata_field(field_name: str) -> bool:
    """
    특정 필드가 Milvus 필터링용 필드인지 확인합니다.
    
    Args:
        field_name: 확인할 필드명
        
    Returns:
        Milvus 필터링용 필드 여부
    """
    return field_name in settings.MILVUS_METADATA_FIELDS


# === Milvus 필터링 예시 ===
MILVUS_FILTER_EXAMPLES = {
    "content_type": {
        "description": "특정 콘텐츠 타입만 검색",
        "examples": [
            'metadata["content_type"] == "pdf"',
            'metadata["content_type"] == "html"'
        ]
    },
    "source_type": {
        "description": "특정 소스 타입만 검색", 
        "examples": [
            'metadata["source_type"] == "file"',
            'metadata["source_type"] == "url"'
        ]
    },
    "tags": {
        "description": "태그 기반 필터링",
        "examples": [
            '"ai" in metadata["tags"]',
            '"news" in metadata["tags"] or "tech" in metadata["tags"]'
        ]
    },
    "language": {
        "description": "언어별 필터링",
        "examples": [
            'metadata["language"] == "ko"',
            'metadata["language"] == "en"'
        ]
    },
    "author": {
        "description": "작성자별 필터링",
        "examples": [
            'metadata["author"] == "강성수"',
            'metadata["author"] == "김철수"'
        ]
    },
    "department": {
        "description": "부서별 필터링",
        "examples": [
            'metadata["department"] == "AI연구소"',
            'metadata["department"] == "개발팀"'
        ]
    },
    "created_date": {
        "description": "날짜 범위 필터링",
        "examples": [
            'metadata["created_date"] >= "2024-01-01"',
            'metadata["created_date"] == "2024-12-01"'
        ]
    },
    "page_count": {
        "description": "페이지 수 기반 필터링",
        "examples": [
            'metadata["page_count"] >= 100',
            'metadata["page_count"] < 50'
        ]
    },
    "status": {
        "description": "상태별 필터링",
        "examples": [
            'metadata["status"] == "active"',
            'metadata["status"] != "deleted"'
        ]
    },
    "is_public": {
        "description": "공개 여부 필터링",
        "examples": [
            'metadata["is_public"] == true',
            'metadata["is_public"] == false'
        ]
    },
    "complex": {
        "description": "복합 조건 필터링",
        "examples": [
            'metadata["content_type"] == "pdf" and "ai" in metadata["tags"]',
            'metadata["source_type"] == "file" and metadata["language"] == "ko"',
            'metadata["author"] == "강성수" and metadata["department"] == "AI연구소"'
        ]
    }
}