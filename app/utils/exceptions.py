"""
커스텀 예외 클래스
"""


class MilvusRAGException(Exception):
    """기본 예외 클래스"""
    pass


class MilvusConnectionError(MilvusRAGException):
    """Milvus 연결 오류"""
    pass


class PostgresConnectionError(MilvusRAGException):
    """PostgreSQL 연결 오류"""
    pass


class EmbeddingError(MilvusRAGException):
    """임베딩 처리 오류"""
    pass


class TransactionError(MilvusRAGException):
    """트랜잭션 오류"""
    pass


class DocumentNotFoundError(MilvusRAGException):
    """문서를 찾을 수 없음"""
    pass


class CollectionNotFoundError(MilvusRAGException):
    """컬렉션을 찾을 수 없음"""
    pass

