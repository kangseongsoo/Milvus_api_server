"""
트랜잭션 관리
Milvus + PostgreSQL 통합 트랜잭션 처리
"""
from typing import Any, Callable
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TransactionManager:
    """
    Milvus와 PostgreSQL의 일관성을 보장하는 트랜잭션 매니저
    
    Note: Milvus는 전통적인 ACID 트랜잭션을 지원하지 않으므로
    실패 시 수동 롤백 로직이 필요합니다.
    """
    
    def __init__(self):
        self.rollback_actions = []
    
    async def execute_with_rollback(
        self,
        postgres_action: Callable,
        milvus_action: Callable,
        rollback_postgres: Callable = None,
        rollback_milvus: Callable = None
    ):
        """
        PostgreSQL과 Milvus 작업을 트랜잭션처럼 실행
        
        Args:
            postgres_action: PostgreSQL 작업
            milvus_action: Milvus 작업
            rollback_postgres: PostgreSQL 롤백 함수
            rollback_milvus: Milvus 롤백 함수
        """
        try:
            # 1. PostgreSQL 작업 실행
            postgres_result = await postgres_action()
            
            try:
                # 2. Milvus 작업 실행
                milvus_result = await milvus_action()
                
                logger.info("트랜잭션 성공")
                return postgres_result, milvus_result
                
            except Exception as milvus_error:
                # Milvus 작업 실패 시 PostgreSQL 롤백
                logger.error(f"Milvus 작업 실패, PostgreSQL 롤백 시작: {str(milvus_error)}")
                
                if rollback_postgres:
                    await rollback_postgres()
                
                raise milvus_error
                
        except Exception as postgres_error:
            logger.error(f"PostgreSQL 작업 실패: {str(postgres_error)}")
            raise postgres_error


# 전역 트랜잭션 매니저
transaction_manager = TransactionManager()

