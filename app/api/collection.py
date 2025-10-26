"""
컬렉션 관리 API
"""
from fastapi import APIRouter, HTTPException, status
from app.models.collection import (
    CollectionInitRequest, 
    CollectionInitResponse,
    BotRegisterRequest,
    BotRegisterResponse
)
from app.utils.logger import setup_logger
from app.core.milvus_client import milvus_client
from app.core.postgres_client import postgres_client
from app.core.partition_manager import partition_manager
from app.config import settings

logger = setup_logger(__name__)
router = APIRouter()


def generate_partition_name(chat_bot_id: str) -> str:
    """chat_bot_id로 파티션명 자동 생성"""
    return f"bot_{chat_bot_id.replace('-', '')}"


@router.post("/create", response_model=CollectionInitResponse, status_code=status.HTTP_201_CREATED)
async def create_collection_endpoint(request: CollectionInitRequest):
    """
    컬렉션 생성 (최초 1회만)
    
    - **account_name**: 계정명 (예: chatty, enterprise)
    
    계정별 Milvus 컬렉션을 생성합니다 (예: collection_chatty)
    벡터 차원은 시스템 설정값(config.EMBEDDING_DIMENSION) 사용
    
    Note: 
    - 이미 존재하면 "already exists" 메시지 반환 (데이터는 그대로 유지)
    - 삭제는 별도 API로만 가능
    """
    try:
        collection_name = f"collection_{request.account_name}"
        dimension = settings.EMBEDDING_DIMENSION
        
        logger.info(f"컬렉션 생성 요청: account={request.account_name}, dimension={dimension} (시스템 설정)")
        
        # 1. PostgreSQL 데이터베이스 생성
        await postgres_client.create_database(request.account_name)
        logger.info(f"✅ PostgreSQL 데이터베이스 생성 완료")
        
        # 2. PostgreSQL 테이블 초기화
        await postgres_client.init_account_tables(request.account_name)
        logger.info(f"✅ PostgreSQL 테이블 초기화 완료")
        
        # 3. Milvus 컬렉션 생성 (이미 존재하면 예외 발생)
        try:
            milvus_client.create_collection(
                account_name=request.account_name,
                dimension=dimension
            )
            logger.info(f"✅ Milvus 컬렉션 생성 완료: {collection_name}")
            
            return CollectionInitResponse(
                status="success",
                message=f"Collection '{collection_name}' created successfully.",
                collection_name=collection_name
            )
            
        except Exception as create_error:
            # 이미 존재하는 경우
            if "already exist" in str(create_error).lower():
                logger.info(f"⚠️ 컬렉션이 이미 존재합니다: {collection_name}")
                return CollectionInitResponse(
                    status="success",
                    message=f"Collection '{collection_name}' already exists.",
                    collection_name=collection_name
                )
            else:
                raise create_error
                
    except Exception as e:
        logger.error(f"컬렉션 생성 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}"
        )


@router.post("/register-bot", response_model=BotRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_bot(request: BotRegisterRequest):
    """
    새로운 봇 등록 (파티션 자동 생성)
    
    - **account_name**: 계정명
    - **chat_bot_id**: 봇 ID (UUID)
    - **bot_name**: 봇 이름 (예: 뉴스봇)
    
    파티션명은 chat_bot_id로 자동 생성되며, PostgreSQL 파티션도 자동 생성됩니다!
    
    Note: 컬렉션(collection_chatty)은 /init API로 먼저 생성되어 있어야 합니다.
    """
    try:
        # 파티션명 자동 생성
        partition_name = generate_partition_name(request.chat_bot_id)
        collection_name = f"collection_{request.account_name}"
        
        logger.info(f"봇 등록 요청 (account: {request.account_name}): bot_id={request.chat_bot_id}, name={request.bot_name}, partition={partition_name}")
        
        # 1. PostgreSQL 봇 등록 (트리거로 PostgreSQL 파티션 자동 생성)
        await postgres_client.register_bot(
            account_name=request.account_name,
            bot_id=request.chat_bot_id,
            bot_name=request.bot_name,
            partition_name=partition_name,
            description=request.description if hasattr(request, 'description') else None,
            metadata=request.metadata if hasattr(request, 'metadata') else None
        )
        logger.info(f"✅ PostgreSQL 봇 등록 완료: {request.bot_name}")
        
        # 2. Milvus 파티션 생성 (collection_{account_name} 내)
        milvus_client.create_partition(
            account_name=request.account_name,
            partition_name=partition_name
        )
        logger.info(f"✅ Milvus 파티션 생성 완료: {partition_name}")
        
        # Note: 파티션 로드는 데이터 삽입/검색 시 자동으로 수행됨 (온디맨드)
        
        return BotRegisterResponse(
            status="success",
            message=f"Bot '{request.bot_name}' registered successfully.",
            partition_name=partition_name,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"컬렉션 생성 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}"
        )

