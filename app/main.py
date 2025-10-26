"""
FastAPI 애플리케이션 진입점
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api import collection, data, search
from app.utils.logger import setup_logger
from app.core.partition_manager import partition_manager
from app.core.auto_flusher import auto_flusher
import asyncio

# 로거 설정
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리"""
    
    # ========== 시작 시 실행 ==========
    logger.info("🚀 FastAPI Application Starting...")
    
    try:
        # Milvus 연결
        from pymilvus import connections
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        logger.info(f"✅ Connected to Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})")
        
        # PostgreSQL 연결 테스트
        logger.info(f"✅ PostgreSQL Connected to ({settings.POSTGRES_HOST}:{settings.POSTGRES_PORT})")
        
        # ⭐ 파티션 상태 동기화 (FastAPI 재시작 후 상태 불일치 해결)
        logger.info("🔄 Syncing partition states with Milvus...")
        try:
            sync_result = await partition_manager.sync_partition_states()
            logger.info(f"✅ Partition state sync completed: {sync_result['partitions_synced']} partitions synced")
        except Exception as sync_error:
            logger.warning(f"⚠️ Partition state sync failed: {sync_error}")
            logger.info("📦 Will continue with on-demand loading")
        
        # ⭐ 사전 로드 비활성화 (요청 시 동적 로드)
        # collections_to_preload = ["collection_chatty"]
        # logger.info(f"📦 Preloading {len(collections_to_preload)} collections...")
        # tasks = [partition_manager.preload_collection(coll_name) for coll_name in collections_to_preload]
        # await asyncio.gather(*tasks)
        logger.info("📦 Partition preload disabled - will load on-demand")
        
        # 파티션 자동 정리 백그라운드 태스크 시작
        cleanup_task = asyncio.create_task(partition_manager.auto_cleanup_loop())
        logger.info(f"✅ Auto partition cleanup started (TTL: {settings.PARTITION_TTL_MINUTES}m, interval: {settings.CLEANUP_INTERVAL_SECONDS}s)")
        
        # 자동 flush 백그라운드 태스크 시작
        flush_task = asyncio.create_task(auto_flusher.start())
        logger.info(f"✅ Auto-flusher started (delay: {auto_flusher.delay_seconds}s, max_wait: {auto_flusher.max_wait_seconds}s)")
        
        logger.info("🎉 FastAPI Application Ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # ========== 종료 시 실행 ==========
    logger.info("🛑 FastAPI Application Shutting Down...")
    
    try:
        # 파티션 cleanup 중지
        await partition_manager.stop_cleanup_loop()
        if 'cleanup_task' in locals():
            cleanup_task.cancel()
            try:
                await asyncio.wait_for(cleanup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.info("✅ Partition cleanup stopped")
        
        # 자동 flush 중지
        await auto_flusher.stop()
        if 'flush_task' in locals():
            flush_task.cancel()
            try:
                await asyncio.wait_for(flush_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.info("✅ Auto-flusher stopped")
        
        # 약간의 대기 (백그라운드 태스크 정리 완료 대기)
        await asyncio.sleep(0.1)
        
        # Milvus 연결 해제
        try:
            connections.disconnect(alias="default")
            logger.info("✅ Disconnected from Milvus")
        except Exception as disconnect_error:
            logger.debug(f"Milvus disconnect: {disconnect_error}")
        
    except asyncio.CancelledError:
        # 정상적인 종료 시그널 - 에러 아님
        logger.info("✅ Graceful shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    
    logger.info("👋 FastAPI Application Stopped")


# FastAPI 앱 생성
app = FastAPI(
    title="Milvus RAG API Server",
    description="RAG 시스템을 위한 Milvus + PostgreSQL 하이브리드 백엔드",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(collection.router, prefix="/collection", tags=["Collection"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(search.router, prefix="/search", tags=["Search"])




@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "Milvus RAG API Server",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """상세 헬스 체크 (파티션 통계 포함)"""
    partition_stats = partition_manager.get_partition_stats()
    
    return {
        "status": "healthy",
        "milvus": {
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT
        },
        "postgres": {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT
        },
        "embedding": {
            "model": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION
        },
        "partitions": partition_stats
    }

