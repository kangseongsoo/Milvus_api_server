"""
FastAPI 애플리케이션 진입점
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api import collection, data, search
from fastapi import APIRouter
from app.utils.logger import setup_logger
from app.core.redis_partition_manager import redis_partition_manager
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
        
        # Redis 파티션 매니저 초기화
        await redis_partition_manager.initialize()
        logger.info("✅ Redis partition manager initialized")
        
        # ⭐ 프로그램 시작 시 Redis에 저장된 파티션들 다시 로드
        logger.info("📦 Reloading partitions from Redis...")
        try:
            reload_result = await redis_partition_manager.reload_partitions_from_redis()
            logger.info(f"✅ Partition reload completed: {reload_result.get('partitions_reloaded', 0)} partitions reloaded")
        except Exception as reload_error:
            logger.error(f"❌ Failed to reload partitions from Redis: {reload_error}")
            # 에러가 발생해도 서버는 계속 시작하도록 함
        
        # Redis 기반 파티션 자동 정리 백그라운드 태스크 시작
        cleanup_task = asyncio.create_task(redis_partition_manager.auto_cleanup_loop())
        logger.info(f"✅ Redis auto partition cleanup started (TTL: {settings.PARTITION_TTL_MINUTES}m, interval: {settings.CLEANUP_INTERVAL_SECONDS}s)")
        
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
        # Redis 파티션 cleanup 중지
        await redis_partition_manager.stop_cleanup_loop()
        if 'cleanup_task' in locals():
            cleanup_task.cancel()
            try:
                await asyncio.wait_for(cleanup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.info("✅ Redis partition cleanup stopped")
        
        # Redis 파티션 매니저 종료
        await redis_partition_manager.shutdown()
        logger.info("✅ Redis partition manager shutdown")
        
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

# 디버깅용 라우터
debug_router = APIRouter()

@debug_router.get("/partitions/status")
async def get_partition_status():
    """파티션 상태 확인 (디버깅용)"""
    return await redis_partition_manager.get_status()

@debug_router.post("/partitions/cleanup")
async def trigger_cleanup():
    """수동으로 정리 실행 (디버깅용)"""
    await redis_partition_manager._cleanup_by_ttl()
    return {"message": "Redis cleanup triggered"}

@debug_router.get("/count/{collection_name}")
async def count_entities(collection_name: str):
    """컬렉션 및 파티션별 벡터 개수 확인 (디버깅용)"""
    from pymilvus import Collection
    try:
        collection = Collection(name=collection_name)
        collection.flush()  # 최신 데이터 반영
        
        # 전체 개수
        total = collection.num_entities
        
        # 파티션별 개수
        partition_counts = {}
        for partition in collection.partitions:
            try:
                count = partition.num_entities
                partition_counts[partition.name] = count
            except Exception as e:
                partition_counts[partition.name] = f"Error: {str(e)}"
        
        return {
            "collection": collection_name,
            "total_entities": total,
            "partitions": partition_counts,
            "status": "success"
        }
    except Exception as e:
        return {"message": str(e), "status": "error"}

@debug_router.post("/flush/{collection_name}")
async def manual_flush(collection_name: str):
    """수동 flush 실행 (디버깅용)"""
    from pymilvus import Collection
    try:
        collection = Collection(name=collection_name)
        collection.load()  # 컬렉션 로드
        collection.flush()  # Flush
        return {"message": f"Flushed {collection_name}", "status": "success"}
    except Exception as e:
        return {"message": str(e), "status": "error"}

app.include_router(debug_router, prefix="/debug", tags=["Debug"])




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
    partition_stats = await redis_partition_manager.get_status()
    
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

