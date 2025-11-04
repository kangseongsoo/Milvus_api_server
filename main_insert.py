"""
FastAPI 삽입 서버 (Insert API Server)
- 데이터 삽입/삭제 API
- 컬렉션 관리 API
- 자동 flush 기능
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api import collection, data
from fastapi import APIRouter
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
    logger.info("🚀 FastAPI Insert Server Starting...")
    
    try:
        # Milvus 연결 (milvus_client.py가 "default" alias 사용)
        from pymilvus import connections
        connections.connect(
            alias="default",  # milvus_client.py와 일치해야 함
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        logger.info(f"✅ Connected to Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})")
        
        # PostgreSQL 연결 테스트
        logger.info(f"✅ PostgreSQL Connected to ({settings.POSTGRES_HOST}:{settings.POSTGRES_PORT})")
        
        # 모든 컬렉션 전체 로드 (시작 시 한 번만)
        logger.info("🔄 Loading all collections...")
        preload_result = await partition_manager.preload_all_collections()
        logger.info(f"✅ All collections loaded: {preload_result['collections_loaded']} collections, {preload_result['total_partitions']} partitions")
        
        # 자동 flush 백그라운드 태스크 시작 (삽입 서버에만 필요)
        flush_task = asyncio.create_task(auto_flusher.start())
        logger.info(f"✅ Auto-flusher started (delay: {auto_flusher.delay_seconds}s, max_wait: {auto_flusher.max_wait_seconds}s)")
        
        logger.info("🎉 FastAPI Insert Server Ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # ========== 종료 시 실행 ==========
    logger.info("🛑 FastAPI Insert Server Shutting Down...")
    
    try:
        # 파티션 매니저 정리 (Redis 없이 동작하므로 별도 종료 불필요)
        logger.info("✅ Partition manager cleanup completed")
        
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
    
    logger.info("👋 FastAPI Insert Server Stopped")


# FastAPI 앱 생성
app = FastAPI(
    title="Milvus RAG API Server - Insert",
    description="RAG 시스템을 위한 Milvus 데이터 삽입 서버 (Insert/Delete/Collection Management)",
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

# 삽입 서버 라우터 등록
app.include_router(collection.router, prefix="/collection", tags=["Collection"])
app.include_router(data.router, prefix="/data", tags=["Data"])

# 디버깅용 라우터
debug_router = APIRouter()

@debug_router.get("/partitions/status")
async def get_partition_status():
    """파티션 상태 확인 (디버깅용)"""
    # 메모리 기반 파티션 상태 조회
    all_partitions = {}
    for collection_name, partition_names in partition_manager.loaded_partitions.items():
        for partition_name in partition_names:
            key = f"{collection_name}/{partition_name}"
            all_partitions[key] = {
                "collection": collection_name,
                "partition": partition_name,
                "status": "loaded"
            }
    
    return {
        "total_loaded_partitions": sum(len(partitions) for partitions in partition_manager.loaded_partitions.values()),
        "collections_with_loaded_partitions": len(partition_manager.loaded_partitions),
        "loaded_partitions": all_partitions
    }

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

@debug_router.get("/flush/status")
async def get_flush_status():
    """Auto-flusher 상태 확인 (디버깅용)"""
    return auto_flusher.get_status()

app.include_router(debug_router, prefix="/debug", tags=["Debug"])


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "Milvus RAG API Server - Insert",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """상세 헬스 체크 (파티션 통계 포함)"""
    # 파티션 통계 (메모리 기반)
    partition_stats = {
        "total_loaded_partitions": sum(len(partitions) for partitions in partition_manager.loaded_partitions.values()),
        "collections_with_loaded_partitions": len(partition_manager.loaded_partitions),
        "collections": list(partition_manager.loaded_partitions.keys())
    }
    flush_stats = auto_flusher.get_status()
    
    return {
        "status": "healthy",
        "service": "insert",
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
        "partitions": partition_stats,
        "auto_flusher": flush_stats
    }
