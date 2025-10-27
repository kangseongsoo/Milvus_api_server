"""
Redis 클라이언트 관리
파티션 상태 및 TTL 관리를 위한 Redis 연결 및 유틸리티
"""
import redis.asyncio as redis
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import asyncio
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisClient:
    """Redis 클라이언트 래퍼"""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        
    async def connect(self):
        """Redis 연결"""
        try:
            # 연결 풀 생성
            self.connection_pool = redis.ConnectionPool.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=True
            )
            
            # Redis 클라이언트 생성
            self.redis = redis.Redis(connection_pool=self.connection_pool)
            
            # 연결 테스트
            await self.redis.ping()
            logger.info(f"✅ Redis connected to {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Redis 연결 해제"""
        if self.redis:
            await self.redis.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        logger.info("✅ Redis disconnected")
    
    async def is_connected(self) -> bool:
        """Redis 연결 상태 확인"""
        try:
            if self.redis:
                await self.redis.ping()
                return True
        except:
            pass
        return False


class PartitionStateManager:
    """Redis 기반 파티션 상태 관리"""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client.redis
        self.partition_key_prefix = "milvus:partitions"
        self.access_time_key_prefix = "milvus:access_time"
        self.load_time_key_prefix = "milvus:load_time"
        
    def _get_partition_key(self, collection_name: str, partition_name: str) -> str:
        """파티션 고유 키 생성"""
        return f"{self.partition_key_prefix}:{collection_name}:{partition_name}"
    
    def _get_access_time_key(self, collection_name: str, partition_name: str) -> str:
        """접근 시간 키 생성"""
        return f"{self.access_time_key_prefix}:{collection_name}:{partition_name}"
    
    def _get_load_time_key(self, collection_name: str, partition_name: str) -> str:
        """로드 시간 키 생성"""
        return f"{self.load_time_key_prefix}:{collection_name}:{partition_name}"
    
    async def set_partition_loaded(self, collection_name: str, partition_name: str, force_update: bool = False):
        """파티션 로드 상태 설정"""
        if not self.redis:
            logger.error("❌ Redis client is not initialized")
            return
            
        now = datetime.now()
        partition_key = self._get_partition_key(collection_name, partition_name)
        access_key = self._get_access_time_key(collection_name, partition_name)
        load_key = self._get_load_time_key(collection_name, partition_name)
        
        # 기존 상태 확인 (force_update가 False인 경우)
        if not force_update:
            exists = await self.redis.exists(partition_key)
            if exists:
                logger.debug(f"🔄 Partition already exists in Redis: {collection_name}/{partition_name} - skipping")
                return
        else:
            logger.info(f"🔄 Force update requested for partition: {collection_name}/{partition_name}")
        
        # 파이프라인으로 원자적 업데이트
        try:
            pipe = self.redis.pipeline()
            pipe.hset(partition_key, mapping={
                "collection": collection_name,
                "partition": partition_name,
                "status": "loaded"
            })
            pipe.set(access_key, now.isoformat())
            pipe.set(load_key, now.isoformat())
            pipe.expire(partition_key, 3600)  # 1시간 TTL
            pipe.expire(access_key, 3600)
            pipe.expire(load_key, 3600)
            
            logger.info(f"🔄 Executing Redis pipeline for partition: {collection_name}/{partition_name}")
            result = await pipe.execute()
            logger.info(f"🔄 Redis pipeline result: {result}")
            logger.info(f"🔄 Partition state {'updated' if force_update else 'saved'}: {collection_name}/{partition_name}")
        except Exception as pipe_error:
            logger.error(f"❌ Redis pipeline failed for partition {collection_name}/{partition_name}: {pipe_error}")
            raise
    
    async def set_partition_unloaded(self, collection_name: str, partition_name: str):
        """파티션 언로드 상태 설정"""
        if not self.redis:
            return
            
        partition_key = self._get_partition_key(collection_name, partition_name)
        access_key = self._get_access_time_key(collection_name, partition_name)
        load_key = self._get_load_time_key(collection_name, partition_name)
        
        # 파이프라인으로 원자적 삭제
        pipe = self.redis.pipeline()
        pipe.delete(partition_key)
        pipe.delete(access_key)
        pipe.delete(load_key)
        
        await pipe.execute()
        logger.debug(f"🗑️ Partition state removed: {collection_name}/{partition_name}")
    
    async def update_access_time(self, collection_name: str, partition_name: str):
        """파티션 접근 시간 업데이트"""
        if not self.redis:
            return
            
        now = datetime.now()
        access_key = self._get_access_time_key(collection_name, partition_name)
        
        await self.redis.set(access_key, now.isoformat(), ex=3600)  # 1시간 TTL
        logger.debug(f"🕒 Access time updated: {collection_name}/{partition_name}")
    
    async def is_partition_loaded(self, collection_name: str, partition_name: str) -> bool:
        """파티션 로드 상태 확인"""
        if not self.redis:
            return False
            
        partition_key = self._get_partition_key(collection_name, partition_name)
        exists = await self.redis.exists(partition_key)
        return bool(exists)
    
    async def get_partition_access_time(self, collection_name: str, partition_name: str) -> Optional[datetime]:
        """파티션 마지막 접근 시간 조회"""
        if not self.redis:
            return None
            
        access_key = self._get_access_time_key(collection_name, partition_name)
        time_str = await self.redis.get(access_key)
        
        if time_str:
            try:
                return datetime.fromisoformat(time_str.decode())
            except:
                return None
        return None
    
    async def get_all_loaded_partitions(self) -> Dict[str, Dict[str, Any]]:
        """모든 로드된 파티션 조회"""
        if not self.redis:
            return {}
            
        pattern = f"{self.partition_key_prefix}:*"
        keys = await self.redis.keys(pattern)
        
        partitions = {}
        for key in keys:
            try:
                key_str = key.decode()
                # 키에서 collection_name과 partition_name 추출
                parts = key_str.split(":")
                if len(parts) >= 4:
                    collection_name = parts[2]
                    partition_name = parts[3]
                    
                    # 파티션 정보 조회
                    partition_data = await self.redis.hgetall(key)
                    access_time = await self.get_partition_access_time(collection_name, partition_name)
                    
                    partitions[f"{collection_name}/{partition_name}"] = {
                        "collection": collection_name,
                        "partition": partition_name,
                        "status": partition_data.get(b"status", b"unknown").decode(),
                        "access_time": access_time
                    }
            except Exception as e:
                logger.warning(f"Failed to parse partition key {key}: {e}")
                continue
                
        return partitions
    
    async def get_expired_partitions(self, ttl_minutes: int) -> list:
        """만료된 파티션 조회"""
        if not self.redis:
            return []
            
        now = datetime.now()
        ttl_threshold = timedelta(minutes=ttl_minutes)
        expired = []
        
        partitions = await self.get_all_loaded_partitions()
        logger.info(f"🔍 Checking {len(partitions)} partitions for expiration")
        
        for key, data in partitions.items():
            access_time = data.get("access_time")
            if access_time:
                time_diff = now - access_time
                logger.info(f"🔍 Partition {key}: access_time={access_time}, diff={time_diff}, threshold={ttl_threshold}")
                if time_diff > ttl_threshold:
                    expired.append(key)
                    logger.info(f"🔍 Partition {key} is EXPIRED")
                else:
                    logger.info(f"🔍 Partition {key} is NOT expired")
            else:
                logger.warning(f"🔍 Partition {key} has no access_time")
                
        return expired
    
    async def cleanup_expired_partitions(self, ttl_minutes: int) -> int:
        """만료된 파티션 정리"""
        if not self.redis:
            return 0
            
        expired = await self.get_expired_partitions(ttl_minutes)
        
        if not expired:
            logger.info("🔍 No expired partitions to cleanup")
            return 0
        
        logger.info(f"🗑️ Cleaning up {len(expired)} expired partitions")
        
        # 파이프라인으로 일괄 삭제
        pipe = self.redis.pipeline()
        for partition_key in expired:
            collection_name, partition_name = partition_key.split("/", 1)
            partition_key_redis = self._get_partition_key(collection_name, partition_name)
            access_key = self._get_access_time_key(collection_name, partition_name)
            load_key = self._get_load_time_key(collection_name, partition_name)
            
            pipe.delete(partition_key_redis)
            pipe.delete(access_key)
            pipe.delete(load_key)
        
        await pipe.execute()
        logger.info(f"✅ Cleaned up {len(expired)} expired partitions")
        return len(expired)


# 전역 Redis 클라이언트 인스턴스
redis_client = RedisClient()
partition_state_manager = PartitionStateManager(redis_client)
