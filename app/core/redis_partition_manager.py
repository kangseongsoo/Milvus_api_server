"""
Redis 기반 Milvus 파티션 매니저
영구 상태 저장 및 TTL 관리
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Set, List, Optional, Any
from pymilvus import Collection, connections
from app.config import settings
from app.core.redis_client import redis_client
from app.utils.logger import setup_logger
import psutil

logger = setup_logger(__name__)


class RedisMilvusPartitionManager:
    """Redis 기반 Milvus 파티션 관리자"""
    
    def __init__(self):
        self.partition_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_running = False
        
    async def initialize(self):
        """Redis 연결 초기화"""
        await redis_client.connect()
        # Redis 클라이언트 초기화 후 self._partition_state_manager 생성
        from app.core.redis_client import PartitionStateManager
        self._partition_state_manager = PartitionStateManager(redis_client)
        logger.info("✅ Redis partition manager initialized")
    
    async def shutdown(self):
        """Redis 연결 해제"""
        await redis_client.disconnect()
        logger.info("✅ Redis partition manager shutdown")
    
    def _get_partition_key(self, collection_name: str, partition_name: str) -> str:
        """파티션 고유 키 생성"""
        return f"{collection_name}/{partition_name}"
    
    def _get_lock(self, collection_name: str, partition_name: str) -> asyncio.Lock:
        """파티션별 Lock 가져오기 (없으면 생성)"""
        key = self._get_partition_key(collection_name, partition_name)
        if key not in self.partition_locks:
            self.partition_locks[key] = asyncio.Lock()
        return self.partition_locks[key]
    
    async def ensure_partition_loaded(
        self, 
        collection_name: str, 
        partition_name: str,
        force_reload: bool = False
    ) -> bool:
        """
        파티션이 로드되었는지 확인하고, 없으면 로드 (동시 로드 방지)
        
        Args:
            collection_name: 컬렉션명
            partition_name: 파티션명
            force_reload: 강제 재로드 여부
        
        Returns:
            로드 성공 여부
        """
        key = self._get_partition_key(collection_name, partition_name)
        
        # Redis에서 로드 상태 확인
        is_loaded = await self._partition_state_manager.is_partition_loaded(collection_name, partition_name)
        
        if not force_reload and is_loaded:
            # 접근 시간 업데이트
            await self._partition_state_manager.update_access_time(collection_name, partition_name)
            logger.debug(f"✅ Partition already loaded: {key}")
            return True
        
        # Lock을 사용하여 동시 로드 방지
        async with self._get_lock(collection_name, partition_name):
            # Lock 대기 중 다른 스레드가 로드했을 수 있으므로 재확인
            is_loaded = await self._partition_state_manager.is_partition_loaded(collection_name, partition_name)
            if not force_reload and is_loaded:
                await self._partition_state_manager.update_access_time(collection_name, partition_name)
                logger.debug(f"✅ Partition loaded by another task: {key}")
                return True
            
            # 파티션 로드 시도
            try:
                start_time = datetime.now()
                logger.info(f"🔄 Loading partition: {key}")
                
                collection = Collection(name=collection_name)
                partition = collection.partition(partition_name)
                
                # 파티션 로드
                partition.load()
                
                # Redis에 상태 저장
                await self._partition_state_manager.set_partition_loaded(collection_name, partition_name)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Partition loaded successfully: {key} ({elapsed:.2f}s)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load partition {key}: {e}")
                return False
    
    async def unload_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        파티션 언로드
        
        Args:
            collection_name: 컬렉션명
            partition_name: 파티션명
        
        Returns:
            언로드 성공 여부
        """
        key = self._get_partition_key(collection_name, partition_name)
        
        try:
            logger.info(f"🔄 Unloading partition: {key}")
            
            collection = Collection(name=collection_name)
            partition = collection.partition(partition_name)
            
            # 파티션 언로드
            partition.release()
            
            # Redis에서 상태 제거
            await self._partition_state_manager.set_partition_unloaded(collection_name, partition_name)
            
            logger.info(f"✅ Partition unloaded successfully: {key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload partition {key}: {e}")
            return False
    
    async def reload_partitions_from_redis(self) -> Dict[str, Any]:
        """
        프로그램 시작 시 Redis에 저장된 파티션들을 다시 로드
        
        Returns:
            로드 결과 통계
        """
        try:
            logger.info("🔄 Starting partition reload from Redis...")
            start_time = datetime.now()
            
            # Redis에서 활성화된 파티션 목록 가져오기
            all_partitions = await self._partition_state_manager.get_all_loaded_partitions()
            
            if not all_partitions:
                logger.info("📦 No partitions found in Redis - nothing to reload")
                return {
                    "partitions_found": 0,
                    "partitions_reloaded": 0,
                    "collections_loaded": 0,
                    "reload_time_seconds": 0
                }
            
            logger.info(f"📦 Found {len(all_partitions)} partitions in Redis to reload")
            
            # 컬렉션별로 파티션 그룹화
            partitions_by_collection: Dict[str, List[str]] = {}
            partition_access_data: Dict[str, Dict[str, Any]] = {}
            
            for partition_key, partition_data in all_partitions.items():
                collection_name = partition_data.get("collection")
                partition_name = partition_data.get("partition")
                
                if not collection_name or not partition_name:
                    logger.warning(f"⚠️ Invalid partition data: {partition_key}")
                    continue
                
                if collection_name not in partitions_by_collection:
                    partitions_by_collection[collection_name] = []
                partitions_by_collection[collection_name].append(partition_name)
                partition_access_data[f"{collection_name}/{partition_name}"] = partition_data
            
            logger.info(f"📦 Grouped into {len(partitions_by_collection)} collections")
            
            reloaded_count = 0
            collections_loaded_count = 0
            skipped_count = 0
            
            from pymilvus import utility, Collection
            
            # 컬렉션별로 처리
            for collection_name, partition_names in partitions_by_collection.items():
                try:
                    logger.info(f"🔍 Processing collection: {collection_name} ({len(partition_names)} partitions)")
                    
                    collection = Collection(name=collection_name)
                    
                    # 1. 컬렉션 로드 상태 확인
                    try:
                        collection_load_state = utility.load_state(collection_name)
                        collection_is_loaded = (collection_load_state == utility.LoadState.Loaded)
                    except Exception:
                        collection_is_loaded = False
                    
                    if collection_is_loaded:
                        # 2. 컬렉션이 로드됨 → 각 파티션 로드 상태 확인 및 로드
                        logger.info(f"   → Collection is loaded, checking partitions individually")
                        for partition_name in partition_names:
                            try:
                                try:
                                    partition_load_state = utility.load_state(collection_name, partition_name)
                                    partition_is_loaded = (partition_load_state == utility.LoadState.Loaded)
                                except Exception:
                                    partition_is_loaded = False
                                
                                if not partition_is_loaded:
                                    # 3. 파티션이 로드 안되어있다면 로드
                                    logger.info(f"   → Partition {partition_name} not loaded, loading partition only")
                                    partition = collection.partition(partition_name)
                                    partition.load()
                                    reloaded_count += 1
                                    logger.info(f"   ✅ Partition reloaded: {partition_name}")
                                else:
                                    logger.info(f"   → Partition {partition_name} already loaded, skipping")
                                    skipped_count += 1
                            except Exception as partition_error:
                                logger.error(f"   ❌ Failed to reload partition {collection_name}/{partition_name}: {partition_error}")
                                continue
                    else:
                        # 4. 컬렉션이 로드 안되어있다면 collection.load(partition_names=[...])으로 한 번에 로드
                        logger.info(f"   → Collection not loaded, loading collection with {len(partition_names)} partitions")
                        collection.load(partition_names=partition_names)
                        collections_loaded_count += 1
                        reloaded_count += len(partition_names)
                        logger.info(f"   ✅ Collection loaded with {len(partition_names)} partitions")
                    
                    # 모든 파티션 접근 시간 업데이트 (컬렉션 로드 여부와 무관하게 한 번에)
                    for partition_name in partition_names:
                        await self._partition_state_manager.update_access_time(collection_name, partition_name)
                    
                except Exception as collection_error:
                    logger.error(f"   ❌ Failed to process collection {collection_name}: {collection_error}")
                    continue
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Partition reload from Redis completed in {elapsed_time:.2f}s")
            logger.info(f"   - Partitions found: {len(all_partitions)}")
            logger.info(f"   - Partitions reloaded: {reloaded_count}")
            logger.info(f"   - Collections loaded: {collections_loaded_count}")
            logger.info(f"   - Partitions skipped (already loaded): {skipped_count}")
            
            return {
                "partitions_found": len(all_partitions),
                "partitions_reloaded": reloaded_count,
                "collections_loaded": collections_loaded_count,
                "partitions_skipped": skipped_count,
                "reload_time_seconds": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to reload partitions from Redis: {e}")
            raise
    
    async def sync_partition_states(self, collection_names: List[str] = None) -> Dict[str, Any]:
        """
        Milvus 파티션 상태와 Redis 동기화
        
        Args:
            collection_names: 동기화할 컬렉션 목록 (None이면 모든 컬렉션 자동 조회)
        
        Returns:
            동기화 결과 통계
        """
        try:
            logger.info("🔄 Starting Redis partition state synchronization...")
            redis_connected = await redis_client.is_connected()
            logger.info(f"🔍 Redis connection status: {redis_connected}")
            if not redis_connected:
                logger.error("❌ Redis is not connected - cannot sync partition states")
                return {
                    "collections_checked": 0,
                    "partitions_synced": 0,
                    "sync_time_seconds": 0,
                    "error": "Redis not connected"
                }
            
            if not self._partition_state_manager:
                logger.error("❌ Partition state manager is not initialized")
                return {
                    "collections_checked": 0,
                    "partitions_synced": 0,
                    "sync_time_seconds": 0,
                    "error": "Partition state manager not initialized"
                }
            start_time = datetime.now()
            
            if collection_names is None:
                # Milvus에서 모든 컬렉션을 동적으로 조회
                logger.info("📋 Discovering all collections in Milvus...")
                from pymilvus import utility
                all_collections = utility.list_collections()
                
                # 시스템 컬렉션 제외
                collection_names = [name for name in all_collections if not name.startswith('_')]
                
                if not collection_names:
                    logger.info("⏭️  No collections found - skipping synchronization")
                    return {
                        "collections_checked": 0,
                        "partitions_synced": 0,
                        "sync_time_seconds": 0
                    }
            
            synced_count = 0
            collections_with_loaded_partitions = 0
            total_partitions_checked = 0
            
            logger.info(f"📦 Processing {len(collection_names)} collections...")
            
            for i, collection_name in enumerate(collection_names, 1):
                try:
                    logger.info(f"🔍 Checking collection: {collection_name}")
                    collection = Collection(name=collection_name)
                    
                    # 컬렉션의 모든 파티션 조회
                    partitions = collection.partitions
                    partition_names = [p.name for p in partitions if p.name != "_default"]
                    logger.info(f"   📋 Found {len(partition_names)} partitions: {partition_names}")
                    
                    if not partition_names:
                        logger.info(f"   ⏭️ No partitions found in {collection_name}")
                        continue
                    
                    # 컬렉션 로드 상태 확인 (다양한 방법 시도)
                    collection_loaded = False
                    try:
                        # 방법 1: has_collection 사용
                        from pymilvus import utility
                        collection_loaded = utility.has_collection(collection_name)
                        logger.info(f"   📊 Collection {collection_name} exists: {collection_loaded}")
                        
                        if collection_loaded:
                            # 방법 2: 컬렉션 속성 확인
                            try:
                                # 컬렉션이 로드되어 있는지 확인하는 다른 방법들
                                collection.load()  # 이미 로드되어 있으면 에러 없이 통과
                                collection_loaded = True
                                logger.info(f"   📊 Collection {collection_name} is already loaded")
                            except Exception as load_error:
                                logger.info(f"   📊 Collection {collection_name} load check: {load_error}")
                                # 이미 로드되어 있으면 특정 에러가 발생할 수 있음
                                if "already loaded" in str(load_error).lower():
                                    collection_loaded = True
                                    logger.info(f"   📊 Collection {collection_name} is already loaded (detected from error)")
                                else:
                                    collection_loaded = False
                                    logger.info(f"   📊 Collection {collection_name} is not loaded")
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to check collection load status: {e}")
                        collection_loaded = False
                    
                    # 컬렉션이 로드되어 있으면 모든 파티션을 로드된 것으로 간주
                    loaded_partitions = []
                    if collection_loaded:
                        loaded_partitions = partition_names.copy()
                        logger.info(f"   ✅ Collection is loaded - all partitions considered loaded: {loaded_partitions}")
                    else:
                        logger.info(f"   ❌ Collection is not loaded - no partitions loaded")
                    
                    total_partitions_checked += len(partition_names)
                    collection_loaded_count = 0
                    
                    for partition_name in partition_names:
                        try:
                            logger.info(f"   🔍 Checking partition: {collection_name}/{partition_name}")
                            
                            # 로드된 파티션 목록에서 확인
                            is_loaded = partition_name in loaded_partitions
                            logger.info(f"      📊 Partition {partition_name} loaded status: {is_loaded}")
                            
                            if is_loaded:
                                logger.info(f"      ✅ Partition {partition_name} is loaded - syncing to Redis")
                                # Redis에 상태 저장 (강제 업데이트)
                                await self._partition_state_manager.set_partition_loaded(collection_name, partition_name, force_update=True)
                                # 접근 시간을 현재 시간으로 설정 (서버 시작 시 동기화된 파티션)
                                await self._partition_state_manager.update_access_time(collection_name, partition_name)
                                synced_count += 1
                                collection_loaded_count += 1
                            else:
                                logger.info(f"      ❌ Partition {partition_name} is not loaded")
                                
                        except Exception as partition_error:
                            logger.error(f"      ❌ Failed to check partition {collection_name}/{partition_name}: {partition_error}")
                            continue
                    
                    # 컬렉션별 요약 로그
                    if collection_loaded_count > 0:
                        collections_with_loaded_partitions += 1
                        logger.info(f"   ✅ [{i:3d}/{len(collection_names)}] {collection_name}: {collection_loaded_count}/{len(partition_names)} partitions synced")
                    else:
                        logger.info(f"   ❌ [{i:3d}/{len(collection_names)}] {collection_name}: 0/{len(partition_names)} partitions synced")
                
                except Exception as collection_error:
                    logger.error(f"❌ Failed to sync collection {collection_name}: {collection_error}")
                    continue
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Redis partition state sync completed in {elapsed_time:.2f}s")
            logger.info(f"   - Collections checked: {len(collection_names)}")
            logger.info(f"   - Collections with loaded partitions: {collections_with_loaded_partitions}")
            logger.info(f"   - Total partitions checked: {total_partitions_checked}")
            logger.info(f"   - Partitions synced: {synced_count}")
            
            return {
                "collections_checked": len(collection_names),
                "collections_with_loaded_partitions": collections_with_loaded_partitions,
                "total_partitions_checked": total_partitions_checked,
                "partitions_synced": synced_count,
                "sync_time_seconds": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to sync partition states: {e}")
            raise
    
    async def auto_cleanup_loop(self):
        """
        백그라운드: TTL 기반 자동 정리 루프
        - 일정 시간마다 미사용 파티션 언로드
        - 메모리 임계값 초과 시 강제 정리
        """
        self._cleanup_running = True
        logger.info(f"🔄 Redis auto cleanup loop started (interval: {settings.CLEANUP_INTERVAL_SECONDS}s, TTL: {settings.PARTITION_TTL_MINUTES}m)")
        
        try:
            while self._cleanup_running:
                await asyncio.sleep(settings.CLEANUP_INTERVAL_SECONDS)
                
                try:
                    # 1. TTL 기반 정리
                    await self._cleanup_by_ttl()
                    
                    # 2. 메모리 임계값 체크
                    await self._check_memory_and_cleanup()
                    
                except Exception as e:
                    logger.error(f"❌ Cleanup error: {e}")
        
        finally:
            self._cleanup_running = False
            logger.info("🛑 Redis auto cleanup loop stopped")
    
    async def _cleanup_by_ttl(self):
        """TTL 기반 파티션 언로드"""
        try:
            # Redis에서 만료된 파티션 조회
            expired_partitions = await self._partition_state_manager.get_expired_partitions(settings.PARTITION_TTL_MINUTES)
            
            # 디버깅: 현재 상태 로그
            logger.info(f"🔍 Cleanup check: TTL={settings.PARTITION_TTL_MINUTES} minutes")
            logger.info(f"🔍 Found {len(expired_partitions)} expired partitions: {expired_partitions}")
            
            if not expired_partitions:
                logger.info("🔍 No expired partitions to cleanup")
                return
            
            logger.info(f"🗑️ TTL expired: unloading {len(expired_partitions)} partitions")
            
            # 만료된 파티션들을 일괄 언로드
            for partition_key in expired_partitions:
                try:
                    collection_name, partition_name = partition_key.split("/", 1)
                    await self.unload_partition(collection_name, partition_name)
                except Exception as e:
                    logger.error(f"❌ Failed to unload {partition_key}: {e}")
            
            # Redis에서 만료된 파티션 정리
            await self._partition_state_manager.cleanup_expired_partitions(settings.PARTITION_TTL_MINUTES)
            
        except Exception as e:
            logger.error(f"❌ TTL cleanup failed: {e}")
    
    async def _check_memory_and_cleanup(self):
        """메모리 임계값 체크 및 강제 정리"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > settings.MEMORY_THRESHOLD_PERCENT:
                logger.warning(f"⚠️ Memory threshold exceeded: {memory_percent:.1f}% > {settings.MEMORY_THRESHOLD_PERCENT}%")
                
                # Redis에서 모든 로드된 파티션 조회
                all_partitions = await self._partition_state_manager.get_all_loaded_partitions()
                
                if not all_partitions:
                    logger.info("🔍 No partitions to unload for memory cleanup")
                    return
                
                # 접근 시간 순으로 정렬 (오래된 것부터)
                sorted_partitions = sorted(
                    all_partitions.items(),
                    key=lambda x: x[1].get("access_time", datetime.min)
                )
                
                # 최소 25% 언로드
                unload_count = max(1, len(sorted_partitions) // 4)
                logger.info(f"🗑️ Force unloading {unload_count} oldest partitions")
                
                for partition_key, _ in sorted_partitions[:unload_count]:
                    try:
                        collection_name, partition_name = partition_key.split("/", 1)
                        await self.unload_partition(collection_name, partition_name)
                        
                        # 메모리 재확인
                        memory = psutil.virtual_memory()
                        if memory.percent < settings.MEMORY_THRESHOLD_PERCENT:
                            logger.info(f"✅ Memory stabilized: {memory.percent:.1f}%")
                            break
                    except Exception as e:
                        logger.error(f"❌ Failed to unload {partition_key}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
    
    async def stop_cleanup_loop(self):
        """Cleanup 루프 중지"""
        self._cleanup_running = False
    
    async def get_status(self) -> Dict[str, Any]:
        """파티션 상태 조회"""
        try:
            # Redis에서 모든 로드된 파티션 조회
            all_partitions = await self._partition_state_manager.get_all_loaded_partitions()
            
            # 통계 계산
            total_loaded = len(all_partitions)
            collections = set()
            oldest_partition = None
            oldest_time = None
            
            if all_partitions:
                for partition_key, data in all_partitions.items():
                    collection_name = data.get("collection", "")
                    collections.add(collection_name)
                    
                    access_time = data.get("access_time")
                    if access_time and (not oldest_time or access_time < oldest_time):
                        oldest_time = access_time
                        oldest_partition = partition_key
            
            return {
                "total_loaded_partitions": total_loaded,
                "collections_with_loaded_partitions": len(collections),
                "collections": list(collections),
                "oldest_partition": {
                    "key": oldest_partition,
                    "access_time": oldest_time.isoformat() if oldest_time else None,
                    "minutes_ago": int((datetime.now() - oldest_time).total_seconds() / 60) if oldest_time else None
                } if oldest_partition else None,
                "loaded_partitions": all_partitions,
                "config": {
                    "ttl_minutes": settings.PARTITION_TTL_MINUTES,
                    "cleanup_interval_seconds": settings.CLEANUP_INTERVAL_SECONDS,
                    "memory_threshold_percent": settings.MEMORY_THRESHOLD_PERCENT
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get status: {e}")
            return {
                "error": str(e),
                "total_loaded_partitions": 0,
                "collections_with_loaded_partitions": 0
            }
    
    async def update_partition_access_time(self, collection_name: str, partition_name: str):
        """파티션 접근 시간 업데이트 (TTL 정리용)"""
        await self._partition_state_manager.update_access_time(collection_name, partition_name)


# 전역 Redis 기반 파티션 매니저 인스턴스
redis_partition_manager = RedisMilvusPartitionManager()
