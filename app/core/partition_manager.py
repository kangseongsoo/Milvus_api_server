"""
Milvus 파티션 로드 관리자
- FastAPI 시작 시 컬렉션 전체 로드
- 신규 파티션 실시간 로드
- 파티션 언로드 및 메모리 관리
"""

import asyncio
import logging
import psutil
from typing import Dict, Set, Optional
from pymilvus import Collection
from pymilvus.exceptions import SchemaNotReadyException
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)


class MilvusPartitionManager:
    """Milvus 파티션 로드 및 관리 (TTL + 메모리 임계값 기반)"""
    
    def __init__(self):
        self.loaded_partitions: Dict[str, Set[str]] = {}  # {collection_name: {partition_names}}
        self.partition_load_time: Dict[str, datetime] = {}  # 로드 시간 추적
        self.last_access_time: Dict[str, datetime] = {}  # 마지막 접근 시간 (TTL용)
        self.partition_locks: Dict[str, asyncio.Lock] = {}  # 파티션별 Lock (동시 로드 방지)
        self._cleanup_running = False  # cleanup 루프 상태
        
    async def preload_collection(self, collection_name: str):
        """
        FastAPI 시작 시 컬렉션의 모든 파티션 로드
        
        Args:
            collection_name: 로드할 컬렉션명 (예: "collection_chatty")
        """
        try:
            logger.info(f"🔄 Starting preload for collection: {collection_name}")
            start_time = datetime.now()
            
            # 컬렉션 연결
            collection = Collection(name=collection_name)
            
            # 모든 파티션 목록 가져오기
            partitions = collection.partitions
            partition_names = [p.name for p in partitions if p.name != "_default"]
            
            if not partition_names:
                logger.warning(f"⚠️ No partitions found in {collection_name}")
                self.loaded_partitions[collection_name] = set()
                return
            
            logger.info(f"📦 Found {len(partition_names)} partitions to load")
            
            # 모든 파티션 로드 (병렬 처리)
            await self._load_partitions_parallel(collection, partition_names)
            
            # 로드된 파티션 추적
            self.loaded_partitions[collection_name] = set(partition_names)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Preload completed in {elapsed_time:.2f}s")
            logger.info(f"   - Collection: {collection_name}")
            logger.info(f"   - Partitions loaded: {len(partition_names)}")
            logger.info(f"   - Estimated memory: ~{len(partition_names) * 25}GB")
            
        except SchemaNotReadyException as e:
            logger.warning(f"⚠️ Collection '{collection_name}' does not exist - skipping preload")
            return
        except Exception as e:
            logger.error(f"❌ Failed to preload collection {collection_name}: {e}")
            raise
    
    async def _load_partitions_parallel(
        self, 
        collection: Collection, 
        partition_names: list[str],
        batch_size: int = 5
    ):
        """
        파티션 병렬 로드 (배치 단위)
        
        Args:
            collection: Milvus 컬렉션
            partition_names: 로드할 파티션 이름 리스트
            batch_size: 동시 로드할 파티션 수
        """
        total = len(partition_names)
        
        for i in range(0, total, batch_size):
            batch = partition_names[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Loading batch {batch_num}/{total_batches}: {len(batch)} partitions")
            
            # 배치 단위로 동시 로드
            tasks = [
                self._load_single_partition(collection, pname) 
                for pname in batch
            ]
            await asyncio.gather(*tasks)
    
    async def _load_single_partition(self, collection: Collection, partition_name: str):
        """단일 파티션 로드"""
        try:
            start = datetime.now()
            
            # 이미 로드되어 있는지 확인
            partition = collection.partition(partition_name)
            try:
                if hasattr(partition, 'is_loaded') and partition.is_loaded:
                    logger.debug(f"   ⏭️  Already loaded: {partition_name}")
                    return
            except:
                # is_loaded 속성이 없거나 오류 발생 시 무시하고 로드 진행
                pass
            
            # 파티션 로드
            collection.load(partition_names=[partition_name])
            
            # 로드 시간 기록
            elapsed = (datetime.now() - start).total_seconds()
            self.partition_load_time[partition_name] = datetime.now()
            
            logger.info(f"   ✅ Loaded {partition_name} in {elapsed:.2f}s ({partition.num_entities:,} entities)")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load {partition_name}: {e}")
    
    async def load_new_partition(self, collection_name: str, partition_name: str):
        """
        새 파티션 실시간 로드 (봇 추가 시)
        
        Args:
            collection_name: 컬렉션명
            partition_name: 새로운 파티션명
        """
        try:
            logger.info(f"🆕 Loading new partition: {collection_name}/{partition_name}")
            
            collection = Collection(name=collection_name)
            await self._load_single_partition(collection, partition_name)
            
            # 추적 리스트에 추가
            if collection_name not in self.loaded_partitions:
                self.loaded_partitions[collection_name] = set()
            self.loaded_partitions[collection_name].add(partition_name)
            
            logger.info(f"✅ New partition loaded and tracked: {partition_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load new partition {partition_name}: {e}")
            raise
    
    async def unload_partition(self, collection_name: str, partition_name: str):
        """
        파티션 언로드 (봇 삭제 시)
        
        Args:
            collection_name: 컬렉션명
            partition_name: 삭제할 파티션명
        """
        try:
            logger.info(f"🗑️  Unloading partition: {collection_name}/{partition_name}")
            
            collection = Collection(name=collection_name)
            
            # 파티션 언로드
            partition = collection.partition(partition_name)
            try:
                if hasattr(partition, 'is_loaded') and partition.is_loaded:
                    collection.release(partition_names=[partition_name])
                    logger.info(f"   ✅ Released memory for {partition_name}")
            except:
                # is_loaded 속성이 없거나 오류 발생 시 강제 언로드
                collection.release(partition_names=[partition_name])
                logger.info(f"   ✅ Released memory for {partition_name}")
            
            # 추적 리스트에서 제거
            if collection_name in self.loaded_partitions:
                self.loaded_partitions[collection_name].discard(partition_name)
            
            # 로드 시간 기록 제거
            self.partition_load_time.pop(partition_name, None)
            
            logger.info(f"✅ Partition unloaded: {partition_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to unload partition {partition_name}: {e}")
            raise
    
    def get_loaded_partitions(self, collection_name: str) -> Set[str]:
        """로드된 파티션 목록 조회"""
        return self.loaded_partitions.get(collection_name, set())
    
    def is_partition_loaded(self, collection_name: str, partition_name: str) -> bool:
        """파티션 로드 상태 확인"""
        return partition_name in self.loaded_partitions.get(collection_name, set())
    
    def get_load_time(self, partition_name: str) -> datetime | None:
        """파티션 로드 시간 조회"""
        return self.partition_load_time.get(partition_name)
    
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
        
        Note:
            - Lock을 사용하여 동일 파티션 중복 로드 방지
            - 마지막 접근 시간 자동 갱신 (TTL용)
        """
        key = self._get_partition_key(collection_name, partition_name)
        lock = self._get_lock(collection_name, partition_name)
        
        # 이미 로드되어 있으면 접근 시간만 갱신
        if not force_reload and self.is_partition_loaded(collection_name, partition_name):
            self.last_access_time[key] = datetime.now()
            logger.debug(f"✅ Partition already loaded: {key}")
            return True
        
        # Lock 획득하여 동시 로드 방지
        async with lock:
            # Lock 대기 중 다른 스레드가 로드했을 수 있으므로 재확인
            if not force_reload and self.is_partition_loaded(collection_name, partition_name):
                self.last_access_time[key] = datetime.now()
                logger.debug(f"✅ Partition loaded by another task: {key}")
                return True
            
            # 메모리 체크 후 필요 시 정리
            await self._check_memory_and_cleanup()
            
            # 파티션 로드
            try:
                logger.info(f"🔄 Loading partition: {key}")
                start_time = datetime.now()
                
                collection = Collection(name=collection_name)
                partition = collection.partition(partition_name)
                
                # 이미 로드되어 있으면 스킵
                try:
                    if not force_reload and hasattr(partition, 'is_loaded') and partition.is_loaded:
                        logger.debug(f"   ⏭️  Already loaded: {partition_name}")
                    else:
                        collection.load(partition_names=[partition_name])
                except:
                    # is_loaded 속성이 없거나 오류 발생 시 강제 로드
                    collection.load(partition_names=[partition_name])
                
                # 추적 정보 업데이트
                if collection_name not in self.loaded_partitions:
                    self.loaded_partitions[collection_name] = set()
                self.loaded_partitions[collection_name].add(partition_name)
                
                now = datetime.now()
                self.partition_load_time[key] = now
                self.last_access_time[key] = now
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Partition loaded in {elapsed:.2f}s: {key} ({partition.num_entities:,} entities)")
                
                return True
                
            except SchemaNotReadyException:
                logger.warning(f"⚠️ Collection '{collection_name}' does not exist")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to load partition {key}: {e}")
                return False
    
    async def auto_cleanup_loop(self):
        """
        백그라운드: TTL 기반 자동 정리 루프
        - 일정 시간마다 미사용 파티션 언로드
        - 메모리 임계값 초과 시 강제 정리
        """
        self._cleanup_running = True
        logger.info(f"🔄 Auto cleanup loop started (interval: {settings.CLEANUP_INTERVAL_SECONDS}s, TTL: {settings.PARTITION_TTL_MINUTES}m)")
        
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
            logger.info("🛑 Auto cleanup loop stopped")
    
    async def stop_cleanup_loop(self):
        """Cleanup 루프 중지"""
        self._cleanup_running = False
    
    async def sync_partition_states(self, collection_names: list[str] = None):
        """
        FastAPI 시작 시 Milvus 파티션 상태와 동기화
        
        Args:
            collection_names: 동기화할 컬렉션 목록 (None이면 모든 컬렉션 자동 조회)
        
        Note:
            - Milvus에 실제로 로드된 파티션들을 FastAPI 메모리에 동기화
            - FastAPI 재시작 후 상태 불일치 문제 해결
            - 동적으로 모든 컬렉션을 자동으로 찾아서 처리
        """
        try:
            logger.info("🔄 Starting partition state synchronization...")
            start_time = datetime.now()
            
            if collection_names is None:
                # Milvus에서 모든 컬렉션을 동적으로 조회
                logger.info("📋 Discovering all collections in Milvus...")
                from pymilvus import utility
                all_collections = utility.list_collections()
                
                # 시스템 컬렉션 제외 (필요시)
                collection_names = [name for name in all_collections if not name.startswith('_')]
                
                if not collection_names:
                    logger.info("⏭️  No collections found - skipping synchronization")
                    return {
                        "collections_checked": 0,
                        "partitions_synced": 0,
                        "sync_time_seconds": 0,
                        "estimated_memory_gb": 0
                    }
            
            synced_count = 0
            collections_with_loaded_partitions = 0
            total_partitions_checked = 0
            
            # 대량 컬렉션 처리를 위한 간결한 로그
            logger.info(f"📦 Processing {len(collection_names)} collections...")
            
            for i, collection_name in enumerate(collection_names, 1):
                try:
                    collection = Collection(name=collection_name)
                    
                    # 컬렉션의 모든 파티션 조회
                    partitions = collection.partitions
                    partition_names = [p.name for p in partitions if p.name != "_default"]
                    
                    if not partition_names:
                        continue
                    
                    total_partitions_checked += len(partition_names)
                    collection_loaded_count = 0
                    
                    for partition_name in partition_names:
                        try:
                            partition = collection.partition(partition_name)
                            
                            # Milvus에서 실제 로드 상태 확인
                            if hasattr(partition, 'is_loaded') and partition.is_loaded:
                                # FastAPI 메모리에 동기화
                                if collection_name not in self.loaded_partitions:
                                    self.loaded_partitions[collection_name] = set()
                                
                                self.loaded_partitions[collection_name].add(partition_name)
                                
                                # 접근 시간을 현재 시간으로 설정 (TTL 정리 방지)
                                key = self._get_partition_key(collection_name, partition_name)
                                self.last_access_time[key] = datetime.now()
                                
                                # 로드 시간도 현재 시간으로 설정
                                self.partition_load_time[key] = datetime.now()
                                
                                synced_count += 1
                                collection_loaded_count += 1
                                
                        except Exception as partition_error:
                            logger.debug(f"Failed to check partition {collection_name}/{partition_name}: {partition_error}")
                            continue
                    
                    # 컬렉션별 요약 로그 (로드된 파티션이 있는 경우만)
                    if collection_loaded_count > 0:
                        collections_with_loaded_partitions += 1
                        logger.info(f"   [{i:3d}/{len(collection_names)}] {collection_name}: {collection_loaded_count}/{len(partition_names)} partitions loaded")
                
                except SchemaNotReadyException:
                    logger.debug(f"Collection '{collection_name}' does not exist - skipping")
                    continue
                except Exception as collection_error:
                    logger.debug(f"Failed to sync collection {collection_name}: {collection_error}")
                    continue
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Partition state synchronization completed in {elapsed_time:.2f}s")
            logger.info(f"   - Collections checked: {len(collection_names)}")
            logger.info(f"   - Collections with loaded partitions: {collections_with_loaded_partitions}")
            logger.info(f"   - Total partitions checked: {total_partitions_checked}")
            logger.info(f"   - Partitions synced: {synced_count}")
            logger.info(f"   - Estimated memory: ~{synced_count * 25}GB")
            
            return {
                "collections_checked": len(collection_names),
                "collections_with_loaded_partitions": collections_with_loaded_partitions,
                "total_partitions_checked": total_partitions_checked,
                "partitions_synced": synced_count,
                "sync_time_seconds": elapsed_time,
                "estimated_memory_gb": synced_count * 25
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to sync partition states: {e}")
            raise
    
    async def _cleanup_by_ttl(self):
        """TTL 기반 파티션 언로드"""
        now = datetime.now()
        ttl_threshold = timedelta(minutes=settings.PARTITION_TTL_MINUTES)
        
        to_unload = []
        
        for key, last_access in list(self.last_access_time.items()):
            if now - last_access > ttl_threshold:
                to_unload.append(key)
        
        if to_unload:
            logger.info(f"🗑️  TTL expired: unloading {len(to_unload)} partitions")
            
            for key in to_unload:
                try:
                    collection_name, partition_name = key.split("/", 1)
                    await self.unload_partition(collection_name, partition_name)
                except Exception as e:
                    logger.error(f"❌ Failed to unload {key}: {e}")
    
    async def _check_memory_and_cleanup(self):
        """메모리 임계값 체크 및 강제 정리"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > settings.MEMORY_THRESHOLD_PERCENT:
            logger.warning(f"⚠️ Memory threshold exceeded: {memory_percent:.1f}% > {settings.MEMORY_THRESHOLD_PERCENT}%")
            
            # 가장 오래된 파티션부터 언로드
            sorted_partitions = sorted(
                self.last_access_time.items(),
                key=lambda x: x[1]
            )
            
            unload_count = max(1, len(sorted_partitions) // 4)  # 최소 25% 언로드
            logger.info(f"🗑️  Force unloading {unload_count} oldest partitions")
            
            for key, _ in sorted_partitions[:unload_count]:
                try:
                    collection_name, partition_name = key.split("/", 1)
                    await self.unload_partition(collection_name, partition_name)
                    
                    # 메모리 재확인
                    memory = psutil.virtual_memory()
                    if memory.percent < settings.MEMORY_THRESHOLD_PERCENT:
                        logger.info(f"✅ Memory stabilized: {memory.percent:.1f}%")
                        break
                except Exception as e:
                    logger.error(f"❌ Failed to unload {key}: {e}")
    
    def get_partition_stats(self) -> dict:
        """
        파티션 통계 조회 (Health Check용)
        
        Returns:
            통계 정보 딕셔너리
        """
        memory = psutil.virtual_memory()
        
        # 가장 오래된 파티션 찾기
        oldest_partition = None
        oldest_time = None
        if self.last_access_time:
            oldest_key = min(self.last_access_time, key=self.last_access_time.get)
            oldest_time = self.last_access_time[oldest_key]
            oldest_partition = oldest_key
        
        # 로드된 파티션 목록
        all_loaded = []
        for collection_name, partitions in self.loaded_partitions.items():
            for partition_name in partitions:
                key = self._get_partition_key(collection_name, partition_name)
                last_access = self.last_access_time.get(key)
                all_loaded.append({
                    "key": key,
                    "last_access": last_access.isoformat() if last_access else None,
                    "minutes_ago": int((datetime.now() - last_access).total_seconds() / 60) if last_access else None
                })
        
        return {
            "loaded_count": sum(len(partitions) for partitions in self.loaded_partitions.values()),
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": round(memory.percent, 1),
                "threshold_percent": settings.MEMORY_THRESHOLD_PERCENT
            },
            "oldest_partition": {
                "key": oldest_partition,
                "last_access": oldest_time.isoformat() if oldest_time else None,
                "minutes_ago": int((datetime.now() - oldest_time).total_seconds() / 60) if oldest_time else None
            } if oldest_partition else None,
            "loaded_partitions": all_loaded,
            "config": {
                "ttl_minutes": settings.PARTITION_TTL_MINUTES,
                "cleanup_interval_seconds": settings.CLEANUP_INTERVAL_SECONDS,
                "max_concurrent_loads": settings.MAX_CONCURRENT_LOADS
            }
        }


# 전역 인스턴스
partition_manager = MilvusPartitionManager()

