"""
Milvus 파티션 로드 관리자
- FastAPI 시작 시 컬렉션 전체 로드
- 파티션 생성 및 추적 관리
"""

import asyncio
import logging
import psutil
from typing import Dict, Set
from pymilvus import Collection
from pymilvus.exceptions import SchemaNotReadyException
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class MilvusPartitionManager:
    """Milvus 컬렉션 및 파티션 로드 관리 (컬렉션 전체 로드 방식)"""
    
    def __init__(self):
        self.loaded_partitions: Dict[str, Set[str]] = {}  # {collection_name: {partition_names}}
        self.partition_load_time: Dict[str, datetime] = {}  # 로드 시간 추적
        self.last_access_time: Dict[str, datetime] = {}  # 마지막 접근 시간 (통계용)
        self._cleanup_running = False  # cleanup 루프 상태 (비활성화됨)
        
    async def preload_collection(self, collection_name: str):
        """
        FastAPI 시작 시 컬렉션 전체 로드 (모든 파티션 포함)
        
        Args:
            collection_name: 로드할 컬렉션명 (예: "collection_chatty")
        
        Note:
            컬렉션 전체를 로드하면 모든 파티션이 자동으로 로드됩니다.
            파티션별 로드보다 효율적이고 단순합니다.
        """
        try:
            logger.info(f"🔄 Starting preload for collection: {collection_name}")
            start_time = datetime.now()
            
            # 컬렉션 연결
            collection = Collection(name=collection_name)
            
            # 컬렉션 전체 로드 (모든 파티션 자동 포함)
            collection.load()
            
            # 모든 파티션 목록 가져오기 (추적용)
            partitions = collection.partitions
            partition_names = [p.name for p in partitions if p.name != "_default"]
            
            # 로드된 파티션 추적
            self.loaded_partitions[collection_name] = set(partition_names)
            
            # 파티션별 접근 시간 초기화
            for partition_name in partition_names:
                key = self._get_partition_key(collection_name, partition_name)
                self.last_access_time[key] = datetime.now()
                self.partition_load_time[key] = datetime.now()
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Collection preload completed in {elapsed_time:.2f}s")
            logger.info(f"   - Collection: {collection_name}")
            logger.info(f"   - Partitions: {len(partition_names)}")
            logger.info(f"   - Total entities: {collection.num_entities:,}")
            
        except SchemaNotReadyException as e:
            logger.warning(f"⚠️ Collection '{collection_name}' does not exist - skipping preload")
            return
        except Exception as e:
            logger.error(f"❌ Failed to preload collection {collection_name}: {e}")
            raise
    
    async def preload_all_collections(self):
        """
        FastAPI 시작 시 모든 컬렉션 전체 로드
        
        Returns:
            로드된 컬렉션 정보 딕셔너리
        """
        try:
            logger.info("🔄 Starting preload for all collections...")
            start_time = datetime.now()
            
            # Milvus에서 모든 컬렉션 조회
            from pymilvus import utility
            all_collections = utility.list_collections()
            
            # 시스템 컬렉션 제외
            collection_names = [name for name in all_collections if not name.startswith('_')]
            
            if not collection_names:
                logger.info("⏭️  No collections found - skipping preload")
                return {
                    "collections_loaded": 0,
                    "total_partitions": 0,
                    "preload_time_seconds": 0
                }
            
            logger.info(f"📦 Found {len(collection_names)} collections to load")
            
            # 모든 컬렉션 병렬 로드
            tasks = [self.preload_collection(name) for name in collection_names]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 집계
            total_partitions = sum(len(partitions) for partitions in self.loaded_partitions.values())
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ All collections preload completed in {elapsed_time:.2f}s")
            logger.info(f"   - Collections loaded: {len(collection_names)}")
            logger.info(f"   - Total partitions: {total_partitions}")
            
            return {
                "collections_loaded": len(collection_names),
                "total_partitions": total_partitions,
                "preload_time_seconds": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to preload all collections: {e}")
            raise
    
    def get_loaded_partitions(self, collection_name: str) -> Set[str]:
        """로드된 파티션 목록 조회"""
        return self.loaded_partitions.get(collection_name, set())
    
    def get_load_time(self, partition_name: str) -> datetime | None:
        """파티션 로드 시간 조회"""
        return self.partition_load_time.get(partition_name)
    
    def _get_partition_key(self, collection_name: str, partition_name: str) -> str:
        """파티션 고유 키 생성"""
        return f"{collection_name}/{partition_name}"
    
    async def ensure_partition_loaded(
        self, 
        collection_name: str, 
        partition_name: str,
        force_reload: bool = False
    ) -> bool:
        """
        파티션 생성 확인 및 접근 시간 업데이트 (컬렉션 전체 로드 방식)
        
        현재 동작:
        1. 컬렉션이 로드되어 있는지 확인
        2. 로드되어 있지 않으면 컬렉션 전체 로드
        3. 파티션이 메모리에 없으면 생성하고 메모리에 추가
        4. 항상 접근 시간 업데이트 (TTL 추적용)
        
        Args:
            collection_name: 컬렉션명
            partition_name: 파티션명
            force_reload: 사용하지 않음 (컬렉션 전체 로드이므로 불필요)
        
        Returns:
            True: 컬렉션이 로드되어 있고 파티션 사용 가능
            False: 컬렉션 로드 실패 또는 컬렉션 존재하지 않음
        
        Note:
            - 컬렉션이 시작 시 전체 로드되지만, 새로 생성된 컬렉션이나 재시작 후 로드 필요 시 자동 로드
            - 파티션이 없으면 생성만 하고 컬렉션은 이미 로드되어 있음
            - 접근 시간은 항상 업데이트하여 TTL 추적
        """
        key = self._get_partition_key(collection_name, partition_name)
        
        # 컬렉션이 로드되어 있지 않으면 전체 로드
        if collection_name not in self.loaded_partitions:
            try:
                logger.info(f"🔄 Collection '{collection_name}' not loaded - loading now...")
                collection = Collection(name=collection_name)
                
                # 컬렉션 전체 로드 (모든 파티션 포함)
                collection.load()
                
                # 모든 파티션 목록 가져오기 (추적용)
                partitions = collection.partitions
                partition_names = [p.name for p in partitions if p.name != "_default"]
                
                # 로드된 파티션 추적
                self.loaded_partitions[collection_name] = set(partition_names)
                
                # 파티션별 접근 시간 초기화
                for pname in partition_names:
                    pkey = self._get_partition_key(collection_name, pname)
                    self.last_access_time[pkey] = datetime.now()
                    self.partition_load_time[pkey] = datetime.now()
                
                logger.info(f"✅ Collection '{collection_name}' loaded: {len(partition_names)} partitions")
                
            except SchemaNotReadyException:
                logger.warning(f"⚠️ Collection '{collection_name}' does not exist")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to load collection '{collection_name}': {e}")
                return False
        
        # 파티션이 FastAPI 추적 딕셔너리에 없으면 처리
        # (컬렉션이 전체 로드되어 있으므로 Milvus에는 이미 로드되어 있음)
        # 새로 생성된 파티션이거나 추적 정보만 업데이트하면 됨
        if partition_name not in self.loaded_partitions[collection_name]:
            try:
                collection = Collection(name=collection_name)
                
                # 파티션이 Milvus에 실제로 존재하는지 확인 및 생성
                if not collection.has_partition(partition_name):
                    # 파티션이 Milvus에 없으면 생성
                    # (컬렉션이 로드되어 있으므로 생성 후 자동으로 사용 가능)
                    logger.info(f"📦 Creating new partition: {key}")
                    try:
                        collection.create_partition(partition_name=partition_name)
                        logger.info(f"✅ Partition created: {partition_name}")
                    except Exception as create_error:
                        # 이미 존재하는 경우 무시 (동시 생성 경쟁 조건)
                        if "already exists" in str(create_error).lower() or "exist" in str(create_error).lower():
                            logger.debug(f"   ⏭️  Partition already exists: {partition_name}")
                        else:
                            raise
                else:
                    # 파티션이 Milvus에 존재하지만 FastAPI 추적 딕셔너리에만 없음
                    # (컬렉션 전체 로드 시 시작 후 생성된 파티션)
                    logger.debug(f"📝 Partition exists in Milvus but not tracked - adding to tracking: {key}")
                
                # FastAPI 추적 딕셔너리에 추가 (접근 시간 추적용)
                self.loaded_partitions[collection_name].add(partition_name)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create/register partition {key}: {e}")
        
        # 항상 접근 시간 업데이트 (TTL 추적용)
        self.last_access_time[key] = datetime.now()
        return True
    
    async def auto_cleanup_loop(self):
        """
        백그라운드: 자동 정리 루프 (비활성화됨)
        
        Note:
            컬렉션 전체 로드 방식이므로 파티션 언로드를 하지 않습니다.
            메모리 관리는 시스템 레벨에서 처리합니다.
        """
        self._cleanup_running = True
        logger.info("ℹ️  Auto cleanup loop disabled (collections are fully loaded at startup)")
        
        try:
            while self._cleanup_running:
                # 파티션 언로드 없이 대기만 (필요시 통계 로깅 가능)
                await asyncio.sleep(settings.CLEANUP_INTERVAL_SECONDS)
                
                # 통계만 로깅 (선택적)
                # logger.debug(f"📊 Loaded partitions: {sum(len(p) for p in self.loaded_partitions.values())}")
        
        finally:
            self._cleanup_running = False
            logger.info("🛑 Auto cleanup loop stopped")
    
    async def stop_cleanup_loop(self):
        """Cleanup 루프 중지"""
        self._cleanup_running = False
    
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

