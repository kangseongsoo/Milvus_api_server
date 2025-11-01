"""
자동 Flush 관리자 (이벤트 기반)
- 데이터 삽입/삭제 시에만 flush 실행
- 지연 flush로 API 응답 속도 향상
- 배치 처리로 리소스 효율성 극대화
"""

import asyncio
import logging
from typing import Set, Dict
from pymilvus import Collection
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AutoFlusher:
    """이벤트 기반 자동 flush 수행"""
    
    def __init__(self, delay_seconds: float = 0.5, max_wait_seconds: int = 5):
        """
        Args:
            delay_seconds: 데이터 변경 후 flush까지 대기 시간 (기본 0.5초)
                          → 이 시간 내 추가 변경을 배치 처리
            max_wait_seconds: 최대 대기 시간 (기본 5초)
                            → 이 시간이 지나면 무조건 flush
        """
        self.delay_seconds = delay_seconds
        self.max_wait_seconds = max_wait_seconds
        self.collections_to_flush: Set[str] = set()
        self.last_change_time: Dict[str, datetime] = {}  # 마지막 데이터 변경 시간
        self.last_flush_time: Dict[str, datetime] = {}   # 마지막 flush 시간
        self._running = False
        self._flush_lock = asyncio.Lock()
        
    async def mark_for_flush(self, collection_name: str):
        """
        데이터 변경 시 flush 마킹 (삽입/삭제 후 호출)
        
        Args:
            collection_name: flush할 컬렉션명
        """
        async with self._flush_lock:
            self.collections_to_flush.add(collection_name)
            self.last_change_time[collection_name] = datetime.now()
            logger.info(f"📌 Marked for flush: {collection_name}")
    
    async def start(self):
        """
        자동 flush 백그라운드 태스크 시작
        - 데이터 변경이 있을 때만 flush
        - delay_seconds 이내 추가 변경을 배치 처리
        """
        if self._running:
            logger.warning("⚠️ Auto-flusher is already running")
            return
            
        self._running = True
        logger.info(f"🔄 Auto-flusher started (delay: {self.delay_seconds}s, max_wait: {self.max_wait_seconds}s)")
        logger.info(f"   - Flush 전략: 이벤트 기반 (데이터 변경 시에만)")
        
        while self._running:
            try:
                # 데이터 변경이 있는지 확인
                if self.collections_to_flush:
                    await self._check_and_flush()
                
                # 짧은 주기로 체크 (리소스 부담 최소화)
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info("🛑 Auto-flusher cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Auto-flush error: {e}")
                await asyncio.sleep(1)
    
    async def _check_and_flush(self):
        """
        flush 조건 체크 및 실행
        - 조건 1: 마지막 변경 후 delay_seconds 경과
        - 조건 2: 마지막 flush 후 max_wait_seconds 경과
        """
        async with self._flush_lock:
            current_time = datetime.now()
            collections_to_process = []
            
            for coll_name in list(self.collections_to_flush):
                last_change = self.last_change_time.get(coll_name, current_time)
                last_flush = self.last_flush_time.get(coll_name, datetime.min)
                
                # 조건 1: 마지막 변경 후 delay_seconds 경과
                time_since_change = (current_time - last_change).total_seconds()
                
                # 조건 2: 마지막 flush 후 max_wait_seconds 경과
                time_since_flush = (current_time - last_flush).total_seconds()
                
                should_flush = (
                    time_since_change >= self.delay_seconds or
                    time_since_flush >= self.max_wait_seconds
                )
                
                if should_flush:
                    collections_to_process.append(coll_name)
            
            # flush 실행
            if collections_to_process:
                await self._flush_collections(collections_to_process)
    
    async def _flush_collections(self, collection_names: list[str]):
        """지정된 컬렉션들 flush"""
        for coll_name in collection_names:
            try:
                start_time = datetime.now()
                
                logger.info(f"🔥 Auto-Flushing: {coll_name}...")
                
                collection = Collection(name=coll_name)
                collection.flush()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Flush 완료: {coll_name} ({elapsed:.2f}초)")
                self.last_flush_time[coll_name] = datetime.now()
                
                # 마킹 제거
                self.collections_to_flush.discard(coll_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to flush {coll_name}: {e}")
    
    async def flush_immediately(self, collection_name: str):
        """
        즉시 flush 실행 (동기적)
        - 중요한 데이터이거나 즉시 검색이 필요한 경우 사용
        
        Args:
            collection_name: flush할 컬렉션명
        """
        try:
            logger.info(f"🔥 Immediate flush requested: {collection_name}")
            start_time = datetime.now()
            
            collection = Collection(name=collection_name)
            collection.flush()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.last_flush_time[collection_name] = datetime.now()
            
            # 마킹 제거
            async with self._flush_lock:
                self.collections_to_flush.discard(collection_name)
            
            logger.info(f"✅ Immediate flush completed in {elapsed:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Immediate flush failed for {collection_name}: {e}")
            raise
    
    async def stop(self):
        """자동 flush 중지"""
        self._running = False
        
        # 남아있는 flush 실행
        if self.collections_to_flush:
            logger.info(f"🔄 Flushing remaining {len(self.collections_to_flush)} collections...")
            await self._flush_collections(list(self.collections_to_flush))
        
        logger.info("🛑 Auto-flusher stopped")
    
    def get_last_flush_time(self, collection_name: str) -> datetime | None:
        """마지막 flush 시간 조회"""
        return self.last_flush_time.get(collection_name)
    
    def get_pending_flush_count(self) -> int:
        """flush 대기 중인 컬렉션 수"""
        return len(self.collections_to_flush)
    
    def get_status(self) -> dict:
        """Auto-flusher 상태 조회"""
        return {
            "running": self._running,
            "delay_seconds": self.delay_seconds,
            "max_wait_seconds": self.max_wait_seconds,
            "pending_flush_count": len(self.collections_to_flush),
            "pending_collections": list(self.collections_to_flush),
            "last_flush_times": {
                coll: time.isoformat() 
                for coll, time in self.last_flush_time.items()
            }
        }


# 전역 인스턴스
auto_flusher = AutoFlusher(delay_seconds=0.5, max_wait_seconds=5)


