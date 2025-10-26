"""
임베딩 처리 서비스
텍스트를 벡터로 변환
"""
from typing import List
from openai import AsyncOpenAI
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """임베딩 서비스 (여러 모델 지원)"""
    
    def __init__(self):
        self.model_type = settings.EMBEDDING_MODEL
        
        if self.model_type == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = "text-embedding-3-small"
        
        logger.info(f"임베딩 서비스 초기화: {self.model_type}")
    
    async def embed(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩
        
        Args:
            text: 임베딩할 텍스트
        
        Returns:
            임베딩 벡터
        """
        if self.model_type == "openai":
            result = await self._embed_openai([text])
            return result[0]
        else:
            raise NotImplementedError(f"모델 '{self.model_type}'는 아직 구현되지 않았습니다.")
    
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        배치 임베딩 처리
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        if self.model_type == "openai":
            return await self._embed_openai(texts)
        else:
            raise NotImplementedError(f"모델 '{self.model_type}'는 아직 구현되지 않았습니다.")
    
    async def batch_embed_with_retry(
        self, 
        texts: List[str], 
        max_retries: int = 3, 
        backoff: float = 2.0
    ) -> List[List[float]]:
        """
        재시도 로직이 포함된 배치 임베딩 처리
        
        Args:
            texts: 임베딩할 텍스트 리스트
            max_retries: 최대 재시도 횟수
            backoff: 재시도 간격 (초)
        
        Returns:
            임베딩 벡터 리스트
        """
        import asyncio
        
        for attempt in range(max_retries + 1):
            try:
                return await self.batch_embed(texts)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ 임베딩 생성 최종 실패 (시도 {max_retries + 1}회): {str(e)}")
                    raise
                else:
                    wait_time = backoff * (2 ** attempt)  # 지수 백오프
                    logger.warning(f"⚠️ 임베딩 생성 실패 (시도 {attempt + 1}/{max_retries + 1}), {wait_time}초 후 재시도: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """
        OpenAI 임베딩 API 호출
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        try:
            # 배치 크기 제한
            batch_size = settings.MAX_BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            
            logger.info(f"임베딩 처리 완료: {len(texts)}개 텍스트")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"임베딩 처리 실패: {str(e)}")
            raise


# 전역 임베딩 서비스 인스턴스
embedding_service = EmbeddingService()

