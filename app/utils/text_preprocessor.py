"""
텍스트 전처리 유틸리티
검색 쿼리 정규화 및 정제
"""
import re
from typing import Optional


def normalize_query(text: str, max_length: int = 8000) -> str:
    """
    검색 쿼리 정규화
    
    Args:
        text: 원본 쿼리 텍스트
        max_length: 최대 길이 (OpenAI API 제한 고려)
    
    Returns:
        정규화된 텍스트
    
    처리 내용:
    1. 앞뒤 공백 제거
    2. 연속된 공백을 하나로
    3. 최대 길이 제한
    4. 빈 문자열 검증
    """
    if not text:
        return ""
    
    # 1. 앞뒤 공백 제거
    text = text.strip()
    
    # 2. 연속된 공백/탭/줄바꿈을 하나의 공백으로
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 최대 길이 제한 (OpenAI API 제한: 8191 토큰, 안전하게 8000자)
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_query(text: str, min_length: int = 1) -> tuple[bool, Optional[str]]:
    """
    검색 쿼리 검증
    
    Args:
        text: 쿼리 텍스트
        min_length: 최소 길이
    
    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Query text is empty"
    
    normalized = normalize_query(text)
    
    if len(normalized) < min_length:
        return False, f"Query text too short (minimum: {min_length} characters)"
    
    # 특수문자만 있는지 확인 (선택적)
    # if re.match(r'^[^a-zA-Z0-9가-힣]+$', normalized):
    #     return False, "Query contains only special characters"
    
    return True, None


def preprocess_for_embedding(text: str) -> str:
    """
    임베딩 생성 전 텍스트 전처리
    
    Args:
        text: 원본 텍스트
    
    Returns:
        전처리된 텍스트
    
    Note:
    - OpenAI 임베딩은 대부분의 텍스트를 잘 처리하므로 최소한의 전처리만 수행
    - 과도한 전처리는 오히려 의미를 손실시킬 수 있음
    """
    # 기본 정규화만 수행
    return normalize_query(text)


# ========== 선택적 고급 전처리 (필요시 사용) ==========

def remove_urls(text: str) -> str:
    """URL 제거 (검색 쿼리에서 URL이 의미 없는 경우)"""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)


def remove_emails(text: str) -> str:
    """이메일 제거 (검색 쿼리에서 이메일이 의미 없는 경우)"""
    email_pattern = r'\S+@\S+'
    return re.sub(email_pattern, '', text)


def normalize_punctuation(text: str) -> str:
    """
    구두점 정규화
    - 여러 개의 마침표를 하나로: "..." → "."
    - 여러 개의 물음표/느낌표를 하나로: "???" → "?"
    """
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    return text


def preprocess_aggressive(text: str) -> str:
    """
    공격적 전처리 (특수한 케이스에만 사용)
    
    Warning:
    - 의미 손실 가능성 있음
    - 일반적으로 권장하지 않음
    """
    text = normalize_query(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_punctuation(text)
    return text.strip()

