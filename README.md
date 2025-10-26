# 🚀 Milvus FastAPI 기반 RAG 백엔드 시스템 (파티셔닝)

RAG(Retrieval-Augmented Generation) 시스템을 위한 **Milvus + PostgreSQL 파티셔닝 하이브리드 백엔드** API 서버입니다.

**3억 건 이상 대용량 데이터 처리 가능!** ⚡

## 📋 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [설치 및 실행](#설치-및-실행)
- [API 문서](#api-문서)
- [프로젝트 구조](#프로젝트-구조)

## 🎯 시스템 개요

이 프로젝트는 벡터 데이터베이스 **Milvus**를 FastAPI 기반 REST API 서버로 감싸서, 문서 데이터 저장, 유사도 검색, 컬렉션 관리 기능을 제공합니다.

### 핵심 특징

- ✅ **파티셔닝 아키텍처**: 봇별 자동 파티션 생성 (3억 행도 300만 행처럼 빠름!)
- ✅ **하이브리드 구조**: Milvus (벡터 검색) + PostgreSQL (메타데이터)
- ✅ **단일 DB 관리**: rag_db_chatty 하나로 모든 봇 관리
- ✅ **100배 성능**: Partition Pruning으로 필요한 파티션만 스캔
- ✅ **자동화**: 봇 등록 시 파티션 자동 생성 (트리거)
- ✅ **트랜잭션 관리**: PostgreSQL과 Milvus 간 데이터 일관성 보장
- ✅ **다양한 임베딩 모델**: OpenAI, BGE-M3, Sentence-BERT 등

## 🔹 주요 기능

| 기능 | 설명 |
|------|------|
| **데이터 삽입 API** | 메타데이터 + 텍스트 청크를 받아 서버에서 임베딩 처리 후 저장 |
| **유사도 검색 API** | 쿼리 텍스트를 임베딩하여 유사 문서 검색 |
| **문서 관리 API** | doc_id 기반으로 문서 단위 조회/수정/삭제 |
| **컬렉션 생성 API** | 새로운 컬렉션 동적 생성 (계정/봇별 분리) |

## 🏗️ 아키텍처

```
                          Milvus FastAPI 서버
                                 ↓
        ┌────────────────────────┴────────────────────────┐
        ↓                                                 ↓
    Milvus (컬렉션별 격리)              PostgreSQL (파티셔닝)
    ├── news_bot_collection            rag_db_chatty
    ├── law_bot_collection             ├── documents (파티셔닝)
    └── medical_bot_collection         │   ├── 뉴스봇 파티션 (300만)
                                       │   ├── 법률봇 파티션 (300만)
                                       │   └── 의료봇 파티션 (300만)
                                       └── bot_registry
```

**파티셔닝의 장점**:
- ⚡ **100배 빠른 검색**: Partition Pruning (3억 행 → 300만 행)
- 🚀 **자동 파티션 생성**: 봇 등록 시 트리거로 자동 생성
- 📦 **단일 DB 관리**: rag_db_chatty 하나로 모든 봇 관리
- 🔧 **관리 편의성**: 논리적으로 1개 테이블

자세한 내용:
- [RAG_Milvus_API_Design.md](./RAG_Milvus_API_Design.md) - 전체 설계
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - 파티셔닝 아키텍처
- [migrations/README_PARTITIONING.md](./migrations/README_PARTITIONING.md) - 파티셔닝 가이드

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
cd "Milvus API Server"

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 설정값 수정
```

### 2. 데이터베이스 설정

#### PostgreSQL 초기화 (파티셔닝 스키마)
```bash
# PostgreSQL 접속
psql -U postgres

# 공동 데이터베이스 생성
CREATE DATABASE rag_db_chatty;

# 파티셔닝 스키마 적용
\c rag_db_chatty
\i migrations/init_chatty.sql

# 봇 등록 (파티션 자동 생성)
INSERT INTO bot_registry (bot_id, bot_name, collection_name)
VALUES ('550e8400-e29b-41d4-a716-446655440000', '뉴스봇', 'news_bot_collection');
```

#### Milvus 실행 (Docker)
```bash
# Milvus Standalone 실행
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

### 3. 서버 실행

```bash
# 개발 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는
python -m uvicorn app.main:app --reload
```

서버가 실행되면:
- API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 헬스 체크: http://localhost:8000/health

## 📚 API 문서

### 봇 등록 및 컬렉션 관리
- `POST /collection/create` - 봇 등록 + Milvus 컬렉션 생성 + PostgreSQL 파티션 자동 생성

### 데이터 관리 (CRUD)
- `POST /data/insert` - 단일 문서 삽입
- `POST /data/insert/batch` - 여러 문서 일괄 삽입 (권장!) ⚡
- `GET /data/document/{doc_id}?chat_bot_id=xxx` - 문서 조회
- `PUT /data/document/{doc_id}` - 문서 전체 업데이트
- `PATCH /data/document/{doc_id}/metadata` - 메타데이터만 업데이트
- `DELETE /data/document/{doc_id}?chat_bot_id=xxx` - 문서 삭제
- `POST /data/document/delete/batch` - 여러 문서 일괄 삭제

### 검색
- `POST /search/query` - 유사도 검색 (파티션 프루닝 적용)

### API 예시

#### 단일 문서 삽입
```json
POST /data/insert
{
  "account_name": "chatty",
  "document": {
    "chat_bot_id": "550e8400-...",
    "title": "AI 뉴스"
  },
  "chunks": [...]
}
```

#### 배치 삽입 (권장!) ⚡
```json
POST /data/insert/batch
{
  "account_name": "chatty",
  "documents": [
    {"document": {...}, "chunks": [...]},
    {"document": {...}, "chunks": [...]},
    // ... 100개 문서
  ]
}

// 장점: 100번 API 호출 → 1번으로 10배 빠름!
```

자세한 API 명세는 `/docs` 엔드포인트에서 확인할 수 있습니다.

## 📁 프로젝트 구조

```
08. Milvus API Server/
├── app/                    # 메인 애플리케이션
│   ├── api/               # API 라우터 (엔드포인트)
│   ├── core/              # 핵심 비즈니스 로직
│   ├── models/            # Pydantic 모델
│   ├── schemas/           # DB 스키마
│   └── utils/             # 유틸리티
├── migrations/            # DB 마이그레이션
├── tests/                 # 테스트 코드
└── requirements.txt       # 의존성
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/test_api.py

# 커버리지 포함
pytest --cov=app tests/
```

## 🔧 설정

주요 환경변수 (`.env` 파일):

```bash
# Milvus 설정
MILVUS_HOST=localhost
MILVUS_PORT=19530

# PostgreSQL 설정 (파티셔닝)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB_NAME=rag_db_chatty  # 단일 공동 DB

# 임베딩 모델
EMBEDDING_MODEL=openai
OPENAI_API_KEY=sk-your-key
EMBEDDING_DIMENSION=1536
```

## 📊 파티셔닝 성능

| 데이터 규모 | 파티셔닝 없음 | 파티셔닝 사용 | 속도 |
|-----------|-------------|-------------|------|
| 봇 1개 (300만 행) | 15ms | 15ms | 동일 |
| 봇 10개 (3,000만 행) | 150ms | 15ms | **10배** ⚡ |
| 봇 100개 (3억 행) | 1,500ms | 15ms | **100배** ⚡ |

## 📖 참고 문서

- [시스템 설계 문서](./RAG_Milvus_API_Design.md)
- [Milvus 공식 문서](https://milvus.io/docs)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

## 📝 라이센스

이 프로젝트는 내부 사용을 위한 것입니다.

## 👤 작성자

강성수 - Milvus RAG API Server 개발

---

**문의사항이나 버그 리포트는 이슈로 등록해주세요.**

