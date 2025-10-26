# 📘 Milvus FastAPI 기반 RAG 백엔드 시스템 설계서  
(Modified: Content 구조 개선 버전)

## 1️⃣ 시스템 개요

### 🎯 목표
RAG(Retrieval-Augmented Generation) 시스템에서 사용하는 벡터 데이터베이스 **Milvus**를  
**FastAPI 기반 REST API 서버**로 감싸서,  
문서 데이터 저장 / 유사도 검색 / 컬렉션 관리 기능을 제공합니다.

---

### 🔹 주요 기능

| 기능 | 설명 |
|------|------|
| **① 데이터 삽입 API** | 원문 텍스트, 메타데이터, 문서명을 받아 **서버에서 임베딩 처리** 후 Milvus에 저장 |
| **② 유사도 검색 API** | 쿼리 텍스트를 받아 **서버에서 임베딩 처리** 후 유사 문서 검색 |
| **③ 문서 관리 API (doc_id 기반)** | 문서 단위로 조회/수정/삭제 (모든 청크 일괄 처리) |
| **④ 컬렉션 생성 API** | 새로운 컬렉션(계정/봇별) 동적 생성 |

---

## 2️⃣ 시스템 아키텍처 (하이브리드 구조)

```
┌────────────────────────────────────┐
│   전처리 서버 (Document Processor)  │
│────────────────────────────────────│
│ • 문서 파싱 (PDF, DOCX, etc)        │
│ • 텍스트 추출 & 청킹               │
│ • 메타데이터 추출                   │
└────────┬───────────────────────────┘
         │
         │ 메타데이터 + 텍스트 청크 (모두 전송)
         │
         ▼
┌────────────────────────────────────────────┐
│        Milvus FastAPI 서버 (단일 진입점)     │
│────────────────────────────────────────────│
│ API 엔드포인트:                             │
│ • POST /data/insert                        │
│ • POST /search/query                       │
│ • GET/PUT/DELETE /data/document/{doc_id}   │
│                                            │
│ ┌────────────────────────────────────┐    │
│ │  임베딩 처리 레이어                 │    │
│ │  - Dense Embedding (OpenAI/BGE)    │    │
│ │  - Sparse Embedding (BM25/SPLADE)  │    │
│ │  - 캐싱 & 배치 처리                 │    │
│ └────────────────────────────────────┘    │
│              ↓                             │
│ ┌────────────────────────────────────┐    │
│ │  통합 저장 레이어 ⭐                │    │
│ │  (트랜잭션 관리)                    │    │
│ └────────────────────────────────────┘    │
│         ↙              ↘                   │
└────────┬───────────────┬───────────────────┘
         │               │
         │ 메타데이터     │ 벡터 + doc_id
         ▼               ▼
┌─────────────────┐  ┌──────────────────┐
│  PostgreSQL DB  │  │    Milvus DB     │
│─────────────────│  │──────────────────│
│ documents       │  │ (Vector Only)    │
│ ├─ doc_id (PK)  │  │                  │
│ ├─ chat_bot_id  │  │ [doc_id: 1234]   │
│ ├─ title        │  │  ├─ vector (201) │
│ ├─ content      │  │  ├─ vector (202) │
│ ├─ file_path    │  │  ├─ vector (203) │
│ ├─ author       │  │  └─ vector (204) │
│ ├─ tags[]       │  │                  │
│ ├─ chunk_count  │  │ ⭐ 청크 개수      │
│ ├─ created_at   │  │                  │
│ └─ metadata     │  └──────────────────┘
│                 │
│ document_chunks │
│ ├─ chunk_id     │
│ ├─ doc_id (FK)  │
│ ├─ chunk_index  │
│ ├─ chunk_text   │
│ └─ ...          │
└─────────────────┘
        ▲
        │
        │ 검색 시 메타데이터 조회 ⚡
        │
     (응답 통합)
```

---

## 3️⃣ 데이터베이스 스키마 설계

### 🔹 Milvus 컬렉션 스키마 (벡터 + 필터링용 메타데이터)

**전략: Milvus는 벡터 검색 + 필터링, 상세 메타데이터는 PostgreSQL**

| 필드명 | 타입 | 설명 | 비고 |
|--------|------|------|------|
| `id` | INT64 | 청크 고유 ID | Primary Key (Milvus 자동 생성) |
| `doc_id` | INT64 | **문서 ID** | ⭐ PostgreSQL과 연결하는 외래 키 |
| `chat_bot_id` | VARCHAR(100) | 챗봇 ID | 파티션 키 |
| `chunk_index` | INT64 | 청크 순서 | 0부터 시작 (정렬용) |
| `embedding_dense` | FLOAT_VECTOR(1536) | Dense 임베딩 벡터 | ANN 검색 대상 |
| `metadata` | JSON | **메타데이터** | ⭐ expr 필터링용 (file_type, tags 등) |
| `embedding_sparse` | SPARSE_FLOAT_VECTOR | Sparse 임베딩 벡터 | Hybrid 검색용 (옵션) |

```python
# Milvus 스키마 (벡터 + 필터링용 메타데이터)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="doc_id", dtype=DataType.INT64),  # PostgreSQL FK
    FieldSchema(name="chat_bot_id", dtype=DataType.VARCHAR, max_length=100),  # 파티션 키
    FieldSchema(name="chunk_index", dtype=DataType.INT64),  # 청크 순서
    FieldSchema(name="embedding_dense", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="metadata", dtype=DataType.JSON),  # ⭐ expr 필터링용
    FieldSchema(name="embedding_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)  # 옵션
]

# 스칼라 인덱스 생성 (빠른 필터링)
collection.create_index(
    field_name="chat_bot_id",
    index_name="idx_chat_bot_id"
)

collection.create_index(
    field_name="doc_id",
    index_name="idx_doc_id"
)

# 벡터 인덱스 생성
collection.create_index(
    field_name="embedding_dense",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }
)
```

**metadata 필드 활용 (expr 필터링)**
```python
# JSON 메타데이터를 통한 필터링 검색
results = collection.search(
    data=[query_vector],
    anns_field="embedding_dense",
    partition_names=["news_bot"],
    expr='metadata["file_type"] == "pdf" and metadata["tags"][0] == "ai"',  # ⭐ JSON 필터링!
    limit=5
)

# 예시:
# - 특정 파일 타입만: metadata["file_type"] == "pdf"
# - 특정 태그 포함: "ai" in metadata["tags"]
# - 날짜 범위: metadata["created_date"] > "2024-01-01"
# - 복합 조건: metadata["file_type"] == "pdf" and metadata["author"] == "강성수"
```

---

### 🔹 PostgreSQL 스키마 (메타데이터 전용)

```sql
-- 문서 테이블
CREATE TABLE documents (
    doc_id BIGSERIAL PRIMARY KEY,
    chat_bot_id VARCHAR(100) NOT NULL,  -- 챗봇 ID (파티션 키)
    title VARCHAR(512) NOT NULL,
    content TEXT,  -- 원문 텍스트 전체
    file_path VARCHAR(1024),
    file_type VARCHAR(50),  -- pdf, docx, url, text
    author VARCHAR(255),
    tags TEXT[],  -- PostgreSQL 배열 타입
    chunk_count INT DEFAULT 0,  -- ⭐ 총 청크 개수 (COUNT 쿼리 없이 빠른 조회)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB  -- 추가 메타데이터
);

-- 청크 테이블 (선택적 - 청크별 상세 정보 필요 시)
CREATE TABLE document_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    doc_id BIGINT REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    start_pos INT,  -- 원문에서의 시작 위치
    end_pos INT,    -- 원문에서의 끝 위치
    page_number INT,  -- PDF 페이지 번호
    UNIQUE(doc_id, chunk_index)
);

-- 인덱스 생성 (빠른 조회)
CREATE INDEX idx_documents_tags ON documents USING GIN(tags);
CREATE INDEX idx_documents_file_type ON documents(file_type);
CREATE INDEX idx_chunks_doc_id ON document_chunks(doc_id);
```

---

### 🎯 하이브리드 구조의 장점

| 구분 | Milvus | PostgreSQL |
|------|--------|-----------|
| **저장 데이터** | 벡터 + doc_id만 | 원문, 메타데이터, 파일 정보 등 모든 것 |
| **크기** | 가벼움 (벡터만) | 상대적으로 큼 |
| **강점** | 벡터 유사도 검색 ⚡ | 메타데이터 조회 ⚡<br>복잡한 필터링<br>JOIN 연산<br>전문 검색 |
| **조회 속도** | 벡터 검색: 빠름<br>메타데이터: 느림 | 정확한 키 조회: 매우 빠름 (0.1ms)<br>복잡한 쿼리: 빠름 |
| **확장성** | 벡터 전용으로 최적화 | 범용 데이터 관리 |

> ⚡ **성능 비교**: PostgreSQL B-tree 인덱스 조회 (0.1ms) vs Milvus 스칼라 필터링 (5-10ms)


---

### 📌 데이터 흐름

#### 삽입 시:
```
전처리 서버
  ↓
Milvus FastAPI 서버로 메타데이터 + 텍스트 청크 전송
  ↓
Milvus FastAPI 서버 (트랜잭션 처리):
  1. PostgreSQL에 문서 메타데이터 저장 → doc_id 획득
  2. PostgreSQL에 청크 텍스트 저장 (옵션)
  3. 텍스트 임베딩 처리
  4. Milvus에 벡터 + doc_id 저장
  ↓
성공 응답 (실패 시 롤백 ⚡)
```

#### 검색 시:
```
사용자 쿼리
  ↓
Milvus FastAPI 서버:
  1. 쿼리 임베딩 처리
  2. Milvus 벡터 검색 → doc_id 리스트 획득
  3. PostgreSQL에서 doc_id로 메타데이터 일괄 조회 ⚡
  4. 결과 통합
  ↓
클라이언트에 통합 결과 반환
```

---

## 4️⃣ 인덱스 구조 (RAG 실시간용)

```python
index_params = {
  "index_type": "HNSW",
  "metric_type": "COSINE",
  "params": {"M": 8, "efConstruction": 64}
}
```

- **검색 시 매개변수 예시**
```python
search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
```

> ✅ 실시간 삽입/삭제/검색이 모두 가능한 인덱스는 `HNSW` 입니다.

---

## 4️⃣-1. 임베딩 모델 설정

### 🔹 Dense Embedding
| 모델 | 차원 | 용도 | API |
|------|------|------|-----|
| **OpenAI text-embedding-3-small** | 1536 | 일반 텍스트 임베딩 | OpenAI API |
| **OpenAI text-embedding-3-large** | 3072 | 고성능 임베딩 | OpenAI API |
| **BGE-M3** | 1024 | 다국어 지원 | Local/HuggingFace |
| **Sentence-BERT (Korean)** | 768 | 한국어 특화 | Local/HuggingFace |

### 🔹 Sparse Embedding (하이브리드 검색용)
| 모델 | 용도 |
|------|------|
| **BM25** | 키워드 기반 검색 |
| **SPLADE** | 학습 기반 희소 벡터 |

### 🔹 서버 설정 예시
```python
# config.py
EMBEDDING_MODEL = "openai"  # or "bge-m3", "sentence-bert"
OPENAI_API_KEY = "sk-..."
EMBEDDING_DIMENSION = 1536
USE_SPARSE_EMBEDDING = True  # 하이브리드 검색 활성화
```

---

## 5️⃣ API 설계

### ✅ 1. 컬렉션 생성 API
**`POST /collection/create`**

**요청 예시**
```json
{
  "collection_name": "rag_docs_bot01",
  "dimension": 1536
}
```

**응답 예시**
```json
{
  "status": "success",
  "message": "Collection 'rag_docs_bot01' created successfully."
}
```

---

### ✅ 2. 데이터 삽입 API
**`POST /data/insert`**

**📌 전처리 서버에서 전송**
- 문서 메타데이터 (title, author, tags, file_path 등)
- 텍스트 청크 배열
- 한 번의 API 호출로 모든 데이터 전송 ⭐

**📌 Milvus API 서버 처리 흐름 (트랜잭션)**
```python
async def insert_document(request):
    async with transaction():  # 트랜잭션 시작
        try:
            # 1. PostgreSQL에 문서 메타데이터 저장
            doc_id = await postgres.insert_document(metadata)
            
            # 2. PostgreSQL에 청크 텍스트 저장 (옵션)
            await postgres.insert_chunks(doc_id, chunks)
            
            # 3. 텍스트 임베딩 처리
            embeddings = await embedding_service.batch_embed(texts)
            
            # 4. Milvus에 벡터 저장
            chunk_ids = await milvus.insert(doc_id, embeddings)
            
            await transaction.commit()  # 커밋
            return success_response
            
        except Exception as e:
            await transaction.rollback()  # 실패 시 롤백 ⚡
            raise
```

**요청 예시**
```json
{
  "account_name": "chatty",
  "document": {
    "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "인공지능 입문서",
    "file_path": "/uploads/docs/ai_intro.pdf",
    "file_type": "pdf",
    "author": "강성수",
    "tags": ["ai", "machine-learning", "beginner"],
    "metadata": {
      "page_count": 120,
      "language": "ko",
      "department": "AI연구소"
    }
  },
  "chunks": [
    {"chunk_index": 0, "text": "인공지능은 데이터를 기반으로...", "page_number": 1},
    {"chunk_index": 1, "text": "머신러닝은 인공지능의...", "page_number": 1},
    {"chunk_index": 2, "text": "딥러닝은 머신러닝의...", "page_number": 2}
    // ... 총 120개 청크
  ]
}
```

**응답 예시**
```json
{
  "status": "success",
  "doc_id": 1234,
  "total_chunks": 120,
  "postgres_insert_time_ms": 5.2,
  "embedding_time_ms": 2450.8,
  "milvus_insert_time_ms": 180.5,
  "total_time_ms": 2636.5
}
```

> ✅ **트랜잭션 보장**: PostgreSQL과 Milvus 삽입 중 하나라도 실패하면 전체 롤백!

> 💡 **단일 진입점**: 전처리 서버는 Milvus API 한 곳만 호출하면 되므로 간단합니다.

> ⚡ **chunk_count 저장**: PostgreSQL에 총 청크 수를 저장하여 COUNT 쿼리 없이 빠른 조회

---

### ✅ 2-1. 배치 데이터 삽입 API (대량 처리)
**`POST /data/insert/batch`**

**📌 전처리 서버에서 여러 문서를 한 번에 전송**
- 100개 문서를 1번 API 호출로 처리 (100번 호출 대비 **10배 빠름!**)
- 임베딩 배치 처리로 비용 절감
- 네트워크 오버헤드 최소화

**요청 예시**
```json
{
  "account_name": "chatty",
  "documents": [
    {
      "document": {
        "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
        "title": "AI 뉴스 1",
        "file_path": "/uploads/news1.pdf",
        "file_type": "pdf",
        "author": "강성수",
        "tags": ["ai", "news"]
      },
      "chunks": [
        {"chunk_index": 0, "text": "AI는 빠르게...", "page_number": 1},
        {"chunk_index": 1, "text": "GPT-4는...", "page_number": 1}
        // ... 120개 청크
      ]
    },
    {
      "document": {
        "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
        "title": "AI 뉴스 2",
        "file_path": "/uploads/news2.pdf",
        "file_type": "pdf",
        "author": "강성수",
        "tags": ["ai", "news"]
      },
      "chunks": [
        {"chunk_index": 0, "text": "인공지능은...", "page_number": 1},
        // ... 95개 청크
      ]
    }
    // ... 총 100개 문서
  ]
}
```

**응답 예시**
```json
{
  "status": "success",
  "total_documents": 100,
  "total_chunks": 12000,
  "success_count": 98,
  "failure_count": 2,
  "results": [
    {
      "doc_id": 1234,
      "title": "AI 뉴스 1",
      "total_chunks": 120,
      "success": true
    },
    {
      "doc_id": 1235,
      "title": "AI 뉴스 2",
      "total_chunks": 95,
      "success": true
    },
    {
      "doc_id": -1,
      "title": "AI 뉴스 50",
      "total_chunks": 150,
      "success": false,
      "error": "Embedding timeout"
    }
    // ... 100개 결과
  ],
  "postgres_insert_time_ms": 520.5,
  "embedding_time_ms": 24500.0,
  "milvus_insert_time_ms": 1800.5,
  "total_time_ms": 26821.0
}
```

**💡 배치 처리 이점**
- ⚡ **네트워크**: 100번 → 1번 호출 (10배 빠름)
- 💰 **비용**: OpenAI 임베딩 배치 처리 (비용 절감)
- 🔒 **안정성**: 부분 실패 시에도 성공한 문서는 처리됨
- 📊 **모니터링**: 전체 성공/실패 현황 한눈에 파악

---

### ✅ 3. 유사도 검색 API (하이브리드 조회)
**`POST /search/query`**

**📌 처리 흐름**
1. 클라이언트가 **쿼리 텍스트** 전송
2. 서버에서 임베딩 모델로 쿼리 벡터화
   - Dense Embedding 생성
   - Sparse Embedding 생성 (하이브리드 검색 시)
3. **Milvus 벡터 검색** → `doc_id` + `chunk_index` + `score` 획득
4. **PostgreSQL 조회** → `doc_id` 기반으로 메타데이터 일괄 조회 ⚡
5. Milvus 결과 + PostgreSQL 메타데이터 통합 반환

**요청 예시**
```json
{
  "collection_name": "rag_docs_bot01",
  "query_text": "인공지능 학습 방법에 대해 알려줘",
  "limit": 5
}
```

**응답 예시**
```json
{
  "status": "success",
  "query": "@@@@",
  "vector_search_time_ms": 15,
  "postgres_query_time_ms": 2,
  "total_time_ms": 17,
  "results": [
    {
      "chunk_id": 201,
      "doc_id": 1234,
      "chunk_index": 2,
      "score": 0.985,
      "chunk_text": "머신러닝은 인공지능의 핵심 기술로...",
      "document": {
        "title": "AI 입문서",
        "file_type": "pdf",
        "file_path": "/uploads/ai_intro.pdf",
        "author": "강성수",
        "tags": ["ai", "machine-learning", "beginner"],
        "created_at": "2024-01-15T10:30:00Z"
      }
    },
    {
      "chunk_id": 305,
      "doc_id": 1235,
      "chunk_index": 0,
      "score": 0.912,
      "chunk_text": "딥러닝은 인공신경망을 활용한...",
      "document": {
        "title": "딥러닝 가이드",
        "file_type": "pdf",
        "author": "강성수",
        "tags": ["deep-learning", "neural-network"]
      }
    }
  ]
}
```

**💡 성능 최적화**
```python
# Milvus 검색 (15ms)
milvus_results = milvus.search(query_vector, limit=5)
# → [(doc_id: 1234, chunk_index: 2, score: 0.985), ...]

# PostgreSQL 일괄 조회 (2ms) ⚡
doc_ids = [r.doc_id for r in milvus_results]
documents = postgres.execute(
    "SELECT * FROM documents WHERE doc_id = ANY($1)",
    [doc_ids]
)

# 청크 텍스트 조회 (옵션)
chunks = postgres.execute(
    "SELECT chunk_text FROM document_chunks WHERE (doc_id, chunk_index) IN (...)"
)

# 결과 통합
return merge_results(milvus_results, documents, chunks)
```

---

### ✅ 4. 문서 조회 API (Read by doc_id)
**`GET /data/document/{doc_id}`**

**설명:** doc_id에 해당하는 문서의 모든 청크 조회

**요청 예시**
```
GET /data/document/1234?collection_name=rag_docs_bot01
```

**응답 예시**
```json
{
  "status": "success",
  "doc_id": 1234,
  "chunk_count": 5,
  "chunks": [
    {
      "id": 201,
      "text": "인공지능은 데이터를 기반으로 학습하는 시스템이다.",
      "content_name": "ai_intro.pdf",
      "content_type": "pdf",
      "metadata": {
        "chunk_index": 1,
        "chunk_total": 5
      }
    },
    {
      "id": 202,
      "text": "머신러닝은 인공지능의 한 분야이다.",
      "metadata": {
        "chunk_index": 2,
        "chunk_total": 5
      }
    }
  ]
}
```

---

### ✅ 5. 문서 수정 API (Update by doc_id)
**`PUT /data/document/{doc_id}`**

**설명:** 
- doc_id에 해당하는 문서 전체를 새로운 데이터로 교체
- 문서가 재처리(re-chunking)된 경우 사용

**📌 Milvus API 서버 처리 흐름 (트랜잭션)**
```python
async def update_document(doc_id, request):
    async with transaction():
        try:
            # 1. PostgreSQL: 문서 메타데이터 업데이트
            await postgres.update_document(doc_id, metadata)
            
            # 2. PostgreSQL: 기존 청크 삭제 후 새 청크 삽입
            await postgres.delete_chunks(doc_id)
            await postgres.insert_chunks(doc_id, new_chunks)
            
            # 3. Milvus: 기존 벡터 삭제
            await milvus.delete(filter=f"doc_id == {doc_id}")
            
            # 4. 새 텍스트 임베딩 처리
            embeddings = await embedding_service.batch_embed(texts)
            
            # 5. Milvus: 새 벡터 삽입
            await milvus.insert(doc_id, embeddings)
            
            await transaction.commit()
            return success_response
            
        except Exception as e:
            await transaction.rollback()  # 실패 시 롤백 ⚡
            raise
```

**요청 예시**
```json
{
  "account_name": "chatty",
  "document": {
    "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "인공지능 입문서 (개정판)",
    "file_path": "/uploads/docs/ai_intro_v2.pdf",
    "file_type": "pdf",
    "author": "강성수",
    "tags": ["ai", "machine-learning", "updated"]
  },
  "chunks": [
    {"chunk_index": 0, "text": "인공지능은 데이터를...", "page_number": 1},
    {"chunk_index": 1, "text": "머신러닝은 인공지능의...", "page_number": 1},
    {"chunk_index": 2, "text": "딥러닝은 머신러닝의...", "page_number": 2}
    // ... 총 85개 청크
  ]
}
```

**응답 예시**
```json
{
  "status": "success",
  "message": "Document updated successfully",
  "doc_id": 1234,
  "deleted_chunks": 120,
  "inserted_chunks": 85,
  "postgres_time_ms": 12.5,
  "embedding_time_ms": 1980.3,
  "milvus_time_ms": 155.8,
  "total_time_ms": 2148.6
}
```

---

**`PATCH /data/document/{doc_id}/metadata`** - 메타데이터만 수정

**설명:** 
- doc_id의 메타데이터만 업데이트 (초고속 ⚡)
- PostgreSQL만 수정, Milvus는 건드리지 않음
- 임베딩 재생성 없음

**📌 처리 흐름**
```python
async def patch_metadata(doc_id, metadata_updates):
    # PostgreSQL UPDATE만 실행 (0.1ms) ⚡
    await postgres.update_document_metadata(doc_id, metadata_updates)
    return success_response
```

**요청 예시**
```json
{
  "account_name": "chatty",
  "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "metadata_updates": {
    "tags": ["ai", "updated", "reviewed"],
    "author": "강성수",
    "reviewed_at": 1704153600,
    "status": "published"
  }
}
```

**응답 예시**
```json
{
  "status": "success",
  "message": "Metadata updated successfully",
  "doc_id": 1234,
  "updated_at": "2024-01-15T10:30:00Z",
  "postgres_time_ms": 0.8
}
```

---

### ✅ 6. 문서 삭제 API (Delete by doc_id)
**`DELETE /data/document/{doc_id}`**

**설명:** doc_id에 해당하는 문서의 모든 청크를 삭제

**📌 Milvus API 서버 처리 흐름 (트랜잭션)**
```python
async def delete_document(doc_id):
    async with transaction():
        try:
            # 1. Milvus: 벡터 삭제
            milvus_result = await milvus.delete(filter=f"doc_id == {doc_id}")
            
            # 2. PostgreSQL: 문서 삭제 (CASCADE로 청크도 자동 삭제)
            postgres_result = await postgres.delete_document(doc_id)
            
            await transaction.commit()
            return success_response
            
        except Exception as e:
            await transaction.rollback()  # 실패 시 롤백 ⚡
            raise
```

**요청 예시**
```
DELETE /data/document/1234?account_name=chatty&chat_bot_id=550e8400-e29b-41d4-a716-446655440000
```

**응답 예시**
```json
{
  "status": "success",
  "message": "Document and all chunks deleted successfully",
  "doc_id": 1234,
  "deleted_vectors": 5,
  "postgres_time_ms": 1.2,
  "milvus_time_ms": 3.5,
  "total_time_ms": 4.7
}
```

---

**`POST /data/document/delete/batch`** - 다건 문서 삭제

**설명:** 여러 doc_id의 문서들을 일괄 삭제

**📌 처리 흐름 (트랜잭션)**
```python
async def delete_documents_batch(doc_ids):
    async with transaction():
        try:
            # 1. Milvus: 벡터 일괄 삭제
            await milvus.delete(filter=f"doc_id in {doc_ids}")
            
            # 2. PostgreSQL: 문서 일괄 삭제
            await postgres.delete_documents(doc_ids)
            
            await transaction.commit()
            return success_response
            
        except Exception as e:
            await transaction.rollback()
            raise
```

**요청 예시**
```json
{
  "collection_name": "rag_docs_bot01",
  "doc_ids": [1234, 1235, 1236, 1237]
}
```

**응답 예시**
```json
{
  "status": "success",
  "message": "Multiple documents deleted successfully",
  "deleted_documents": 4,
  "deleted_vectors": 23,
  "postgres_time_ms": 3.8,
  "milvus_time_ms": 8.2,
  "total_time_ms": 12.0
}
```

---

## ✅ 요약

| 구분 | 내용 |
|------|------|
| **아키텍처** | 하이브리드 구조 (Milvus + PostgreSQL) + **단일 진입점** ⭐ |
| **단일 진입점** | 전처리 서버 → **Milvus FastAPI 서버** (메타데이터 + 텍스트 모두 전송)<br>→ Milvus API 서버에서 PostgreSQL + Milvus 통합 관리 |
| **Milvus 역할** | 벡터 검색 전용 (id, doc_id, chunk_index, embeddings만 저장) |
| **PostgreSQL 역할** | 메타데이터 관리 (원문, 파일 정보, 태그, 작성자 등 모든 정보) ⚡ |
| **트랜잭션 관리** | Milvus API 서버에서 PostgreSQL + Milvus 통합 트랜잭션 처리<br>실패 시 자동 롤백 ⚡ |
| **핵심 장점** | 1. 단일 API 호출로 모든 데이터 저장<br>2. Milvus: 벡터 검색 특화<br>3. PostgreSQL: 메타데이터 조회 특화 (0.1ms)<br>4. 트랜잭션으로 데이터 일관성 보장 |
| **검색 흐름** | 1. Milvus 벡터 검색 → doc_id 획득<br>2. PostgreSQL에서 메타데이터 조회 ⚡<br>3. 통합 결과 반환 |
| **인덱스** | Milvus: HNSW (벡터)<br>PostgreSQL: B-tree (doc_id, 태그 등) |
| **임베딩 처리** | Milvus API 서버에서 텍스트 → 벡터 변환 (OpenAI, BGE-M3 등) |
| **API 구성** | **컬렉션**: `/collection/create`<br>**데이터**: `/data/insert` (C), `/data/document/{doc_id}` (R/U/D)<br>**검색**: `/search/query` (하이브리드 조회) |
| **문서 관리** | doc_id 기반으로 문서 단위 조회/수정/삭제 (모든 청크 일괄 처리) |
| **확장성** | 계정/봇별 컬렉션 분리로 무한 확장 가능 |

---

## 📋 전체 API 목록

### 🗂️ 컬렉션 관리
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/collection/create` | 컬렉션 생성 |

### 📝 데이터 관리 (CRUD - doc_id 기반)
| Method | Endpoint | 설명 |
|--------|----------|------|
| **POST** | `/data/insert` | 데이터 삽입 (Create) |
| **GET** | `/data/document/{doc_id}` | 문서의 모든 청크 조회 (Read) |
| **PUT** | `/data/document/{doc_id}` | 문서 전체 교체 (Update) |
| **PATCH** | `/data/document/{doc_id}/metadata` | 문서의 메타데이터만 수정 (Update) |
| **DELETE** | `/data/document/{doc_id}` | 문서의 모든 청크 삭제 (Delete) |
| **POST** | `/data/document/delete/batch` | 여러 문서 일괄 삭제 (Delete) |

### 🔍 검색
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/search/query` | 유사도 검색 (벡터 검색) |

---

## 💡 하이브리드 아키텍처의 핵심 장점

### ⚡ 성능 비교

| 작업 | Milvus 단독 | 하이브리드 (Milvus + PostgreSQL) |
|------|-------------|--------------------------------|
| **벡터 검색** | 15ms ✅ | 15ms ✅ (동일) |
| **메타데이터 조회** | 5-10ms 😐 | 0.1ms ⚡ (50~100배 빠름) |
| **복잡한 필터링** | 느림 😐 | 빠름 ⚡ (SQL의 강력함) |
| **JOIN 연산** | 불가능 ❌ | 가능 ✅ |
| **전문 검색** | 제한적 😐 | 강력 ✅ (PostgreSQL FTS) |
| **데이터 크기** | 큼 (메타 포함) | 작음 ⚡ (벡터만) |

### 🎯 doc_id 기반 문서 단위 관리

```
PostgreSQL (doc_id: 1234)
├─ 문서 메타데이터 (title, author, tags, etc)
└─ 청크 텍스트 (옵션)

Milvus (doc_id: 1234)
├─ 청크 0 벡터 (chunk_id: 201)
├─ 청크 1 벡터 (chunk_id: 202)
├─ 청크 2 벡터 (chunk_id: 203)
└─ 청크 3 벡터 (chunk_id: 204)

삭제 요청: DELETE /data/document/1234
  ↓
PostgreSQL: 1개 행 삭제 (CASCADE로 청크도 삭제)
Milvus: 4개 벡터 삭제
✅ 완료
```

### 📌 실전 사용 시나리오

#### 1. **고급 필터링 (PostgreSQL의 강력함 활용)**
```sql
-- 예: 특정 작성자의 최근 1주일 PDF 문서만 검색
SELECT doc_id FROM documents
WHERE author = '강성수'
  AND file_type = 'pdf'
  AND created_at > NOW() - INTERVAL '7 days'
  AND 'machine-learning' = ANY(tags);

-- 이 doc_id 리스트로 Milvus 검색 필터링
```

#### 2. **문서 업데이트 (트랜잭션으로 안전하게)**
```
사용자가 PDF 재업로드
  ↓
전처리 서버 → Milvus API 서버 (메타데이터 + 새 청크 전송)
  ↓
Milvus API 서버 (트랜잭션):
  1. PostgreSQL: UPDATE documents WHERE doc_id = 1234
  2. PostgreSQL: 기존 청크 삭제 → 새 청크 삽입
  3. Milvus: 기존 벡터 삭제
  4. 텍스트 임베딩 처리
  5. Milvus: 새 벡터 삽입
  6. 트랜잭션 커밋 (실패 시 롤백 ⚡)
  ↓
완료
```

#### 3. **통계 및 분석 (PostgreSQL의 강력한 쿼리)**
```sql
-- 문서 통계
SELECT file_type, COUNT(*), AVG(chunk_count)
FROM documents
GROUP BY file_type;

-- 태그별 문서 수
SELECT unnest(tags) as tag, COUNT(*)
FROM documents
GROUP BY tag
ORDER BY count DESC;

-- 가장 많이 검색된 문서 (검색 로그 테이블 JOIN)
SELECT d.title, COUNT(s.search_id) as search_count
FROM documents d
JOIN search_logs s ON s.doc_id = d.doc_id
GROUP BY d.title
ORDER BY search_count DESC
LIMIT 10;
```

#### 4. **메타데이터만 수정 (초고속 ⚡)**
```
문서에 새 태그 추가
  ↓
PostgreSQL UPDATE만 실행 (0.1ms)
  ↓
Milvus는 건드리지 않음 (벡터는 그대로)
✅ 즉시 완료
```

### 💾 저장 공간 최적화

```
100만 개 문서 (평균 10개 청크) = 1000만 벡터

Milvus 단독:
- 벡터: 1000만 × 1536 × 4 bytes = 61 GB
- 메타데이터: 1000만 × ~500 bytes = 5 GB
총: 66 GB

하이브리드:
- Milvus: 벡터만 = 61 GB
- PostgreSQL: 메타데이터 = 5 GB
총: 66 GB (동일)

하지만 성능은 압도적! ⚡
- Milvus: 벡터 검색 특화
- PostgreSQL: 메타데이터 조회/필터링 특화
```

---

### 🔧 전처리 서버 구현 예시

```python
# 전처리 서버 (Document Processor)
import httpx
from document_parser import parse_pdf
from chunker import chunk_text

class DocumentProcessor:
    def __init__(self):
        self.milvus_api_url = "http://milvus-api:8000"
    
    async def process_and_upload_document(self, file_path):
        # 1. 문서 파싱
        content = parse_pdf(file_path)
        
        # 2. 텍스트 청킹
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        
        # 3. 메타데이터 추출
        metadata = {
            "title": extract_title(file_path),
            "file_path": file_path,
            "file_type": "pdf",
            "author": extract_author(content),
            "tags": extract_tags(content),
            "metadata": {
                "page_count": len(content.pages),
                "language": detect_language(content)
            }
        }
        
        # 4. Milvus API로 한 번에 전송 ⭐
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.milvus_api_url}/data/insert",
                json={
                    "collection_name": "rag_docs_bot01",
                    "document": metadata,  # 메타데이터
                    "chunks": [  # 텍스트 청크
                        {
                            "chunk_index": i,
                            "text": chunk.text,
                            "page_number": chunk.page
                        }
                        for i, chunk in enumerate(chunks)
                    ]
                }
            )
        
        return response.json()

# 사용 예시
processor = DocumentProcessor()
result = await processor.process_and_upload_document("ai_intro.pdf")
# → Milvus API 서버가 PostgreSQL + Milvus 모두 처리
print(f"문서 저장 완료: doc_id={result['doc_id']}")
```

**전처리 서버의 책임:**
- ✅ 문서 파싱 (PDF, DOCX, HTML 등)
- ✅ 텍스트 추출
- ✅ 청킹 (의미 단위 분할)
- ✅ 메타데이터 추출
- ✅ Milvus API 한 곳만 호출 (간단!)

**Milvus API 서버의 책임:**
- ✅ 텍스트 임베딩 처리
- ✅ PostgreSQL 저장 (메타데이터)
- ✅ Milvus 저장 (벡터)
- ✅ 트랜잭션 관리

---

## 5️⃣-1. Milvus 실시간 데이터 검색 메커니즘

### 🔥 파티션당 300만 개 + 신규 데이터 실시간 검색

**핵심**: 파티션이 이미 로드되어 있으면, 새 데이터 삽입 → Flush → 1초 이내 검색 가능!

#### Milvus 세그먼트 구조

```
파티션 (bot_550e8400...) - 300만 개 벡터
├── Sealed Segments (불변, 인덱스 완료)
│   ├── Segment 1: 100만 개 (HNSW 인덱스) → 5ms
│   ├── Segment 2: 100만 개 (HNSW 인덱스) → 5ms
│   └── Segment 3: 100만 개 (HNSW 인덱스) → 5ms
│
└── Growing Segments (가변, 신규 데이터) ⭐
    └── Segment 4: 신규 500개 (Brute-force) → 1ms
    
총 검색 시간: max(5ms) = 5ms (병렬 처리)
신규 데이터 영향: +1ms (거의 없음!)
```

#### 데이터 상태 흐름

```python
"""
1. Insert → 메모리 버퍼 (검색 불가)
2. Flush  → 디스크 저장 (검색 가능!) ⭐
3. Seal   → 세그먼트 고정 (자동)
4. Index  → HNSW 인덱스 구축 (백그라운드)
"""

# 1. 데이터 삽입
collection.insert(data=entities, partition_name="bot_550e8400...")
# 상태: 메모리 버퍼, 검색 ❌

# 2. Flush (1초 이내)
collection.flush()
# 상태: 디스크 저장, 검색 ✅

# 3. 검색 (기존 300만 + 신규 500개 모두 검색)
results = collection.search(
    data=[query_vector],
    partition_names=["bot_550e8400..."],
    limit=5
)
# 기존 Sealed Segments: HNSW 인덱스 (5ms)
# 신규 Growing Segment: Brute-force (1ms)
# 총: 5ms (병렬 처리로 max 값)
```

#### 자동 Flush 전략

```python
# 방법 1: 즉시 Flush (동기, 블로킹)
collection.insert(data=entities)
collection.flush()  # 0.5~1초 대기
# 장점: 즉시 검색 가능
# 단점: 응답 느림

# 방법 2: 백그라운드 Flush (비동기, 권장) ⭐
collection.insert(data=entities)
await auto_flusher.mark_for_flush(collection_name)  # 즉시 반환
# 장점: 응답 빠름 (0.1초)
# 단점: 1초 후 검색 가능
# 결론: 대부분의 사용 사례에 적합!
```

#### 성능 영향

| Growing Segment 크기 | Brute-force 시간 | 전체 검색 시간 | 영향 |
|----------------------|------------------|----------------|------|
| 100개 | 0.1ms | 15.1ms | ✅ 거의 없음 |
| 500개 | 0.5ms | 15.5ms | ✅ 거의 없음 |
| 1,000개 | 1ms | 16ms | ✅ 거의 없음 |
| 5,000개 | 5ms | 20ms | ✅ 허용 가능 |
| 10,000개 | 10ms | 25ms | ⚠️ 약간 영향 |

**결론**: 신규 데이터 수천 개까지는 검색 성능에 거의 영향 없음!

---

## 5️⃣-2. FastAPI 시작 시 컬렉션 로드 전략

### 🚀 전체 컬렉션 사전 로드 + 신규 파티션 실시간 관리

#### 로드 전략

```python
# FastAPI 시작 시
1. 특정 컬렉션(collection_chatty)의 모든 파티션 로드
   - 10개 파티션 × 12초 = 120초 (병렬 처리로 단축)
   - 배치 단위 로드 (5개씩)
   
2. 신규 봇 등록 시
   - 파티션 생성 + 즉시 로드
   - 로드 시간: 12초 (비동기 처리 가능)
   
3. 봇 삭제 시
   - 파티션 언로드 + 삭제
   - 메모리 즉시 해제
```

#### CPU 서버 파티션 로드 성능

| 하드웨어 | 메모리 | 300만 벡터 로드 시간 | 상태 |
|---------|--------|---------------------|------|
| CPU 8코어, 32GB | 25GB | 15-25초 | ✅ 양호 |
| CPU 16코어, 64GB | 25GB | 10-15초 | ✅ 우수 |
| CPU 32코어, 128GB | 25GB | 5-10초 | ✅ 최적 |

**메모리 계산:**
```
300만 벡터 × 1536차원 × 4바이트 = 18.4GB
+ 인덱스 오버헤드 (20%) = 3.7GB
+ 메타데이터 (5%) = 0.9GB
+ 시스템 오버헤드 (10%) = 2GB
= 총 25GB per partition
```

#### 파티션 관리 전략

```python
# 시나리오 1: 소규모 (10개 봇)
- 전략: 모든 파티션 사전 로드
- 메모리: 10 × 25GB = 250GB
- 장점: 모든 검색 즉시 응답 (0ms)

# 시나리오 2: 중규모 (50개 봇)
- 전략: 자주 사용되는 10개만 사전 로드, 나머지 지연 로드
- 메모리: 10 × 25GB = 250GB
- 장점: 80% 즉시 응답, 메모리 효율적

# 시나리오 3: 대규모 (100개+ 봇)
- 전략: LRU 캐시 방식 (최대 20개 로드)
- 메모리: 20 × 25GB = 500GB
- 장점: 메모리 효율적, 자동 관리
```

#### 일관성 레벨

```python
# 검색 시 일관성 레벨 설정
results = collection.search(
    data=[query_vector],
    partition_names=["bot_550e8400..."],
    consistency_level="Strong",  # ⭐ 최신 데이터 보장
    limit=5
)

"""
Consistency Level:
- Strong (권장): Flush 후 즉시 검색 가능, 최신 데이터 보장
- Bounded: 약간의 지연 허용 (1초 이내)
- Eventually: 최대 지연 허용, 최고 성능
"""
```

---

## 6️⃣ 파티셔닝 아키텍처 (대용량 데이터 처리)

### 🎯 최종 구조: 계정=컬렉션, 봇=파티셔닝

**완벽한 대칭 구조로 양쪽 모두 파티셔닝!**

```
┌─────────────────────────────────────────────────────────────┐
│           계정 레벨 (chatty)                                  │
│─────────────────────────────────────────────────────────────│
│  Milvus: chatty_collection (단일 컬렉션) ⭐                  │
│  PostgreSQL: rag_db_chatty (단일 DB) ⭐                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           봇 레벨 (파티셔닝) ⭐                               │
│─────────────────────────────────────────────────────────────│
│  Milvus 파티션:                PostgreSQL 파티션:           │
│  ├── news_bot (뉴스봇)         ├── documents_550e8400...    │
│  │   └── 300만 벡터             │   └── 300만 행             │
│  ├── law_bot (법률봇)          ├── documents_7c9e6679...    │
│  │   └── 300만 벡터             │   └── 300만 행             │
│  ├── medical_bot (의료봇)      ├── documents_8a7f5678...    │
│  │   └── 300만 벡터             │   └── 300만 행             │
│  └── ... (100개 봇)           └── ... (100개 파티션)       │
│                                                             │
│  총: 3억 벡터                   총: 3억 행                   │
└─────────────────────────────────────────────────────────────┘

매핑 테이블: bot_registry
├── bot_id (UUID) → partition_name (예: news_bot, law_bot)
└── bot_name (예: 뉴스봇, 법률봇)
```

### 🔥 구조 비교: 양쪽 파티셔닝 vs 기타 방안

| 항목 | 방안1: 봇별 컬렉션/테이블 | 방안2: Milvus만 파티셔닝 | **방안3: 양쪽 모두 파티셔닝 (채택)** ✅ |
|------|----------------------|---------------------|--------------------------------|
| **Milvus** | news_bot_collection (100개) | chatty_collection (1개) + 파티션 | **chatty_collection (1개) + 파티션** ⭐ |
| **PostgreSQL** | news_bot 테이블 (100개) | news_bot 테이블 (100개) | **documents (1개) + 파티션** ⭐ |
| **관리** | 컬렉션 100개 + 테이블 100개 😭 | 컬렉션 1개 + 테이블 100개 😐 | **컬렉션 1개 + 테이블 1개** ✅ |
| **Milvus 검색** | 15ms (개별) | **15ms (파티션)** ✅ | **15ms (파티션)** ✅ |
| **PostgreSQL 조회** | 0.15초 (개별) | 0.15초 (개별) 😐 | **0.15초 (파티션)** ✅ |
| **전체 통계** | Milvus 100번, PostgreSQL 100번 😭 | Milvus 1번, PostgreSQL 100번 😐 | **양쪽 모두 1번!** ⭐ |
| **일관성** | 구조 다름 😭 | 구조 다름 😐 | **완벽한 대칭!** ⭐ |
| **확장성** | 수동 생성 😭 | 반자동 😐 | **완전 자동** ✅ |

### ⚡ 양쪽 파티션 프루닝 효과

#### PostgreSQL 파티션 프루닝
```sql
-- 뉴스봇 문서 검색
SELECT * FROM documents 
WHERE chat_bot_id = '550e8400-e29b-41d4-a716-446655440000'
  AND title LIKE '%AI%';

-- PostgreSQL 실행 계획:
-- ✅ documents_550e8400e29b41d4... 파티션만 스캔 (300만 행)
-- ❌ 다른 99개 파티션 무시 (2억9700만 행 스킵!)
-- 결과: 0.15초 vs 15초 (100배 차이!)
```

#### Milvus 파티션 지정 검색
```python
# Milvus 검색 (파티션 지정!)
collection = Collection(name="chatty_collection")
collection.load(partition_names=["news_bot"])  # 뉴스봇 파티션만 로드

results = collection.search(
    data=[query_vector],
    anns_field="embedding_dense",
    param=search_params,
    limit=5,
    partition_names=["news_bot"],  # ⭐ 이 파티션만 검색!
    expr="chat_bot_id == '550e8400-e29b-41d4-a716-446655440000'"
)

# 결과:
# ✅ news_bot 파티션만 검색 (300만 벡터)
# ❌ 다른 99개 파티션 무시 (2억9700만 벡터 스킵!)
# 성능: 15ms vs 150ms (10배 차이!)
```

### 📦 양쪽 자동 파티션 생성

```json
// API 요청: 봇 등록
POST /collection/create
{
  "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "bot_name": "뉴스봇",
  "partition_name": "news_bot",
  "dimension": 1536
}
```

**서버 처리 흐름:**

```sql
-- 1. PostgreSQL: 봇 레지스트리에 등록
INSERT INTO bot_registry (bot_id, bot_name, partition_name)
VALUES ('550e8400-e29b-41d4-a716-446655440000', '뉴스봇', 'news_bot');

-- 2. PostgreSQL 트리거가 자동으로 파티션 생성 ⚡
-- documents_550e8400e29b41d4a716446655440000
-- document_chunks_550e8400e29b41d4a716446655440000
-- + 인덱스 자동 생성
```

```python
# 3. Milvus: chatty_collection 내 파티션 생성 ⚡
collection = Collection(name="chatty_collection")
collection.create_partition(partition_name="news_bot")
# → chatty_collection/news_bot 파티션 생성
```

```sql
-- 4. 바로 사용 가능!
-- PostgreSQL
INSERT INTO documents (chat_bot_id, title, content)
VALUES ('550e8400-e29b-41d4-a716-446655440000', '첫 문서', '...');
-- → 자동으로 documents_550e8400... 파티션에 저장
```

```python
# Milvus
collection.insert(entities, partition_name="news_bot")
# → 자동으로 chatty_collection/news_bot 파티션에 저장
```

### 📊 성능 비교 (양쪽 파티셔닝 효과)

#### PostgreSQL 성능

| 봇 수 | 총 문서 | 파티셔닝 없음 | **파티셔닝 사용** | 성능 향상 |
|------|---------|-------------|----------------|----------|
| 1개 | 300만 | 0.15초 | 0.15초 | 동일 |
| 10개 | 3,000만 | 1.5초 | **0.15초** | **10배** ⚡ |
| 100개 | 3억 | 15초 | **0.15초** | **100배** ⚡ |
| 1,000개 | 30억 | 150초 | **0.15초** | **1000배** ⚡ |

#### Milvus 성능

| 봇 수 | 총 벡터 | 컬렉션 분리 | 파티션 미지정 | **파티션 지정** | 효과 |
|------|---------|-----------|-------------|--------------|------|
| 1개 | 300만 | 15ms | 15ms | 15ms | 동일 |
| 10개 | 3,000만 | 15ms (개별) | 150ms | **15ms** | **10배** ⚡ |
| 100개 | 3억 | 15ms (개별) | 1,500ms | **15ms** | **100배** ⚡ |

#### 통합 검색 성능 (Milvus + PostgreSQL)

| 봇 수 | 기존 (봇별 분리) | 파티셔닝 미사용 | **양쪽 파티셔닝** | 최종 효과 |
|------|---------------|-------------|---------------|----------|
| 100개 | 0.17초 | 16.5초 😭 | **0.17초** ✅ | **97배 빠름!** ⚡ |

### 🔧 파티션 관리

#### PostgreSQL 파티션
```sql
-- 파티션 목록 확인
SELECT tablename, pg_size_pretty(pg_total_relation_size('public.'||tablename))
FROM pg_tables
WHERE tablename LIKE 'documents_%';

-- 봇별 문서 수 확인 (단일 쿼리로 전체 통계!)
SELECT 
    br.bot_name,
    br.partition_name,
    COUNT(d.*) as doc_count,
    pg_size_pretty(SUM(pg_column_size(d.*))) as total_size
FROM bot_registry br
LEFT JOIN documents d ON br.bot_id = d.chat_bot_id
GROUP BY br.bot_name, br.partition_name
ORDER BY doc_count DESC;

-- 특정 봇 파티션만 VACUUM (독립적 유지보수)
VACUUM ANALYZE documents_550e8400e29b41d4a716446655440000;
```

#### Milvus 파티션
```python
# Milvus 파티션 목록 확인
from pymilvus import Collection

collection = Collection(name="chatty_collection")
partitions = collection.partitions
for partition in partitions:
    print(f"파티션: {partition.name}, 엔티티 수: {partition.num_entities}")

# 출력:
# 파티션: news_bot, 엔티티 수: 3000000
# 파티션: law_bot, 엔티티 수: 3000000
# 파티션: medical_bot, 엔티티 수: 3000000

# 특정 파티션만 로드/언로드 (메모리 관리)
collection.load(partition_names=["news_bot"])  # 필요한 것만 로드
collection.release(partition_names=["news_bot"])  # 사용 후 해제
```

### 🎯 계정-컬렉션-파티션 매핑 (최종)

```
계정 레벨 (1:1:1 매핑):
  account_name: "chatty"
  ├── Milvus 컬렉션: collection_chatty ⭐
  └── PostgreSQL DB: rag_db_chatty ⭐

봇 레벨 (파티셔닝):
  ├── 뉴스봇
  │   ├── bot_id: "550e8400-e29b-41d4-a716-446655440000"
  │   ├── Milvus 파티션: collection_chatty / news_bot
  │   └── PostgreSQL 파티션: rag_db_chatty / documents_550e8400e29b41d4...
  │
  ├── 법률봇
  │   ├── bot_id: "7c9e6679-7425-40de-944b-e07fc1f90ae7"
  │   ├── Milvus 파티션: collection_chatty / law_bot
  │   └── PostgreSQL 파티션: rag_db_chatty / documents_7c9e6679742540de...
  │
  └── 의료봇
      ├── bot_id: "8a7f5678-8234-51ef-b345-567890abcdef"
      ├── Milvus 파티션: collection_chatty / medical_bot
      └── PostgreSQL 파티션: rag_db_chatty / documents_8a7f5678823451ef...
```

### 📋 네이밍 규칙

| 레벨 | 입력 | Milvus | PostgreSQL |
|------|------|--------|-----------|
| **계정** | `account_name="chatty"` | `collection_chatty` ⭐ | `rag_db_chatty` ⭐ |
| **계정** | `account_name="enterprise"` | `collection_enterprise` ⭐ | `rag_db_enterprise` ⭐ |
| **봇** | `bot_id="550e8400..."` + `partition_name="news_bot"` | 파티션: `news_bot` | 파티션: `documents_550e8400...` |

### 💡 프리픽스 전략

```python
# Milvus: collection_ 프리픽스
account_name = "chatty"
collection_name = f"collection_{account_name}"  # "collection_chatty" ⭐

# PostgreSQL: rag_db_ 프리픽스
db_name = f"rag_db_{account_name}"  # "rag_db_chatty" ⭐

# 장점:
# ✅ 모든 컬렉션이 collection_로 시작 (관리 편의)
# ✅ 모든 DB가 rag_db_로 시작 (시스템 DB와 구분)
# ✅ 일관된 네이밍 규칙
```

### 💡 핵심 설계 원칙

1. **계정 = 컬렉션 = DB** (1:1:1 매핑)
   - `account_name="chatty"` → `collection_chatty` (Milvus) + `rag_db_chatty` (PostgreSQL)
   - `account_name="enterprise"` → `collection_enterprise` (Milvus) + `rag_db_enterprise` (PostgreSQL)

2. **봇 = 파티션** (양쪽 모두)
   - 뉴스봇 → `news_bot` (Milvus 파티션) + `documents_550e8400...` (PostgreSQL 파티션)
   - 법률봇 → `law_bot` (Milvus 파티션) + `documents_7c9e6679...` (PostgreSQL 파티션)

3. **완벽한 대칭성**
   - Milvus와 PostgreSQL의 구조가 동일 → 이해하기 쉬움
   - `collection_{계정}` ↔ `rag_db_{계정}`

4. **자동화**
   - 봇 등록 시 양쪽 파티션 자동 생성 → 코드 변경 없음

5. **확장성**
   - 새 계정 추가: `collection_{새계정}` + `rag_db_{새계정}` 생성
   - 새 봇 추가: 해당 계정의 컬렉션/DB에 파티션 추가

---

## 7️⃣ 프로젝트 구조

```
08. Milvus API Server/
├── app/                                # 메인 애플리케이션 디렉토리
│   ├── __init__.py
│   ├── main.py                         # FastAPI 애플리케이션 진입점
│   ├── config.py                       # 설정 관리 (환경변수, DB 설정)
│   │
│   ├── api/                            # API 라우터 레이어
│   │   ├── __init__.py
│   │   ├── collection.py               # 컬렉션 생성/관리 API
│   │   ├── data.py                     # 데이터 삽입/조회/수정/삭제 API
│   │   └── search.py                   # 유사도 검색 API
│   │
│   ├── core/                           # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── milvus_client.py            # Milvus 연결 및 작업 처리
│   │   ├── postgres_client.py          # PostgreSQL 연결 및 작업 처리
│   │   ├── embedding.py                # 임베딩 처리 (OpenAI, BGE-M3 등)
│   │   └── transaction.py              # 트랜잭션 관리 (Milvus + PostgreSQL)
│   │
│   ├── models/                         # Pydantic 모델 (Request/Response)
│   │   ├── __init__.py
│   │   ├── collection.py               # 컬렉션 관련 모델
│   │   ├── document.py                 # 문서/청크 모델
│   │   └── search.py                   # 검색 요청/응답 모델
│   │
│   ├── schemas/                        # 데이터베이스 스키마
│   │   ├── __init__.py
│   │   ├── milvus_schema.py            # Milvus 컬렉션 스키마 정의
│   │   └── postgres_schema.py          # PostgreSQL 테이블 스키마 정의
│   │
│   └── utils/                          # 유틸리티 함수
│       ├── __init__.py
│       ├── logger.py                   # 로깅 설정
│       └── exceptions.py               # 커스텀 예외 처리
│
├── migrations/                         # PostgreSQL 마이그레이션
│   └── init.sql                        # 초기 테이블 생성 스크립트
│
├── tests/                              # 테스트 코드
│   ├── __init__.py
│   ├── test_api.py                     # API 엔드포인트 테스트
│   └── test_core.py                    # 핵심 로직 테스트
│
├── .env.example                        # 환경변수 예시 파일
├── .gitignore                          # Git 제외 파일 목록
├── requirements.txt                    # Python 패키지 의존성
├── README.md                           # 프로젝트 설명서
└── RAG_Milvus_API_Design.md           # 시스템 설계 문서
```

### 🔹 레이어별 역할

#### 1. **API 레이어** (`app/api/`)
- FastAPI 라우터 정의
- 요청/응답 검증 (Pydantic)
- HTTP 엔드포인트 노출
- 라우팅만 담당, 비즈니스 로직은 `core/`로 위임

#### 2. **Core 레이어** (`app/core/`)
- 실제 비즈니스 로직 구현
- Milvus, PostgreSQL 클라이언트 관리
- 임베딩 처리 (OpenAI API, 로컬 모델 등)
- 트랜잭션 관리 및 롤백 처리

#### 3. **Models 레이어** (`app/models/`)
- Pydantic 모델 정의
- API 요청/응답 스키마
- 데이터 검증 및 직렬화

#### 4. **Schemas 레이어** (`app/schemas/`)
- Milvus 컬렉션 스키마 정의
- PostgreSQL 테이블 스키마 정의
- 데이터베이스 초기화 로직

#### 5. **Utils 레이어** (`app/utils/`)
- 로깅 설정
- 예외 처리
- 공통 유틸리티 함수

### 🔹 주요 파일 설명

| 파일 | 역할 |
|------|------|
| `app/main.py` | FastAPI 애플리케이션 진입점, 라우터 등록, CORS 설정 |
| `app/config.py` | 환경변수 로드, 설정 관리 (Milvus/PostgreSQL 연결 정보) |
| `app/core/milvus_client.py` | Milvus 연결, 컬렉션 생성, 벡터 CRUD |
| `app/core/postgres_client.py` | PostgreSQL 연결, 문서/청크 CRUD |
| `app/core/embedding.py` | 텍스트 → 벡터 변환 (OpenAI, BGE-M3 등) |
| `app/core/transaction.py` | Milvus + PostgreSQL 통합 트랜잭션 |
| `migrations/init.sql` | PostgreSQL 초기 테이블 생성 SQL |
| `requirements.txt` | FastAPI, pymilvus, asyncpg, openai 등 |
| `.env.example` | 환경변수 템플릿 |

### 🔹 확장 가능한 구조

```python
# 예시: 새로운 임베딩 모델 추가
# app/core/embedding.py

class EmbeddingService:
    def __init__(self, model_type: str):
        if model_type == "openai":
            self.embedder = OpenAIEmbedder()
        elif model_type == "bge-m3":
            self.embedder = BGEEmbedder()
        elif model_type == "sentence-bert":
            self.embedder = SentenceBertEmbedder()
        # 새 모델 추가 시 여기에 추가
    
    async def embed(self, text: str) -> List[float]:
        return await self.embedder.embed(text)
```

### 🔹 환경 설정 예시 (파티셔닝 기반)

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Milvus 설정 (단일 컬렉션 + 파티셔닝)
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "chatty_collection"  # ⭐ 단일 컬렉션
    
    # PostgreSQL 설정 (단일 DB + 파티셔닝)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB_NAME: str = "rag_db_chatty"  # ⭐ 단일 DB
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    
    # 봇 테이블 네이밍
    BOT_TABLE_PREFIX: str = "bot_"
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = "openai"  # or "bge-m3", "sentence-bert"
    OPENAI_API_KEY: str = ""
    EMBEDDING_DIMENSION: int = 1536
    
    # 하이브리드 검색 설정
    USE_SPARSE_EMBEDDING: bool = False
    
    def get_bot_table_name(self, bot_id: str) -> str:
        """UUID → PostgreSQL 테이블명 변환"""
        sanitized = bot_id.replace("-", "").replace("_", "").lower()
        return f"{self.BOT_TABLE_PREFIX}{sanitized}"
    
    class Config:
        env_file = ".env"
```

### 🔹 API 요청 예시 (최종)

#### 봇 등록
```json
POST /collection/create
{
  "account_name": "chatty",  // ⭐ 계정명
  "chat_bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "bot_name": "뉴스봇",
  "partition_name": "news_bot"
}

// 서버가 자동으로:
// - Milvus: collection_chatty 컬렉션의 news_bot 파티션 생성
// - PostgreSQL: rag_db_chatty DB의 documents_550e8400... 파티션 생성
```

#### 문서 삽입
```json
POST /data/insert
{
  "account_name": "chatty",  // ⭐ 계정명 (컬렉션/DB 선택)
  "document": {
    "chat_bot_id": "550e8400-...",  // ⭐ 봇 ID (파티션 선택)
    "title": "AI 뉴스",
    ...
  },
  "chunks": [...]
}

// 서버가 자동으로:
// - Milvus: collection_chatty 컬렉션 사용
//   → bot_registry에서 partition_name 조회 (news_bot)
//   → collection_chatty/news_bot 파티션에 저장
// 
// - PostgreSQL: rag_db_chatty DB 사용
//   → WHERE chat_bot_id = '550e8400...' 
//   → documents_550e8400... 파티션 자동 선택 (Partition Pruning)
```

#### 검색
```json
POST /search/query
{
  "account_name": "chatty",  // collection_chatty, rag_db_chatty 사용
  "chat_bot_id": "550e8400-...",  // news_bot 파티션만 검색
  "query_text": "AI 최신 동향",
  "limit": 5
}

// 성능:
// - Milvus: collection_chatty/news_bot 파티션만 검색 (300만/3억) ⚡
// - PostgreSQL: documents_550e8400... 파티션만 스캔 (300만/3억) ⚡
// - 결과: 100배 빠름!
```
