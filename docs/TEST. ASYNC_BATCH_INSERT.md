# 무중단 배치 삽입 API 설계서

## 📋 개요

클라이언트에서 대량 문서를 전송할 때, FastAPI 서버가 재시작되어도 작업이 중단되지 않고 계속 진행되도록 보장하는 비동기 배치 삽입 API입니다.

### 목표
- ✅ **무중단**: 서버 재시작 시에도 작업 자동 재개
- ✅ **추적 가능**: Job ID로 상태 조회
- ✅ **동시성**: 여러 배치 작업 병렬 처리
- ✅ **부분 실패 허용**: 일부 문서 실패해도 나머지 처리
- ✅ **중복 처리**: 기존 중복 체크 로직 유지

---

## 🏗️ 아키텍처

```
┌─────────────────┐
│   클라이언트    │
│                 │
│ 1. POST /job    │────┐
│    → job_id     │    │
│                 │    │
│ 2. GET /job/:id │────┼─────────────────┐
│    ← status     │    │                 │
│                 │    │                 │
└─────────────────┘    │                 │
                       │                 │
                       ▼                 ▼
              ┌─────────────────────────────────────┐
              │     FastAPI Server                  │
              │                                     │
              │  ┌─────────────────────────────┐   │
              │  │  Job Queue Manager (Redis)  │   │
              │  │                             │   │
              │  │  • job_id: {                │   │
              │  │      status: pending        │   │
              │  │      request: {...}         │   │
              │  │      results: []            │   │
              │  │      created_at: ...        │   │
              │  │      updated_at: ...        │   │
              │  │    }                        │   │
              │  └─────────────────────────────┘   │
              │                │                    │
              │                ▼                    │
              │  ┌─────────────────────────────┐   │
              │  │  Background Worker          │   │
              │  │  (async def process_job)    │   │
              │  │                             │   │
              │  │  1. 중복 체크               │   │
              │  │  2. PostgreSQL 삽입         │   │
              │  │  3. 임베딩 생성             │   │
              │  │  4. Milvus 삽입             │   │
              │  │  5. 상태 업데이트           │   │
              │  └─────────────────────────────┘   │
              │                                     │
              └─────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          ┌─────────┐ ┌──────────┐ ┌─────────┐
          │PostgreSQL│ │Embedding │ │ Milvus  │
          │         │ │  Service │ │         │
          └─────────┘ └──────────┘ └─────────┘
```

---

## 🔑 핵심 구성 요소

### 1. Job Queue Manager (Redis 기반)

**Redis 선택 이유:**
- ✅ **빠름**: 메모리 기반으로 읽기/쓰기 초고속
- ✅ **TTL 지원**: 자동 만료로 메모리 관리
- ✅ **Pub/Sub**: 실시간 상태 알림 가능
- ✅ **기존 인프라**: 이미 `redis_partition_manager`에서 사용 중

**Redis 키 구조:**
```
# Job 메타데이터 (Hash)
job:{job_id}:meta → {
    "status": "processing",
    "account_name": "chatty",
    "total_documents": 100,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:05:00",
    "ttl_hours": 24
}

# Job 처리 결과 (List)
job:{job_id}:results → [
    {"doc_id": 1, "content_name": "url1", "success": true},
    {"doc_id": 2, "content_name": "url2", "success": false, "error": "..."},
    ...
]

# 진행 중 Job 목록 (Set)
active_jobs → {job_id1, job_id2, ...}

# 처리 큐 (List, Optional)
job_queue → ["job_id1", "job_id2", ...]
```

**TTL (Time To Live):**
- Job 생성: 24시간 후 자동 삭제
- 완료 Job: 7일 후 자동 삭제
- 실패 Job: 3일 후 자동 삭제

---

### 2. API 엔드포인트

#### 2.1. 배치 삽입 Job 생성
```http
POST /job/batch-insert
Content-Type: application/json

Request Body:
{
    "account_name": "chatty",
    "documents": [
        {
            "chat_bot_id": "bot001",
            "content_name": "https://example.com/article1",
            "chunks": [
                {"chunk_index": 0, "text": "첫 번째 문단..."},
                {"chunk_index": 1, "text": "두 번째 문단..."}
            ],
            "metadata": {
                "title": "AI 학습 가이드",
                "author": "김개발"
            }
        },
        ...
    ]
}

Response (202 Accepted):
{
    "job_id": "abc123-def456-ghi789",
    "status": "pending",
    "total_documents": 100,
    "created_at": "2024-01-01T00:00:00",
    "message": "Job created successfully. Use GET /job/{job_id} to check status."
}
```

#### 2.2. Job 상태 조회
```http
GET /job/{job_id}

Response (200 OK):
{
    "job_id": "abc123-def456-ghi789",
    "status": "processing",  # pending, processing, completed, failed
    "progress": {
        "total_documents": 100,
        "processed_documents": 45,
        "success_count": 42,
        "failure_count": 3,
        "percentage": 45.0
    },
    "results": [
        {
            "content_name": "https://example.com/article1",
            "doc_id": 12345,
            "success": true,
            "chunks_count": 5
        },
        {
            "content_name": "https://example.com/article2",
            "success": false,
            "error": "이미 존재하는 문서",
            "chunks_count": 0
        }
    ],
    "timing": {
        "created_at": "2024-01-01T00:00:00",
        "started_at": "2024-01-01T00:00:05",
        "updated_at": "2024-01-01T00:05:30",
        "estimated_completion": "2024-01-01T00:06:00"
    },
    "message": "Job is processing..."
}
```

#### 2.3. Job 취소
```http
DELETE /job/{job_id}

Response (200 OK):
{
    "job_id": "abc123-def456-ghi789",
    "status": "cancelled",
    "message": "Job cancelled successfully"
}
```

---

### 3. Job 상태 플로우

```
┌──────────┐
│ pending  │ ← Job 생성 즉시
└────┬─────┘
     │ Background Worker 시작
     ▼
┌─────────────────┐
│   processing    │ ← 작업 진행 중
└────┬─────┬──────┘
     │     │
     │     └──► (실패)
     │            │
     │            ▼
     │      ┌──────────┐
     │      │  failed  │
     │      └──────────┘
     │
     └──► (성공)
            │
            ▼
      ┌───────────┐
      │ completed │
      └───────────┘
```

**상태별 동작:**
- **pending**: Job 생성됨, 아직 처리 안 됨
- **processing**: 작업 진행 중 (일부 문서 처리 완료/실패)
- **completed**: 모든 작업 완료 (성공 또는 부분 성공)
- **failed**: 전체 작업 실패 (시스템 오류 등)
- **cancelled**: 사용자가 취소

---

### 4. Background Worker

**구현:**
- `FastAPI.BackgroundTasks` 사용 안 함 (서버 재시작 시 소실)
- `asyncio.create_task()`로 독립 태스크 생성
- `App.startup` 이벤트에서 Worker 풀 시작
- 재시작 시 Redis에서 `active_jobs` 스캔하여 재개

**처리 로직:**
```python
async def process_job(job_id: str):
    try:
        # 1. 상태: pending → processing
        await job_manager.update_status(job_id, "processing", started_at=datetime.now())
        
        # 2. Job 메타데이터 조회
        job_data = await job_manager.get_job(job_id)
        request_data = job_data["request"]
        
        # 3. 중복 체크 (기존 로직 재사용)
        unique_docs, failed_docs = await check_duplicates(request_data)
        
        # 4. 문서별 개별 처리 (기존 batch_insert_documents 로직 재사용)
        successful_docs = []
        failed_processing_docs = []
        
        for doc in unique_docs:
            try:
                # PostgreSQL → Embedding → Milvus
                doc_id = await insert_document_saga(doc)
                
                successful_docs.append({
                    "content_name": doc.content_name,
                    "doc_id": doc_id,
                    "success": True
                })
                
                # 진행 상태 업데이트
                await job_manager.update_progress(job_id, len(successful_docs))
                
            except Exception as e:
                failed_processing_docs.append({
                    "content_name": doc.content_name,
                    "success": False,
                    "error": str(e)
                })
        
        # 5. 최종 결과 저장
        all_results = successful_docs + failed_docs + failed_processing_docs
        await job_manager.save_results(job_id, all_results)
        
        # 6. 상태: processing → completed
        await job_manager.update_status(
            job_id, 
            "completed",
            updated_at=datetime.now()
        )
        
    except Exception as e:
        # 상태: processing → failed
        await job_manager.update_status(job_id, "failed", error=str(e))
```

---

### 5. 서버 재시작 시 복구 로직

**`app/main.py`의 `startup` 이벤트:**

```python
@app.on_event("startup")
async def startup_event():
    # 기존 로직...
    
    # 🆕 미완료 Job 복구
    logger.info("🔍 Checking for incomplete jobs...")
    incomplete_jobs = await job_manager.get_incomplete_jobs()
    
    if incomplete_jobs:
        logger.info(f"🔄 Recovering {len(incomplete_jobs)} incomplete jobs")
        for job_id in incomplete_jobs:
            asyncio.create_task(process_job(job_id))
            logger.info(f"  ✅ Job resumed: {job_id}")
    else:
        logger.info("✨ No incomplete jobs found")
```

**복구 조건:**
- Redis에 `job:{job_id}:meta` 존재
- `status`가 `pending` 또는 `processing`
- `ttl`이 아직 남음

---

## 📊 성능 고려사항

### 동시 처리
- **Worker 수**: CPU 코어 수 * 2
- **우선순위**: FIFO (순차 처리)
- **제한**: 동시에 최대 10개 Job 처리

### 메모리 관리
- Redis Job 메타데이터: Hash (최대 10KB/Job)
- Redis 결과 데이터: List (최대 1MB/Job)
- TTL 적용으로 오래된 Job 자동 삭제

### 클라이언트 상태 조회 전략

**옵션 1: 폴링 (Polling)** ⭐ 권장
```python
# Exponential Backoff로 주기적 조회
while True:
    status = await get_job_status(job_id)
    
    if status in ["completed", "failed", "cancelled"]:
        break
    
    # 1초 → 2초 → 4초 → 8초 → 16초 (최대)
    await asyncio.sleep(min(2 ** retry_count, 16))
    retry_count += 1
```
✅ **장점**: 구현 간단, 서버 부하 적음  
❌ **단점**: 완료 즉시 알림 불가

**옵션 2: Webhook 콜백** 🔔
```python
# Job 생성 시 webhook_url 전달
POST /job/batch-insert
{
    "account_name": "chatty",
    "documents": [...],
    "webhook_url": "https://your-server.com/job/notify"  # 🆕
}

# 완료 시 서버가 자동 호출
POST https://your-server.com/job/notify
{
    "job_id": "abc123",
    "status": "completed",
    "progress": {...},
    "results": [...]
}
```
✅ **장점**: 완료 즉시 알림, 실시간  
❌ **단점**: 외부 서버 필요, 보안 고려

**옵션 3: Server-Sent Events (SSE)** 📡
```python
# 연결 유지하며 실시간 이벤트 수신
GET /job/{job_id}/stream

# 서버가 상태 변경 시 자동 전송
data: {"job_id": "abc123", "status": "processing", "progress": {...}}
data: {"job_id": "abc123", "status": "completed", "results": [...]}
```
✅ **장점**: 실시간, 효율적  
❌ **단점**: 연결 관리 복잡, 방화벽 이슈 가능

**권장**: **폴링** (간단, 안정적)

---

## 🗂️ 파일 구조

```
app/
├── api/
│   ├── job.py              # 🆕 Job API 엔드포인트
│   └── data.py             # 기존 (유지)
├── core/
│   ├── job_manager.py      # 🆕 Job Queue Manager
│   ├── redis_client.py     # 🆕 Redis 클라이언트 (또는 기존 사용)
│   └── postgres_client.py  # 기존 (유지)
└── models/
    └── job.py              # 🆕 Job 관련 Pydantic 모델

docs/
└── 05. ASYNC_BATCH_INSERT.md  # 🆕 이 문서
```

---

## 🔒 에러 처리

### Job 실패 시나리오

**1. 일부 문서 실패 (부분 성공)**
```
status: "completed"
progress: {
    success_count: 80,
    failure_count: 20
}
```
→ 전체 Job은 성공, 개별 결과에 실패 내역 포함

**2. 전체 실패 (시스템 오류)**
```
status: "failed"
error: "PostgreSQL connection timeout"
```
→ 모든 문서 롤백, Job 전체 실패 처리

**3. 타임아웃**
- 단일 문서: 30초 초과 시 개별 실패
- 전체 Job: 1시간 초과 시 전체 실패

---

## 🧪 테스트 시나리오

### 1. 정상 플로우
```bash
# 1. Job 생성
POST /job/batch-insert
→ job_id 받음

# 2. 상태 조회
GET /job/{job_id}
→ pending → processing → completed

# 3. 결과 확인
→ 100개 모두 성공
```

### 2. 서버 재시작
```bash
# 1. Job 생성 및 처리 시작
POST /job/batch-insert
→ processing

# 2. FastAPI 서버 강제 종료
kill -9 $PID

# 3. FastAPI 서버 재시작
uvicorn app.main:app --reload

# 4. 상태 조회
GET /job/{job_id}
→ processing (자동 재개됨)

# 5. 완료 확인
GET /job/{job_id}
→ completed
```

### 3. 일부 문서 실패
```bash
# 1. Job 생성 (일부 중복 포함)
POST /job/batch-insert
{
    "documents": [
        {"content_name": "existing_url"},  # 중복
        {"content_name": "new_url"}        # 신규
    ]
}

# 2. 완료 확인
GET /job/{job_id}
{
    "status": "completed",
    "progress": {
        "success_count": 1,
        "failure_count": 1
    },
    "results": [
        {"content_name": "existing_url", "success": false, "error": "이미 존재"},
        {"content_name": "new_url", "success": true, "doc_id": 123}
    ]
}
```

---

## 📝 구현 순서

1. ✅ **설계서 작성** (이 문서)
2. 🔲 **Redis 클라이언트 추가/확인**
3. 🔲 **Job Manager 구현** (`app/core/job_manager.py`)
4. 🔲 **Job 모델 정의** (`app/models/job.py`)
5. 🔲 **Job API 엔드포인트** (`app/api/job.py`)
6. 🔲 **Background Worker 구현**
7. 🔲 **재시작 복구 로직** (`app/main.py`)
8. 🔲 **통합 테스트**

---

## ❓ FAQ

**Q: 기존 `/data/insert/batch` API는 폐기하나요?**  
A: 아니요. 기존 API는 동기 방식으로 빠른 응답이 필요한 경우에 사용. 새로운 Job API는 대량/무중단이 필요한 경우에 사용.

**Q: Redis가 다운되면?**  
A: Job 상태가 사라져 복구 불가. Redis HA(High Availability) 설정 권장.

**Q: 처리 중인 Job을 취소하면?**  
A: 현재 처리 중인 문서는 완료하고, 미처리 문서만 중단. 부분 성공 응답 반환.

**Q: Job 결과를 언제까지 보관하나요?**  
A: 완료 후 7일 (Redis TTL). 더 길게 보관하려면 별도 스토리지(PostgreSQL 등)에 저장.

**Q: 동시에 여러 Job을 생성하면?**  
A: 순차 처리 (FIFO). 동시 처리하려면 Worker 수 증가 필요.

---

## 🔄 다음 단계

설계 검토 후 구현 시작:
1. Redis 클라이언트 확인/추가
2. Job Manager 구현
3. API 엔드포인트 구현
4. 통합 테스트

**예상 소요 시간:** 4-6시간

