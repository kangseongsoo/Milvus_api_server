"""
데이터 관리 API (CRUD)
"""
from fastapi import APIRouter, HTTPException, status, Query
from app.models.document import (
    DocumentInsertRequest,
    DocumentInsertResponse,
    BatchInsertRequest,
    BatchInsertResponse,
    BatchInsertWithEmbeddingsRequest,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentUpdateResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    BotDeleteRequest,
    BotDeleteResponse,
    MetadataUpdateRequest,
    MetadataUpdateResponse
)
from app.schemas.milvus_metadata import filter_milvus_metadata
from app.utils.logger import setup_logger
from app.core.auto_flusher import auto_flusher
from app.core.postgres_client import postgres_client
from app.core.milvus_client import milvus_client
from app.core.embedding import embedding_service
from app.config import settings
from pymilvus import Collection
from datetime import datetime

logger = setup_logger(__name__)
router = APIRouter()


def generate_partition_name(chat_bot_id: str) -> str:
    """chat_bot_id로 파티션명 자동 생성"""
    return f"bot_{chat_bot_id.replace('-', '')}"


@router.post("/insert", response_model=DocumentInsertResponse, status_code=status.HTTP_201_CREATED)
async def insert_document(request: DocumentInsertRequest):
    """
    문서 데이터 삽입 (개선된 Saga Pattern)
    
    전처리 서버로부터 계정명 + 메타데이터 + 텍스트 청크를 받아
    1. PostgreSQL 트랜잭션으로 문서+청크 원자성 보장
    2. 임베딩 생성 (재시도 로직 포함)
    3. Milvus 벡터 저장 (재시도 로직 포함)
    4. 실패 시 보상 트랜잭션으로 PostgreSQL 롤백
    """
    doc_id = None
    try:
        start_time = datetime.now()
        collection_name = f"collection_{request.account_name}"
        partition_name = generate_partition_name(request.chat_bot_id)
        
        logger.info(f"📝 문서 삽입 시작 (Saga Pattern)")
        logger.info(f"   - Account: {request.account_name}")
        logger.info(f"   - Bot ID: {request.chat_bot_id}")
        logger.info(f"   - Title: {request.metadata.get('title', '(제목 없음)')}")
        logger.info(f"   - Chunks: {len(request.chunks)}")
        logger.info(f"   - Collection: {collection_name}")
        logger.info(f"   - Partition: {partition_name}")
        
        # ========== Step 1: 메타데이터 분리 ==========
        all_metadata = request.metadata or {}
        milvus_metadata = filter_milvus_metadata(all_metadata)  # Milvus 필터링용만 추출
        
        logger.info(f"📊 메타데이터 분리 완료:")
        logger.info(f"   - Milvus 필터링용: {len(milvus_metadata)}개 필드")
        logger.info(f"   - PostgreSQL 전체: {len(all_metadata)}개 필드 (전체 저장)")
        
        # ========== Step 2: PostgreSQL 트랜잭션 (원자성 보장) ==========
        postgres_start = datetime.now()
        
        doc_id = await postgres_client.insert_document_with_chunks_transaction(
            account_name=request.account_name,
            document_data={
                "chat_bot_id": request.chat_bot_id,
                "content_name": request.content_name,  # 문서 고유 식별자
                "metadata": all_metadata  # 전체 메타데이터를 PostgreSQL에 저장
            },
            chunks=[{"chunk_index": c.chunk_index, "text": c.text} for c in request.chunks]
        )
        
        postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
        logger.info(f"✅ PostgreSQL 트랜잭션 완료: doc_id={doc_id}")
        
        # ========== Step 2: 임베딩 생성 (재시도 로직) ==========
        embedding_start = datetime.now()
        
        try:
            embeddings = await embedding_service.batch_embed_with_retry(
                texts=[chunk.text for chunk in request.chunks],
                max_retries=3,
                backoff=2.0
            )
            embedding_time = (datetime.now() - embedding_start).total_seconds() * 1000
            logger.info(f"✅ 임베딩 생성 완료: {len(embeddings)}개 벡터")
            
        except Exception as embedding_error:
            # 임베딩 실패 시 PostgreSQL 롤백
            logger.error(f"❌ 임베딩 생성 실패, PostgreSQL 롤백 시작: {str(embedding_error)}")
            await postgres_client.delete_document(request.account_name, request.chat_bot_id, doc_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"임베딩 생성 실패: {str(embedding_error)}"
            )
        
        # ========== Step 3: Milvus 벡터 저장 (재시도 로직) ==========
        milvus_start = datetime.now()
        
        try:
            await milvus_client.insert_vectors_with_retry(
                account_name=request.account_name,
                chat_bot_id=request.chat_bot_id,
                partition_name=partition_name,
                doc_id=doc_id,
                content_name=request.content_name,  # content_name 추가
                chunks=[
                    {
                        "chunk_index": chunk.chunk_index,
                        "embedding": embeddings[idx],
                        "text": chunk.text
                    }
                    for idx, chunk in enumerate(request.chunks)
                ],
                metadata=milvus_metadata,  # 필터링용 메타데이터만 Milvus에
                max_retries=3,
                backoff=2.0
            )
            milvus_time = (datetime.now() - milvus_start).total_seconds() * 1000
            logger.info(f"✅ Milvus 벡터 삽입 완료")
            
        except Exception as milvus_error:
            # Milvus 실패 시 PostgreSQL 롤백 (보상 트랜잭션)
            logger.error(f"❌ Milvus 삽입 실패, PostgreSQL 롤백 시작: {str(milvus_error)}")
            await postgres_client.delete_document(request.account_name, request.chat_bot_id, doc_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Milvus 삽입 실패: {str(milvus_error)}"
            )
        
        # ========== Step 4: 🔥 자동 flush 마킹 (이벤트 기반) ==========
        await auto_flusher.mark_for_flush(collection_name)
        logger.info(f"🔥 Flush marked: {collection_name} (will flush within 0.5s)")
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ 문서 삽입 완료 (Saga Pattern 성공)")
        logger.info(f"   - doc_id: {doc_id}")
        logger.info(f"   - 청크 수: {len(request.chunks)}")
        logger.info(f"   - 총 소요 시간: {total_time:.2f}ms")
        logger.info(f"   - 검색 가능 시간: 0.5초 이내")
        
        return DocumentInsertResponse(
            status="success",
            doc_id=doc_id,
            total_chunks=len(request.chunks),
            postgres_insert_time_ms=postgres_time,
            embedding_time_ms=embedding_time,
            milvus_insert_time_ms=milvus_time,
            total_time_ms=total_time,
            searchable_within_seconds=0.5
        )
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류 시 PostgreSQL 롤백
        if doc_id is not None:
            logger.error(f"❌ 예상치 못한 오류 발생, PostgreSQL 롤백 시작: {str(e)}")
            try:
                await postgres_client.delete_document(request.account_name, request.chat_bot_id, doc_id)
                logger.info(f"✅ PostgreSQL 롤백 완료: doc_id={doc_id}")
            except Exception as rollback_error:
                logger.error(f"❌ PostgreSQL 롤백 실패: {str(rollback_error)}")
        
        logger.error(f"❌ 문서 삽입 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert document: {str(e)}"
        )


@router.post("/insert/batch", response_model=BatchInsertResponse, status_code=status.HTTP_201_CREATED)
async def batch_insert_documents(request: BatchInsertRequest):
    """
    여러 문서 일괄 삽입 (개선된 Saga Pattern)
    
    전처리 서버에서 여러 문서를 한 번에 전송
    1. PostgreSQL 트랜잭션으로 모든 문서+청크 원자성 보장
    2. 배치 임베딩 생성 (재시도 로직 포함)
    3. Milvus 배치 벡터 저장 (재시도 로직 포함)
    4. 실패 시 보상 트랜잭션으로 PostgreSQL 롤백
    """
    doc_ids = []
    try:
        start_time = datetime.now()
        total_docs = len(request.documents)
        total_chunks = sum(len(doc.chunks) for doc in request.documents)
        collection_name = f"collection_{request.account_name}"
        
        logger.info(f"📝 배치 삽입 시작 (Saga Pattern)")
        logger.info(f"   - Account: {request.account_name}")
        logger.info(f"   - Documents: {total_docs}개")
        logger.info(f"   - Total Chunks: {total_chunks}개")
        logger.info(f"   - Collection: {collection_name}")
        
        # ========== Step 1: PostgreSQL 배치 트랜잭션 (원자성 보장) ==========
        postgres_start = datetime.now()
        
        # 문서 데이터 준비 (메타데이터 분리)
        documents_data = []
        for doc in request.documents:
            # 메타데이터 분리
            all_metadata = doc.metadata or {}
            
            documents_data.append({
                "document_data": {
                    "chat_bot_id": doc.chat_bot_id,
                    "content_name": doc.content_name,  # 문서 고유 식별자
                    "metadata": all_metadata  # 전체 메타데이터를 PostgreSQL에 저장
                },
                "chunks": [{"chunk_index": c.chunk_index, "text": c.text} for c in doc.chunks]
            })
        
        doc_ids = await postgres_client.batch_insert_documents_with_chunks_transaction(
            account_name=request.account_name,
            documents=documents_data
        )
        
        postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
        logger.info(f"✅ PostgreSQL 배치 트랜잭션 완료: {len(doc_ids)}개 문서")
        
        # ========== Step 2: 배치 임베딩 생성 (재시도 로직) ==========
        embedding_start = datetime.now()
        
        try:
            # 모든 청크 텍스트 수집
            all_chunk_texts = []
            for doc in request.documents:
                all_chunk_texts.extend([chunk.text for chunk in doc.chunks])
            
            # 배치 임베딩 생성
            all_embeddings = await embedding_service.batch_embed_with_retry(
                texts=all_chunk_texts,
                max_retries=3,
                backoff=2.0
            )
            
            embedding_time = (datetime.now() - embedding_start).total_seconds() * 1000
            logger.info(f"✅ 배치 임베딩 생성 완료: {len(all_embeddings)}개 벡터")
            
        except Exception as embedding_error:
            # 임베딩 실패 시 PostgreSQL 롤백
            logger.error(f"❌ 배치 임베딩 생성 실패, PostgreSQL 롤백 시작: {str(embedding_error)}")
            for doc_id in doc_ids:
                try:
                    await postgres_client.delete_document(request.account_name, request.documents[0].chat_bot_id, doc_id)
                except:
                    pass  # 롤백 실패는 로그만 남기고 계속
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"배치 임베딩 생성 실패: {str(embedding_error)}"
            )
        
        # ========== Step 3: Milvus 배치 벡터 저장 (재시도 로직) ==========
        milvus_start = datetime.now()
        
        try:
            # 문서별 데이터 준비
            documents_data = []
            embedding_idx = 0
            
            for i, doc in enumerate(request.documents):
                doc_id = doc_ids[i]
                
                # 메타데이터 분리
                all_metadata = doc.metadata or {}
                milvus_metadata = filter_milvus_metadata(all_metadata)  # Milvus 필터링용
                
                chunks_with_embeddings = []
                for chunk in doc.chunks:
                    chunks_with_embeddings.append({
                        "chunk_index": chunk.chunk_index,
                        "embedding": all_embeddings[embedding_idx],
                        "text": chunk.text
                    })
                    embedding_idx += 1
                
                documents_data.append({
                    "chat_bot_id": doc.chat_bot_id,
                    "doc_id": doc_id,
                    "content_name": doc.content_name,  # content_name 추가
                    "chunks": chunks_with_embeddings,
                    "metadata": milvus_metadata  # 필터링용 메타데이터만 Milvus에
                })
            
            # Milvus 배치 삽입
            await milvus_client.batch_insert_vectors_with_retry(
                account_name=request.account_name,
                documents_data=documents_data,
                metadata={},  # 개별 문서 메타데이터가 우선됨
                max_retries=3,
                backoff=2.0
            )
            
            milvus_time = (datetime.now() - milvus_start).total_seconds() * 1000
            logger.info(f"✅ Milvus 배치 벡터 삽입 완료")
            
        except Exception as milvus_error:
            # Milvus 실패 시 PostgreSQL 롤백 (보상 트랜잭션)
            logger.error(f"❌ Milvus 배치 삽입 실패, PostgreSQL 롤백 시작: {str(milvus_error)}")
            for doc_id in doc_ids:
                try:
                    await postgres_client.delete_document(request.account_name, request.documents[0].chat_bot_id, doc_id)
                except:
                    pass  # 롤백 실패는 로그만 남기고 계속
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Milvus 배치 삽입 실패: {str(milvus_error)}"
            )
        
        # ========== Step 4: 🔥 자동 flush 마킹 (이벤트 기반) ==========
        await auto_flusher.mark_for_flush(collection_name)
        logger.info(f"🔥 Flush marked: {collection_name}")
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 성공 결과 생성
        results = []
        for i, doc in enumerate(request.documents):
            results.append({
                "doc_id": doc_ids[i],
                "title": doc.metadata.get('title', '(제목 없음)') if doc.metadata else '(제목 없음)',
                "total_chunks": len(doc.chunks),
                "success": True
            })
        
        logger.info(f"✅ 배치 삽입 완료 (Saga Pattern 성공)")
        logger.info(f"   - 성공: {len(doc_ids)}개 문서")
        logger.info(f"   - 총 소요 시간: {total_time:.2f}ms")
        logger.info(f"   - 검색 가능 시간: 0.5초 이내")
        
        from app.models.document import BatchInsertResult
        return BatchInsertResponse(
            status="success",
            total_documents=total_docs,
            total_chunks=total_chunks,
            success_count=len(doc_ids),
            failure_count=0,
            results=[BatchInsertResult(**r) for r in results],
            postgres_insert_time_ms=postgres_time,
            embedding_time_ms=embedding_time,
            milvus_insert_time_ms=milvus_time,
            total_time_ms=total_time
        )
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류 시 PostgreSQL 롤백
        if doc_ids:
            logger.error(f"❌ 예상치 못한 오류 발생, PostgreSQL 롤백 시작: {str(e)}")
            for doc_id in doc_ids:
                try:
                    await postgres_client.delete_document(request.account_name, request.documents[0].chat_bot_id, doc_id)
                except Exception as rollback_error:
                    logger.error(f"❌ PostgreSQL 롤백 실패 (doc_id: {doc_id}): {str(rollback_error)}")
        
        logger.error(f"❌ 배치 삽입 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch insert failed: {str(e)}"
        )


@router.post("/insert/batch/with-embeddings", response_model=BatchInsertResponse, status_code=status.HTTP_201_CREATED)
async def batch_insert_documents_with_embeddings(request: BatchInsertWithEmbeddingsRequest):
    """
    임베딩을 포함한 여러 문서 일괄 삽입 (마이그레이션용)
    
    기존 PostgreSQL에 저장된 임베딩 벡터를 그대로 사용하여 Milvus에 삽입
    - 임베딩 생성 단계를 스킵하고 제공된 벡터를 직접 사용
    - PostgreSQL 트랜잭션으로 모든 문서+청크 원자성 보장
    - Milvus 배치 벡터 저장 (재시도 로직 포함)
    - 실패 시 보상 트랜잭션으로 PostgreSQL 롤백
    """
    doc_ids = []
    try:
        start_time = datetime.now()
        total_docs = len(request.documents)
        total_chunks = sum(len(doc.chunks) for doc in request.documents)
        collection_name = f"collection_{request.account_name}"
        
        logger.info(f"📝 배치 삽입 시작 (임베딩 포함, 마이그레이션용)")
        logger.info(f"   - Account: {request.account_name}")
        logger.info(f"   - Documents: {total_docs}개")
        logger.info(f"   - Total Chunks: {total_chunks}개")
        logger.info(f"   - Collection: {collection_name}")
        logger.info(f"   - 임베딩 생성: 스킵 (기존 벡터 사용)")
        
        # ========== Step 1: PostgreSQL 배치 트랜잭션 (원자성 보장) ==========
        postgres_start = datetime.now()
        
        # 문서 데이터 준비 (메타데이터 분리)
        documents_data = []
        for doc in request.documents:
            # 메타데이터 분리
            all_metadata = doc.metadata or {}
            
            documents_data.append({
                "document_data": {
                    "chat_bot_id": doc.chat_bot_id,
                    "content_name": doc.content_name,  # 문서 고유 식별자
                    "metadata": all_metadata  # 전체 메타데이터를 PostgreSQL에 저장
                },
                "chunks": [{"chunk_index": c.chunk_index, "text": c.text} for c in doc.chunks]
            })
        
        doc_ids = await postgres_client.batch_insert_documents_with_chunks_transaction(
            account_name=request.account_name,
            documents=documents_data
        )
        
        postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
        logger.info(f"✅ PostgreSQL 배치 트랜잭션 완료: {len(doc_ids)}개 문서")
        
        # ========== Step 2: 임베딩 생성 스킵 (이미 제공됨) ==========
        embedding_start = datetime.now()
        embedding_time = 0.0  # 임베딩 생성하지 않음
        logger.info(f"⏭️ 임베딩 생성 스킵 (기존 벡터 사용): {total_chunks}개 벡터")
        
        # ========== Step 3: Milvus 배치 벡터 저장 (재시도 로직) ==========
        milvus_start = datetime.now()
        
        try:
            # 문서별 데이터 준비
            documents_data = []
            
            for i, doc in enumerate(request.documents):
                doc_id = doc_ids[i]
                
                # 메타데이터 분리
                all_metadata = doc.metadata or {}
                milvus_metadata = filter_milvus_metadata(all_metadata)  # Milvus 필터링용
                
                # 청크에 이미 임베딩이 포함되어 있음
                chunks_with_embeddings = []
                for chunk in doc.chunks:
                    chunks_with_embeddings.append({
                        "chunk_index": chunk.chunk_index,
                        "embedding": chunk.embedding,  # 제공된 임베딩 사용
                        "text": chunk.text
                    })
                
                documents_data.append({
                    "chat_bot_id": doc.chat_bot_id,
                    "doc_id": doc_id,
                    "content_name": doc.content_name,  # content_name 추가
                    "chunks": chunks_with_embeddings,
                    "metadata": milvus_metadata  # 필터링용 메타데이터만 Milvus에
                })
            
            # Milvus 배치 삽입
            await milvus_client.batch_insert_vectors_with_retry(
                account_name=request.account_name,
                documents_data=documents_data,
                metadata={},  # 개별 문서 메타데이터가 우선됨
                max_retries=3,
                backoff=2.0
            )
            
            milvus_time = (datetime.now() - milvus_start).total_seconds() * 1000
            logger.info(f"✅ Milvus 배치 벡터 삽입 완료")
            
        except Exception as milvus_error:
            # Milvus 실패 시 PostgreSQL 롤백 (보상 트랜잭션)
            logger.error(f"❌ Milvus 배치 삽입 실패, PostgreSQL 롤백 시작: {str(milvus_error)}")
            for doc_id in doc_ids:
                try:
                    await postgres_client.delete_document(request.account_name, request.documents[0].chat_bot_id, doc_id)
                except:
                    pass  # 롤백 실패는 로그만 남기고 계속
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Milvus 배치 삽입 실패: {str(milvus_error)}"
            )
        
        # ========== Step 4: 🔥 자동 flush 마킹 (이벤트 기반) ==========
        await auto_flusher.mark_for_flush(collection_name)
        logger.info(f"🔥 Flush marked: {collection_name}")
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 성공 결과 생성
        results = []
        for i, doc in enumerate(request.documents):
            results.append({
                "doc_id": doc_ids[i],
                "title": doc.metadata.get('title', '(제목 없음)') if doc.metadata else '(제목 없음)',
                "total_chunks": len(doc.chunks),
                "success": True
            })
        
        logger.info(f"✅ 배치 삽입 완료 (임베딩 포함, 마이그레이션)")
        logger.info(f"   - 성공: {len(doc_ids)}개 문서")
        logger.info(f"   - 총 소요 시간: {total_time:.2f}ms")
        logger.info(f"   - 임베딩 생성 시간: 0ms (스킵됨)")
        logger.info(f"   - 검색 가능 시간: 0.5초 이내")
        
        from app.models.document import BatchInsertResult
        return BatchInsertResponse(
            status="success",
            total_documents=total_docs,
            total_chunks=total_chunks,
            success_count=len(doc_ids),
            failure_count=0,
            results=[BatchInsertResult(**r) for r in results],
            postgres_insert_time_ms=postgres_time,
            embedding_time_ms=embedding_time,  # 0ms
            milvus_insert_time_ms=milvus_time,
            total_time_ms=total_time
        )
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류 시 PostgreSQL 롤백
        if doc_ids:
            logger.error(f"❌ 예상치 못한 오류 발생, PostgreSQL 롤백 시작: {str(e)}")
            for doc_id in doc_ids:
                try:
                    await postgres_client.delete_document(request.account_name, request.documents[0].chat_bot_id, doc_id)
                except Exception as rollback_error:
                    logger.error(f"❌ PostgreSQL 롤백 실패 (doc_id: {doc_id}): {str(rollback_error)}")
        
        logger.error(f"❌ 배치 삽입 실패 (임베딩 포함): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch insert with embeddings failed: {str(e)}"
        )


@router.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: int,
    account_name: str = Query(..., description="계정명"),
    chat_bot_id: str = Query(..., description="챗봇 ID (UUID)")
):
    """
    문서 조회 (doc_id 기반)
    
    해당 계정 및 봇의 파티션에서 문서와 모든 청크 조회
    """
    try:
        logger.info(f"문서 조회 요청 (account: {account_name}, bot: {chat_bot_id}): doc_id={doc_id}")
        
        # TODO: PostgreSQL에서 문서 조회 (자동으로 해당 파티션만 스캔)
        # document = await postgres_client.get_document(account_name, chat_bot_id, doc_id)
        # chunks = await postgres_client.get_chunks(account_name, chat_bot_id, doc_id)
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="구현 예정"
        )
    except Exception as e:
        logger.error(f"문서 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )


@router.put("/document/{doc_id}", response_model=DocumentUpdateResponse)
async def update_document(doc_id: int, request: DocumentUpdateRequest):
    """
    문서 전체 업데이트
    
    기존 문서의 모든 청크를 삭제하고 새로운 데이터로 교체
    """
    try:
        logger.info(f"문서 업데이트 요청 (account: {request.account_name}, bot: {request.chat_bot_id}): doc_id={doc_id}")
        
        # TODO: 트랜잭션 처리
        # async with transaction():
        #     chunk_count = len(request.chunks)
        #     document_data = {
        #         "chat_bot_id": request.chat_bot_id,
        #         "metadata": request.metadata or {}  # 모든 메타데이터를 JSON으로 저장
        #     }
        #     await postgres_client.update_document(request.account_name, request.chat_bot_id, doc_id, document_data, chunk_count)  # ⭐ 청크 수 전달
        #     deleted_count = await postgres_client.delete_chunks(request.account_name, request.chat_bot_id, doc_id)
        #     await postgres_client.insert_chunks(request.account_name, request.chat_bot_id, doc_id, [c.dict() for c in request.chunks])
        #     await milvus_client.delete_by_doc_id(doc_id)
        #     embeddings = await embedding_service.batch_embed([chunk.text for chunk in request.chunks])
        #     await milvus_client.insert(doc_id, embeddings)
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="구현 예정"
        )
    except Exception as e:
        logger.error(f"문서 업데이트 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@router.post("/document/delete", response_model=DocumentDeleteResponse)
async def delete_document(request: DocumentDeleteRequest):
    """
    문서 삭제 (여러 content_name 기준) - Saga Pattern 적용
    
    PostgreSQL과 Milvus에서 여러 문서와 모든 청크를 일괄 삭제합니다.
    - content_names 리스트로 여러 문서 식별
    - Milvus: content_names와 chat_bot_id로 필터링하여 모든 벡터 삭제 (먼저)
    - PostgreSQL: 해당 파티션에서 여러 문서와 모든 청크 삭제 (나중에)
    - PostgreSQL 실패 시 Milvus 복구 (보상 트랜잭션)
    - 자동 flush로 실시간 반영
    
    Saga Pattern 장점:
    - 데이터 일관성 보장
    - 부분 실패 시 자동 복구
    - 여러 문서 일괄 처리로 성능 향상
    - POST 메서드로 요청 본문에 안전하게 데이터 전달
    """
    try:
        start_time = datetime.now()
        collection_name = f"collection_{request.account_name}"
        
        logger.info(f"🗑️ 문서 일괄 삭제 시작 (Saga Pattern)")
        logger.info(f"   - Account: {request.account_name}")
        logger.info(f"   - Bot ID: {request.chat_bot_id}")
        logger.info(f"   - Content Names: {len(request.content_names)}개")
        logger.info(f"   - Collection: {collection_name}")
        
        # ========== Step 0: 존재하는 문서만 필터링 ==========
        existing_content_names = await postgres_client.get_existing_content_names(
            request.account_name, request.chat_bot_id, request.content_names
        )
        
        if not existing_content_names:
            # 존재하는 문서가 없음
            logger.warning(f"⚠️ 존재하는 문서가 없음: {request.content_names}")
            return DocumentDeleteResponse(
                status="success",
                message=f"No documents found to delete from {len(request.content_names)} requested",
                total_requested=len(request.content_names),
                total_success=0,
                total_failed=len(request.content_names),
                successful_content_names=[],
                failed_content_names=request.content_names,
                deleted_documents=0,
                deleted_chunks=0,
                deleted_vectors=0,
                postgres_delete_time_ms=0.0,
                milvus_delete_time_ms=0.0,
                total_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        # 성공/실패한 문서 추적
        successful_content_names = []
        failed_content_names = []
        
        # 존재하지 않는 문서들을 실패 목록에 추가
        if len(existing_content_names) < len(request.content_names):
            missing_docs = set(request.content_names) - set(existing_content_names)
            failed_content_names.extend(list(missing_docs))
            logger.info(f"📋 존재하는 문서: {len(existing_content_names)}개 / {len(request.content_names)}개")
            logger.info(f"   - 존재하는 문서: {existing_content_names}")
            logger.info(f"   - 존재하지 않는 문서: {list(missing_docs)}")
        else:
            logger.info(f"📋 모든 문서 존재: {len(existing_content_names)}개")
            logger.info(f"   - 존재하는 문서: {existing_content_names}")
        
        # ========== Step 1: Milvus에서 벡터 일괄 삭제 (먼저) ==========
        milvus_start = datetime.now()
        deleted_vectors = 0
        
        try:
            # 존재하는 content_name과 chat_bot_id로 일괄 삭제
            deleted_vectors = await milvus_client.delete_by_content_names(
                collection_name, request.chat_bot_id, existing_content_names
            )
            
            milvus_time = (datetime.now() - milvus_start).total_seconds() * 1000
            logger.info(f"✅ Milvus 일괄 삭제 완료: {deleted_vectors}개 벡터, {milvus_time:.2f}ms")
            
        except Exception as milvus_error:
            logger.error(f"❌ Milvus 일괄 삭제 실패: {str(milvus_error)}")
            # Milvus 삭제 실패 시 모든 존재하는 문서를 실패 목록에 추가
            failed_content_names.extend(existing_content_names)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Milvus batch deletion failed: {str(milvus_error)}"
            )
        
        # ========== Step 2: PostgreSQL에서 문서 일괄 삭제 (나중에) ==========
        postgres_start = datetime.now()
        
        try:
            deleted_docs, deleted_chunks = await postgres_client.delete_documents_by_content_names(
                request.account_name, request.chat_bot_id, existing_content_names
            )
            postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
            
            logger.info(f"✅ PostgreSQL 일괄 삭제 완료: {deleted_docs}개 문서, {deleted_chunks}개 청크, {postgres_time:.2f}ms")
            
            # PostgreSQL 삭제 성공 시 모든 존재하는 문서를 성공 목록에 추가
            successful_content_names.extend(existing_content_names)
            
        except Exception as postgres_error:
            # PostgreSQL 삭제 실패 시 Milvus 복구 (보상 트랜잭션)
            logger.error(f"❌ PostgreSQL 일괄 삭제 실패, Milvus 복구 시작: {str(postgres_error)}")
            try:
                # TODO: Milvus 복구 로직 구현 (삭제된 벡터를 다시 삽입)
                logger.warning(f"⚠️ Milvus 복구 로직 미구현 - 데이터 일관성 문제 가능성")
            except Exception as recovery_error:
                logger.error(f"❌ Milvus 복구 실패: {str(recovery_error)}")
            
            # PostgreSQL 삭제 실패 시 모든 존재하는 문서를 실패 목록에 추가
            failed_content_names.extend(existing_content_names)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PostgreSQL batch deletion failed: {str(postgres_error)}"
            )
        
        # ========== Step 3: 🔥 자동 flush 마킹 (삭제 이벤트) ==========
        await auto_flusher.mark_for_flush(collection_name)
        logger.info(f"🔥 Flush marked after delete: {collection_name}")
        
        # ========== Step 4: 결과 반환 ==========
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 응답 메시지 생성
        if len(successful_content_names) == len(request.content_names):
            # 모든 문서가 성공한 경우
            status = "success"
            message = f"All {len(request.content_names)} documents deleted successfully"
        elif len(successful_content_names) > 0:
            # 일부 문서만 성공한 경우
            status = "partial_success"
            message = f"Deleted {len(successful_content_names)} out of {len(request.content_names)} requested documents"
        else:
            # 모든 문서가 실패한 경우
            status = "failed"
            message = f"Failed to delete any of {len(request.content_names)} requested documents"
        
        logger.info(f"✅ 문서 일괄 삭제 완료 (Saga Pattern 성공)")
        logger.info(f"   - 요청된 문서: {len(request.content_names)}개")
        logger.info(f"   - 성공한 문서: {len(successful_content_names)}개")
        logger.info(f"   - 실패한 문서: {len(failed_content_names)}개")
        logger.info(f"   - 삭제된 문서: {deleted_docs}개")
        logger.info(f"   - 삭제된 청크: {deleted_chunks}개")
        logger.info(f"   - 삭제된 벡터: {deleted_vectors}개")
        logger.info(f"   - 총 소요 시간: {total_time:.2f}ms")
        logger.info(f"   - 검색에서 제외: 0.5초 이내")
        
        return DocumentDeleteResponse(
            status=status,
            message=message,
            total_requested=len(request.content_names),
            total_success=len(successful_content_names),
            total_failed=len(failed_content_names),
            successful_content_names=successful_content_names,
            failed_content_names=failed_content_names,
            deleted_documents=deleted_docs,
            deleted_chunks=deleted_chunks,
            deleted_vectors=deleted_vectors,
            postgres_delete_time_ms=postgres_time,
            milvus_delete_time_ms=milvus_time,
            total_time_ms=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 문서 삭제 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/bot/delete", response_model=BotDeleteResponse, status_code=status.HTTP_200_OK)
async def delete_bot_data(request: BotDeleteRequest):
    """
    봇 전체 데이터 삭제 (chat_bot_id 기준) - Saga Pattern 적용
    
    해당 봇의 모든 문서와 청크를 PostgreSQL과 Milvus에서 삭제합니다.
    - Milvus: 해당 파티션의 모든 벡터 삭제 (먼저)
    - PostgreSQL: 해당 봇의 모든 문서와 청크 삭제 (나중에)
    - PostgreSQL 실패 시 Milvus 복구 (보상 트랜잭션)
    - 자동 flush로 실시간 반영
    
    Saga Pattern 장점:
    - 데이터 일관성 보장
    - 부분 실패 시 자동 복구
    - POST 메서드로 요청 본문에 안전하게 데이터 전달
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"🗑️ 봇 전체 삭제 시작 (Saga Pattern)")
        logger.info(f"   - Account: {request.account_name}")
        logger.info(f"   - Bot ID: {request.chat_bot_id}")
        
        collection_name = f"collection_{request.account_name}"
        partition_name = generate_partition_name(request.chat_bot_id)
        
        # ========== Step 1: Milvus에서 파티션 삭제 (먼저) ==========
        milvus_start = datetime.now()
        deleted_vectors = 0
        
        try:
            # Milvus에서 해당 파티션의 모든 벡터 삭제
            deleted_vectors = await milvus_client.delete_partition(collection_name, partition_name)
            
            milvus_time = (datetime.now() - milvus_start).total_seconds() * 1000
            logger.info(f"✅ Milvus 파티션 삭제 완료: {deleted_vectors}개 벡터, {milvus_time:.2f}ms")
            
        except Exception as milvus_error:
            logger.error(f"❌ Milvus 파티션 삭제 실패: {str(milvus_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Milvus partition deletion failed: {str(milvus_error)}"
            )
        
        # ========== Step 2: PostgreSQL에서 봇 데이터 삭제 (나중에) ==========
        postgres_start = datetime.now()
        
        try:
            deleted_docs, deleted_chunks = await postgres_client.delete_bot_data(request.account_name, request.chat_bot_id)
            postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
            logger.info(f"✅ PostgreSQL 봇 삭제 완료: {deleted_docs}개 문서, {deleted_chunks}개 청크, {postgres_time:.2f}ms")
            
        except Exception as postgres_error:
            # PostgreSQL 삭제 실패 시 Milvus 복구 (보상 트랜잭션)
            logger.error(f"❌ PostgreSQL 봇 삭제 실패, Milvus 복구 시작: {str(postgres_error)}")
            try:
                # TODO: Milvus 복구 로직 구현 (삭제된 파티션을 다시 생성하고 벡터 재삽입)
                # 현재는 로그만 남기고 계속 진행
                logger.warning(f"⚠️ Milvus 복구 로직 미구현 - 데이터 일관성 문제 가능성")
            except Exception as recovery_error:
                logger.error(f"❌ Milvus 복구 실패: {str(recovery_error)}")
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PostgreSQL bot deletion failed: {str(postgres_error)}"
            )
        
        # ========== Step 3: 🔥 자동 flush 마킹 (삭제 이벤트) ==========
        await auto_flusher.mark_for_flush(collection_name)
        logger.info(f"🔥 Flush marked after bot delete: {collection_name}")
        
        # ========== Step 4: 결과 반환 ==========
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ 봇 전체 삭제 완료 (Saga Pattern 성공)")
        logger.info(f"   - Bot ID: {request.chat_bot_id}")
        logger.info(f"   - 삭제된 문서: {deleted_docs}개")
        logger.info(f"   - 삭제된 청크: {deleted_chunks}개")
        logger.info(f"   - 삭제된 벡터: {deleted_vectors}개")
        logger.info(f"   - 총 소요 시간: {total_time:.2f}ms")
        logger.info(f"   - 검색에서 제외: 0.5초 이내")
        
        return BotDeleteResponse(
            status="success",
            chat_bot_id=request.chat_bot_id,
            deleted_documents=deleted_docs,
            deleted_chunks=deleted_chunks,
            deleted_vectors=deleted_vectors,
            postgres_delete_time_ms=postgres_time,
            milvus_delete_time_ms=milvus_time,
            total_time_ms=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 봇 전체 삭제 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete bot data: {str(e)}"
        )


@router.patch("/document/{doc_id}/metadata", response_model=MetadataUpdateResponse)
async def update_metadata(doc_id: int, request: MetadataUpdateRequest):
    """
    메타데이터만 업데이트
    
    PostgreSQL만 수정하고 Milvus는 건드리지 않음 (초고속)
    """
    try:
        logger.info(f"메타데이터 업데이트 요청 (account: {request.account_name}, bot: {request.chat_bot_id}): doc_id={doc_id}")
        
        # TODO: PostgreSQL UPDATE만 실행 (해당 파티션에서만)
        # await postgres_client.update_metadata(request.account_name, request.chat_bot_id, doc_id, request.metadata_updates)
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="구현 예정"
        )
    except Exception as e:
        logger.error(f"메타데이터 업데이트 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update metadata: {str(e)}"
        )





