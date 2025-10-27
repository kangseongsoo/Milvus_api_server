"""
검색 API
"""
from fastapi import APIRouter, HTTPException, status
from app.models.search import SearchRequest, SearchResponse
from app.utils.logger import setup_logger
from app.core.redis_partition_manager import redis_partition_manager
from app.core.embedding import embedding_service
from app.core.milvus_client import milvus_client
from app.core.postgres_client import postgres_client
from pymilvus import Collection
from datetime import datetime

logger = setup_logger(__name__)
router = APIRouter()


def generate_partition_name(chat_bot_id: str) -> str:
    """chat_bot_id로 파티션명 자동 생성"""
    return f"bot_{chat_bot_id.replace('-', '')}"


@router.post("/query", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    유사도 검색 (벡터 검색)
    
    1. 파티션 로드 (검색을 위해 메모리에 올림)
    2. 쿼리 텍스트를 임베딩 처리
    3. Milvus에서 벡터 검색 → doc_id 리스트 획득
    4. 계정별 PostgreSQL에서 메타데이터 조회
    5. 결과 통합하여 반환
    """
    try:
        collection_name = f"collection_{request.account_name}"
        partition_name = generate_partition_name(request.chat_bot_id)
        
        logger.info(f"검색 요청 (account: {request.account_name}, bot: {request.chat_bot_id}): '{request.query_text}', limit={request.limit}")
        
        # ========== Step 0: 파티션 로드 (검색 필수!) ==========
        logger.info(f"파티션 로드 확인: {partition_name}")
        await redis_partition_manager.ensure_partition_loaded(
            collection_name=collection_name,
            partition_name=partition_name
        )
        logger.info(f"파티션 로드 완료 (검색 가능)")
        
        # 파티션 접근 시간 업데이트 (TTL 정리용)
        await redis_partition_manager.update_partition_access_time(collection_name, partition_name)
        
        # ========== Step 1: 쿼리 임베딩 생성 ==========
        embedding_start = datetime.now()
        
        try:
            query_vector = await embedding_service.embed(request.query_text)
            embedding_time = (datetime.now() - embedding_start).total_seconds() * 1000
            logger.info(f"쿼리 임베딩 완료: {embedding_time:.2f}ms")
        except Exception as embedding_error:
            logger.error(f"쿼리 임베딩 실패: {str(embedding_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query embedding failed: {str(embedding_error)}"
            )
        
        # ========== Step 2: Milvus 벡터 검색 ==========
        search_start = datetime.now()
        
        try:
            collection = Collection(name=collection_name)
            
            # 검색 파라미터
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 필터 표현식 구성
            base_expr = f"chat_bot_id == '{request.chat_bot_id}'"
            if request.filter_expr:
                expr = f"{base_expr} and {request.filter_expr}"
            else:
                expr = base_expr
            
            logger.info(f"Milvus 검색 시작: expr='{expr}'")
            
            # 벡터 검색 실행
            search_results = collection.search(
                data=[query_vector],
                anns_field="embedding_dense",
                param=search_params,
                limit=request.limit,
                partition_names=[partition_name],
                expr=expr,
                output_fields=["doc_id", "chunk_index"]  # metadata 제거
            )
            
            search_time = (datetime.now() - search_start).total_seconds() * 1000
            logger.info(f"Milvus 검색 완료: {search_time:.2f}ms")
            
            # 검색 결과 처리
            if not search_results or not search_results[0]:
                logger.info("검색 결과 없음")
                return SearchResponse(
                    status="success",
                    query=request.query_text,
                    total_results=0,
                    results=[],
                    search_time_ms=search_time,
                    embedding_time_ms=embedding_time
                )
            
            # 결과 파싱
            hits = search_results[0]
            doc_ids = []
            scores = []
            chunk_indices = []
            
            for hit in hits:
                scores.append(hit.score)
                # entity에서 필드 추출 (딕셔너리 스타일)
                doc_ids.append(hit.entity.get("doc_id"))
                chunk_indices.append(hit.entity.get("chunk_index"))
            
            logger.info(f"검색 결과: {len(hits)}개 벡터, {len(set(doc_ids))}개 문서")
            
        except Exception as search_error:
            logger.error(f"Milvus 검색 실패: {str(search_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vector search failed: {str(search_error)}"
            )
        
        # ========== Step 3: PostgreSQL 메타데이터 조회 ==========
        postgres_start = datetime.now()
        
        try:
            # 고유한 doc_id만 조회
            unique_doc_ids = list(set(doc_ids))
            unique_chunk_indices = list(set(chunk_indices))
            documents = await postgres_client.get_documents_with_chunks_by_ids(
                account_name=request.account_name,
                chat_bot_id=request.chat_bot_id,
                doc_ids=unique_doc_ids,
                chunk_indices=unique_chunk_indices
            )
            
            postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
            logger.info(f"PostgreSQL 메타데이터 조회 완료: {postgres_time:.2f}ms")
            
            # 디버깅: documents 구조 확인
            logger.info(f"PostgreSQL 결과 타입: {type(documents)}, 개수: {len(documents) if documents else 0}")
            if documents and len(documents) > 0:
                logger.info(f"첫 번째 문서 타입: {type(documents[0])}")
                if isinstance(documents[0], dict):
                    logger.info(f"첫 번째 문서 키: {list(documents[0].keys())}")
                else:
                    logger.info(f"첫 번째 문서 내용: {str(documents[0])[:100]}...")
            
        except Exception as postgres_error:
            logger.error(f"PostgreSQL 조회 실패: {str(postgres_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Metadata retrieval failed: {str(postgres_error)}"
            )
        
        # ========== Step 4: 결과 통합 ==========
        try:
            # 문서 메타데이터를 딕셔너리로 변환 (documents는 딕셔너리 리스트)
            doc_metadata = {doc["doc_id"]: doc for doc in documents}
            
            # 검색 결과 구성
            results = []
            for i, (doc_id, score, chunk_index) in enumerate(zip(doc_ids, scores, chunk_indices)):
                doc_info = doc_metadata.get(doc_id)
                
                if doc_info:
                    # metadata가 문자열인 경우 JSON 파싱
                    metadata = doc_info.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # 청크 텍스트 추출 (document_chunks 테이블에서 직접 가져오기)
                    chunks = doc_info.get("chunks", {})
                    chunk_data = chunks.get(chunk_index, {})
                    chunk_text = chunk_data.get("chunk_text", "")
                    
                    results.append({
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "score": float(score),
                        "chunk_text": chunk_text,
                        "document": {
                            "title": metadata.get("title", "(제목 없음)"),
                            "content_name": doc_info.get("content_name", ""),
                            "metadata": metadata
                        }
                    })
            
            # 시간 계산
            total_time = (datetime.now() - search_start).total_seconds() * 1000
            vector_search_time = search_time + embedding_time
            
            logger.info(f"검색 완료: {len(results)}개 결과")
            
            return SearchResponse(
                status="success",
                vector_search_time_ms=vector_search_time,
                postgres_query_time_ms=postgres_time,
                total_time_ms=total_time,
                results=results
            )
            
        except Exception as merge_error:
            logger.error(f"❌ 결과 통합 실패: {str(merge_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Result merging failed: {str(merge_error)}"
            )
    except Exception as e:
        logger.error(f"검색 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

