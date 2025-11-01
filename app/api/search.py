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
        partition_load_start = datetime.now()
        logger.info(f"파티션 로드 확인: {partition_name}")
        
        # 검색 시 컬렉션 로드 상태 확인하여 적절히 로드
        from pymilvus import utility, Collection
        collection_obj = Collection(name=collection_name)
        
        try:
            collection_load_state = utility.load_state(collection_name)
            collection_is_loaded = (collection_load_state == utility.LoadState.Loaded)
        except Exception:
            collection_is_loaded = False
        
        if collection_is_loaded:
            # 컬렉션이 이미 로드되어 있으면 기존 방식대로 파티션만 로드
            await redis_partition_manager.ensure_partition_loaded(
                collection_name=collection_name,
                partition_name=partition_name
            )
        else:
            # 컬렉션이 언로드되어 있으면 컬렉션을 로드하면서 특정 파티션만 선택적으로 로드
            logger.info(f"Collection {collection_name} is not loaded, loading collection with partition: {partition_name}")
            collection_obj.load(partition_names=[partition_name])
            logger.info(f"✅ Collection {collection_name} loaded with partition {partition_name}")
            
            # Redis에 상태 저장 및 접근 시간 업데이트
            from app.core.redis_client import redis_client
            partition_state_manager = redis_partition_manager._partition_state_manager
            await partition_state_manager.set_partition_loaded(collection_name, partition_name)
            await partition_state_manager.update_access_time(collection_name, partition_name)
        
        partition_load_time = (datetime.now() - partition_load_start).total_seconds() * 1000
        logger.info(f"✅ 파티션 로드 완료: {partition_load_time:.2f}ms (검색 가능)")
        
        # ========== Step 0.5: 쿼리 검증 ==========
        # 빈 문자열 또는 공백만 있는 경우 에러
        if not request.query_text or not request.query_text.strip():
            logger.warning(f"빈 검색 쿼리 요청: '{request.query_text}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text is empty or contains only whitespace"
            )
        
        # ========== Step 1: 쿼리 임베딩 생성 ==========
        embedding_start = datetime.now()
        
        try:
            query_vector = await embedding_service.embed(request.query_text.strip())
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
            base_expr = f'chat_bot_id == "{request.chat_bot_id}"'
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
                total_time = (datetime.now() - partition_load_start).total_seconds() * 1000
                return SearchResponse(
                    status="success",
                    partition_load_time_ms=partition_load_time,
                    vector_search_time_ms=search_time + embedding_time,
                    postgres_query_time_ms=0.0,
                    total_time_ms=total_time,
                    results=[]
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
            logger.info(f"PostgreSQL 문서 조회 시작: {len(unique_doc_ids)}개 doc_id, {len(unique_chunk_indices)}개 chunk_index")
            
            documents = await postgres_client.get_documents_with_chunks_by_ids(
                account_name=request.account_name,
                chat_bot_id=request.chat_bot_id,
                doc_ids=unique_doc_ids,
                chunk_indices=unique_chunk_indices
            )
            
            postgres_time = (datetime.now() - postgres_start).total_seconds() * 1000
            logger.info(f"PostgreSQL 메타데이터 조회 완료: {postgres_time:.2f}ms, {len(documents)}개 문서 발견")
            
            # PostgreSQL에서 찾지 못한 doc_id 로깅
            found_doc_ids = {doc["doc_id"] for doc in documents}
            missing_doc_ids = set(unique_doc_ids) - found_doc_ids
            if missing_doc_ids:
                logger.warning(f"⚠️ PostgreSQL에서 문서를 찾지 못한 doc_id: {missing_doc_ids} (Milvus에는 있지만 PostgreSQL에는 없음 - 데이터 불일치 가능성)")
            
            # 디버깅: documents 구조 확인
            #logger.info(f"PostgreSQL 결과 타입: {type(documents)}, 개수: {len(documents) if documents else 0}")
            #if documents and len(documents) > 0:
            #    logger.info(f"첫 번째 문서 타입: {type(documents[0])}")
            #    if isinstance(documents[0], dict):
            #        logger.info(f"첫 번째 문서 키: {list(documents[0].keys())}")
            #    else:
            #        logger.info(f"첫 번째 문서 내용: {str(documents[0])[:100]}...")
            
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
            
            # 검색 결과 구성 (PostgreSQL에 존재하는 문서만 포함)
            results = []
            skipped_count = 0  # PostgreSQL에 없는 문서 수
            
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
                else:
                    # PostgreSQL에 문서가 없는 경우 제외
                    skipped_count += 1
                    logger.debug(f"PostgreSQL에서 문서를 찾지 못함: doc_id={doc_id}, chunk_index={chunk_index} (검색 결과에서 제외)")
            
            if skipped_count > 0:
                logger.warning(f"⚠️ 검색 결과에서 {skipped_count}개 문서 제외됨 (PostgreSQL에 존재하지 않음 - Milvus와 데이터 불일치)")
            
            # 시간 계산
            total_time = (datetime.now() - partition_load_start).total_seconds() * 1000
            vector_search_time = search_time + embedding_time
            
            logger.info(f"검색 완료: {len(results)}개 결과 반환 (Milvus에서 {len(hits)}개 발견, PostgreSQL에서 {len(documents)}개 존재, {skipped_count}개 제외)")
            logger.info(f"⏱️  시간 측정 - 파티션 로드: {partition_load_time:.2f}ms, 벡터 검색: {vector_search_time:.2f}ms, PostgreSQL: {postgres_time:.2f}ms, 총: {total_time:.2f}ms")
            
            return SearchResponse(
                status="success",
                partition_load_time_ms=partition_load_time,
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

