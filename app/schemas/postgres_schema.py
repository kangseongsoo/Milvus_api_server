"""
PostgreSQL 테이블 스키마 정의
"""

# PostgreSQL 스키마 (migrations/init.sql에서 실제 생성)

DOCUMENTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id BIGSERIAL PRIMARY KEY,
    chat_bot_id VARCHAR(100) NOT NULL,
    content_name VARCHAR(500) NOT NULL,
    chunk_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    UNIQUE(chat_bot_id, content_name)
);
"""

DOCUMENT_CHUNKS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    doc_id BIGINT REFERENCES documents(doc_id) ON DELETE CASCADE,
    chat_bot_id VARCHAR(100) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    UNIQUE(doc_id, chunk_index)
);
"""

INDEXES_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_documents_chat_bot_id ON documents(chat_bot_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_name ON documents(content_name);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
"""

def get_init_sql() -> str:
    """
    초기 테이블 생성 SQL 반환
    
    Returns:
        전체 초기화 SQL
    """
    return f"""
{DOCUMENTS_TABLE_SCHEMA}

{DOCUMENT_CHUNKS_TABLE_SCHEMA}

{INDEXES_SCHEMA}
"""

