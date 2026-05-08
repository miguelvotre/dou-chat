"""
MotherDuck / DuckDB VSS — schema creation and indexing operations.

Tables:
  dou.atos          - full act metadata (no embedding)
  dou.chunks_e5     - chunks with multilingual-e5-large (1024 dims)
  dou.chunks_labse  - chunks with LaBSE                 (768 dims)
  dou.chunks_bge    - chunks with BAAI/bge-m3            (1024 dims)
"""

import os

import duckdb

BASE_DDL = """
INSTALL vss;
LOAD vss;

CREATE SCHEMA IF NOT EXISTS dou;

CREATE TABLE IF NOT EXISTS dou.atos (
    id            VARCHAR PRIMARY KEY,
    identificador VARCHAR,
    titulo        VARCHAR,
    subtitulo     VARCHAR,
    texto         VARCHAR,
    orgao         VARCHAR,
    secao         VARCHAR,
    data          DATE,
    tipo_ato      VARCHAR,
    assina        VARCHAR,
    cargo         VARCHAR,
    fonte         VARCHAR DEFAULT 'federal',
    uf            VARCHAR,
    municipio     VARCHAR,
    url_original  VARCHAR,
    criado_em     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CHUNKS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    id          VARCHAR PRIMARY KEY,
    ato_id      VARCHAR,
    texto       VARCHAR NOT NULL,
    embedding   FLOAT[{dim}],
    orgao       VARCHAR,
    secao       VARCHAR,
    data        DATE,
    tipo_ato    VARCHAR,
    fonte       VARCHAR DEFAULT 'federal',
    uf          VARCHAR,
    municipio   VARCHAR,
    chunk_index INTEGER
);
CREATE INDEX IF NOT EXISTS {idx}_ato_id ON {table} (ato_id);
CREATE INDEX IF NOT EXISTS {idx}_data   ON {table} (data);
"""

HNSW_DDL = """
CREATE INDEX IF NOT EXISTS {idx}_hnsw ON {table}
USING HNSW (embedding) WITH (metric = 'cosine');
"""

# Chunk tables and their embedding dimensions
CHUNK_TABLES = {
    "e5":    ("dou.chunks_e5",    1024),
    "labse": ("dou.chunks_labse", 768),
    "bge":   ("dou.chunks_bge",   1024),
}


def get_connection() -> duckdb.DuckDBPyConnection:
    """Returns a MotherDuck connection (production) or local DuckDB (dev)."""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        conn = duckdb.connect(f"md:?motherduck_token={token}")
        conn.execute("CREATE DATABASE IF NOT EXISTS diario_oficial")
        conn.execute("USE diario_oficial")
        return conn
    import pathlib
    pathlib.Path("data").mkdir(exist_ok=True)
    return duckdb.connect("data/diario_oficial.db")


def init_schema(conn: duckdb.DuckDBPyConnection, embed_keys: list[str] | None = None) -> None:
    """
    Creates the base schema (dou.atos) and chunk tables for the given embedding models.

    embed_keys: subset of ["e5", "labse", "bge"]; None = all
    """
    for stmt in BASE_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    keys = embed_keys if embed_keys is not None else list(CHUNK_TABLES.keys())
    for key in keys:
        table, dim = CHUNK_TABLES[key]
        idx = table.replace(".", "_").replace("dou_", "")
        for stmt in CHUNKS_TABLE_DDL.format(table=table, dim=dim, idx=idx).strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        try:
            conn.execute(HNSW_DDL.format(idx=idx, table=table))
        except Exception:
            pass


def drop_chunks_table(conn: duckdb.DuckDBPyConnection, key: str) -> None:
    """Drops the chunk table for a specific embedding model."""
    table, _ = CHUNK_TABLES[key]
    conn.execute(f"DROP TABLE IF EXISTS {table}")


def upsert_ato(conn: duckdb.DuckDBPyConnection, ato) -> None:
    """Inserts an act, ignoring duplicates (deduplication by id)."""
    conn.execute(
        """
        INSERT OR IGNORE INTO dou.atos
        (id, identificador, titulo, subtitulo, texto, orgao, secao, data,
         tipo_ato, assina, cargo, fonte, uf, municipio, url_original)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ato.id, ato.identificador, ato.titulo, ato.subtitulo, ato.texto,
            ato.orgao, ato.secao, ato.data, ato.tipo_ato, ato.assina, ato.cargo,
            ato.fonte, ato.uf, ato.municipio, ato.url_original,
        ],
    )


def upsert_chunks(
    conn: duckdb.DuckDBPyConnection,
    ato_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    meta: dict,
    embed_key: str = "e5",
) -> None:
    """Inserts chunks with embeddings into the specified model's table."""
    table, _ = CHUNK_TABLES[embed_key]
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{ato_id}_{embed_key}_c{i}"
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {table}
            (id, ato_id, texto, embedding, orgao, secao, data, tipo_ato,
             fonte, uf, municipio, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                chunk_id, ato_id, text, emb,
                meta.get("orgao"), meta.get("secao"), meta.get("data"),
                meta.get("tipo_ato"), meta.get("fonte", "federal"),
                meta.get("uf"), meta.get("municipio"), i,
            ],
        )


def similarity_search(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    top_k: int = 10,
    filters: dict | None = None,
    chunks_table: str = "dou.chunks_e5",
) -> list[dict]:
    """
    Cosine similarity search with optional metadata filters.

    chunks_table: dou.chunks_e5 | dou.chunks_labse | dou.chunks_bge
    filters: secao, orgao, tipo_ato, fonte, data_inicio, data_fim
    """
    dim = len(query_embedding)
    where_clauses: list[str] = []
    params: list = []

    if filters:
        if "secao" in filters:
            where_clauses.append("c.secao = ?")
            params.append(filters["secao"])
        if "orgao" in filters:
            where_clauses.append("c.orgao ILIKE ?")
            params.append(f"%{filters['orgao']}%")
        if "tipo_ato" in filters:
            where_clauses.append("c.tipo_ato = ?")
            params.append(filters["tipo_ato"])
        if "fonte" in filters:
            where_clauses.append("c.fonte = ?")
            params.append(filters["fonte"])
        if "data_inicio" in filters:
            where_clauses.append("c.data >= ?")
            params.append(filters["data_inicio"])
        if "data_fim" in filters:
            where_clauses.append("c.data <= ?")
            params.append(filters["data_fim"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.ato_id,
            c.texto AS chunk_texto,
            c.orgao,
            c.secao,
            c.data,
            c.tipo_ato,
            a.titulo,
            a.subtitulo,
            a.texto AS ato_texto,
            a.url_original,
            array_cosine_similarity(c.embedding, ?::FLOAT[{dim}]) AS score
        FROM {chunks_table} c
        JOIN dou.atos a ON a.id = c.ato_id
        {where_sql}
        ORDER BY score DESC
        LIMIT ?
    """

    params_final = [query_embedding] + params + [top_k]
    rows = conn.execute(sql, params_final).fetchall()
    cols = [
        "chunk_id", "ato_id", "chunk_texto", "orgao", "secao", "data",
        "tipo_ato", "titulo", "subtitulo", "ato_texto", "url_original", "score",
    ]
    return [dict(zip(cols, row)) for row in rows]
