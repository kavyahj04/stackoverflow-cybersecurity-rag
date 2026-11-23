"""One way to connect to Postgres.

Every connection gets pgvector's types registered and the HNSW search
setting applied, so no caller can forget either. (For the API this becomes
a connection pool.)
"""
import psycopg
from pgvector.psycopg import register_vector

from rag.config import get_settings


def connect(autocommit: bool = True) -> psycopg.Connection:
    settings = get_settings()
    conn = psycopg.connect(settings.database_url, autocommit=autocommit)
    register_vector(conn)
    # SET can't take a query parameter; int() guarantees only a number goes into the SQL
    conn.execute(f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}")
    return conn
