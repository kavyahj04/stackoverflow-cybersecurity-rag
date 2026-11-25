"""One way to connect to Postgres.

Every connection gets pgvector's types registered and the HNSW search
setting applied, so no caller can forget either.
"""
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from rag.config import get_settings


def configure(conn: psycopg.Connection) -> None:
    """What every connection needs before its first search."""
    register_vector(conn)
    # SET can't take a query parameter; int() guarantees only a number goes into the SQL
    conn.execute(f"SET hnsw.ef_search = {int(get_settings().hnsw_ef_search)}")


def connect(autocommit: bool = True) -> psycopg.Connection:
    """One connection, for scripts and evaluations."""
    conn = psycopg.connect(get_settings().database_url, autocommit=autocommit)
    configure(conn)
    return conn


def make_pool() -> ConnectionPool:
    """A pool of ready connections, for the API (many requests at the same time).

    Opening a connection to Supabase takes ~0.5 s, so the API borrows an open one
    per request and gives it back. check= tests a connection before lending it,
    because the server closes connections that sit idle for a long time.
    """
    settings = get_settings()
    return ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        timeout=settings.db_pool_timeout_s,   # waiting longer than this for a free connection -> error
        kwargs={"autocommit": True},
        configure=configure,
        check=ConnectionPool.check_connection,
        open=True,
    )
