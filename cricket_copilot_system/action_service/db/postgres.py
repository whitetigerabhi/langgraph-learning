import os
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import SimpleConnectionPool

PG_HOST = os.environ["PG_HOST"]
PG_DB = os.environ.get("PG_DB", "postgres")
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_SSLMODE = os.environ.get("PG_SSLMODE", "require")

PG_STATEMENT_TIMEOUT_MS = int(os.environ.get("PG_STATEMENT_TIMEOUT_MS", "8000"))

_pool: SimpleConnectionPool | None = None

def init_pool(minconn: int = 1, maxconn: int = 5) -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=PG_HOST,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            port=PG_PORT,
            sslmode=PG_SSLMODE,
        )
    return _pool

@contextmanager
def get_conn():
    pool = init_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {PG_STATEMENT_TIMEOUT_MS}")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def fetch_all(sql: str, params: dict | None = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return cols, rows

def execute(sql: str, params: dict | None = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.rowcount