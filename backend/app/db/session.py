import os
from contextlib import asynccontextmanager

import asyncpg


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://medical_rag:medical_rag_password@localhost:5433/medical_rag",
)


@asynccontextmanager
async def get_db_connection():
    conn = await asyncpg.connect(DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        yield conn
    finally:
        await conn.close()