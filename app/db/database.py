from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import logging
from urllib.parse import quote_plus

from app.core.config import settings

logger = logging.getLogger(__name__)

# Build connection string based on auth type
if settings.DB_USER and settings.DB_PASSWORD:
    # SQL Authentication (Cloud)
    driver = settings.DB_DRIVER.replace(' ', '+')
    password = quote_plus(settings.DB_PASSWORD)
    connection_string = (
        f"mssql+aioodbc://{settings.DB_USER}:{password}@{settings.DB_SERVER}/{settings.DB_NAME}"
        f"?driver={driver}"
        f"&TrustServerCertificate={'yes' if settings.DB_TRUST_SERVER_CERTIFICATE else 'no'}"
    )
else:
    # Windows Authentication (Local)
    connection_string = (
        f"mssql+aioodbc://"
        f"?driver={settings.DB_DRIVER.replace(' ', '+')}"
        f"&server={settings.DB_SERVER}"
        f"&database={settings.DB_NAME}"
        f"&trusted_connection={'yes' if settings.DB_TRUSTED_CONNECTION else 'no'}"
        f"&TrustServerCertificate={'yes' if settings.DB_TRUST_SERVER_CERTIFICATE else 'no'}"
    )

engine = create_async_engine(
    connection_string,
    echo=settings.DEBUG,
    poolclass=NullPool,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("Continuing without database connection...")


async def close_db():
    await engine.dispose()
    logger.info("Database connections closed")

