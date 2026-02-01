from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    PROJECT_NAME: str = "ExpenseAlly AI Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    JWT_KEY: str = "ACDt1vR3lXToPQ1g3MyNqklasoiurhhfjkk397ahi"
    JWT_ISSUER: str = "https://localhost:44329"
    JWT_AUDIENCE: str = "http://localhost:7008"
    
    DB_SERVER: str = "."
    DB_NAME: str = "ExpenseAllyDatabase"
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"
    DB_TRUSTED_CONNECTION: bool = True
    DB_TRUST_SERVER_CERTIFICATE: bool = True
    
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:7008,https://localhost:44329"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        if isinstance(self.ALLOWED_ORIGINS, list):
            return self.ALLOWED_ORIGINS
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # ML Model Configuration
    ANOMALY_DETECTION_CONTAMINATION: float = 0.15
    ANOMALY_PERCENTAGE_THRESHOLD: float = 50.0  # Minimum percentage change to flag
    ANOMALY_STD_THRESHOLD: float = 1.5  # Standard deviations for statistical detection
    ANOMALY_MIN_TRANSACTIONS: int = 5
    ANOMALY_MIN_HISTORICAL_PERIODS: int = 2
    ANOMALY_HISTORICAL_DAYS: int = 90  # Days of history to analyze
    
    # Performance Configuration
    ENABLE_CACHING: bool = False  # Set to True to enable result caching
    CACHE_TTL_SECONDS: int = 300  # 5 minutes cache TTL
    
    # Rate Limiting (if implemented)
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

