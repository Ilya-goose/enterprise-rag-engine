from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Если в .env файле нет этих переменных, будут использованы значения по умолчанию
    api_token: str = "enterprise-rag-2024"

    # Размерность векторов (зависит от ML-модели, для нашей будет 384)
    vector_dim: int = 384

    # Отличная мультиязычная модель, понимает русский и английский
    embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Создаем глобальный объект настроек
settings = Settings()