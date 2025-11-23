# app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_VERSION: str = "0.1.0"
    MODEL_PATH: str = "model/logistic_model.pkl"
    THRESHOLD: float = 0.5

settings = Settings()