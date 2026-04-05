import os
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    ASKCOS_BASE_URL: str = "http://localhost:9100"
    HTTP_TIMEOUT: float = 60.0
    USER_ASKCOS: str
    PASSWORD_ASKCOS: str
    

    model_config = SettingsConfigDict(env_file=os.getenv("ENV_FILE", os.path.join(os.path.dirname(__file__), ".env")))


settings = Settings()
