from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Header both directions of the backend<->ML-service traffic must send:
    # the backend calls POST /classify with this key, and this service uses
    # the same key when it calls back into the backend's retrain-data-pull
    # endpoint. Two services, one shared secret — no need for anything
    # heavier for this project's scale.
    service_api_key: str = "dev-insecure-ml-service-key-change-me"

    model_dir: Path = Path(__file__).resolve().parent.parent / "models"
    model_version: str = "v1"

    backend_url: str = "http://localhost:8000/api"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"mineral_classifier_{self.model_version}.joblib"

    @property
    def meta_path(self) -> Path:
        return self.model_dir / f"mineral_classifier_{self.model_version}.meta.json"


settings = Settings()
