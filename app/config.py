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

    # One trained model per real sensor in MDMIS_IoT_Budget.docx. "lab" is
    # the Raman/RRUFF model (the SRS's "Lab Spectrometer -> ground truth
    # validation" role); sentinel2/as7265x are the two actual field-sensor
    # classifiers. A sensor_type not in this map (gpr/em/magnetometer/gamma)
    # has no model yet - see README "Not yet built".
    model_name_by_sensor: dict[str, str] = {
        "lab": "mineral_classifier",
        "sentinel2": "mineral_classifier_s2",
        "as7265x": "mineral_classifier_as7265x",
    }

    def model_path(self, sensor_type: str) -> Path:
        name = self.model_name_by_sensor.get(sensor_type, self.model_name_by_sensor["lab"])
        return self.model_dir / f"{name}_{self.model_version}.joblib"

    def meta_path(self, sensor_type: str) -> Path:
        name = self.model_name_by_sensor.get(sensor_type, self.model_name_by_sensor["lab"])
        return self.model_dir / f"{name}_{self.model_version}.meta.json"


settings = Settings()
