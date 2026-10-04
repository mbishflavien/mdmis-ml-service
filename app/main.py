from fastapi import Depends, FastAPI, HTTPException, status

from app import model
from app.config import settings
from app.schemas import ClassifyRequest, ClassifyResponse, ConfidenceAlternative
from app.security import require_service_key
from app.spectral import SpectrumRangeError

app = FastAPI(title="MDMIS ML Service", version="1.0.0")


@app.get("/health")
def health():
    versions = {}
    for sensor_type in settings.model_name_by_sensor:
        try:
            versions[sensor_type] = model.model_version(sensor_type)
        except FileNotFoundError:
            versions[sensor_type] = None  # trained but not yet run, or dataset not built yet
    return {"status": "ok", "service": "mdmis-ml-service", "model_versions": versions}


@app.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(require_service_key)])
def classify(payload: ClassifyRequest):
    if payload.sensor_type not in settings.model_name_by_sensor:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"No model for sensor_type={payload.sensor_type!r}. Supported: "
            f"{sorted(settings.model_name_by_sensor)}.",
        )
    if len(payload.x_values) != len(payload.intensities):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "x_values and intensities must be the same length.")
    try:
        result = model.classify(payload.x_values, payload.intensities, sensor_type=payload.sensor_type)
    except SpectrumRangeError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"No trained model artifact for sensor_type={payload.sensor_type!r} yet.",
        )

    return ClassifyResponse(
        mineral_type=result["mineral_type"],
        confidence_score=result["confidence_score"],
        confidence_alternatives=[ConfidenceAlternative(**a) for a in result["alternatives"]],
        model_version=model.model_version(payload.sensor_type),
    )
