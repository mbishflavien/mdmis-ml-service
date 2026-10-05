from fastapi import Depends, FastAPI, HTTPException, status

from app import model, pathfinder
from app.config import settings
from app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ConfidenceAlternative,
    PathfinderCategoryAlternative,
    PathfinderRequest,
    PathfinderResponse,
)
from app.security import require_service_key
from app.spectral import SpectrumRangeError

app = FastAPI(title="MDMIS ML Service", version="1.0.0")

_PATHFINDER_SENSORS = ("sentinel2", "as7265x")


@app.get("/health")
def health():
    versions = {}
    for sensor_type in settings.model_name_by_sensor:
        try:
            versions[sensor_type] = model.model_version(sensor_type)
        except FileNotFoundError:
            versions[sensor_type] = None  # trained but not yet run, or dataset not built yet
    pathfinder_versions = {}
    for sensor_type in _PATHFINDER_SENSORS:
        try:
            pathfinder_versions[sensor_type] = pathfinder.pathfinder_model_version(sensor_type)
        except FileNotFoundError:
            pathfinder_versions[sensor_type] = None
    return {
        "status": "ok",
        "service": "mdmis-ml-service",
        "model_versions": versions,
        "pathfinder_versions": pathfinder_versions,
    }


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


@app.post("/pathfinder", response_model=PathfinderResponse, dependencies=[Depends(require_service_key)])
def classify_pathfinder(payload: PathfinderRequest):
    """Gold-pathfinder alteration indicator — separate from /classify on
    purpose. See app/pathfinder.py for why this never outputs "gold"."""
    if payload.sensor_type not in _PATHFINDER_SENSORS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"No pathfinder model for sensor_type={payload.sensor_type!r}. Supported: {list(_PATHFINDER_SENSORS)}.",
        )
    if len(payload.x_values) != len(payload.intensities):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "x_values and intensities must be the same length.")
    try:
        result = pathfinder.classify_pathfinder(payload.x_values, payload.intensities, sensor_type=payload.sensor_type)
    except SpectrumRangeError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"No trained pathfinder model for sensor_type={payload.sensor_type!r} yet.",
        )

    return PathfinderResponse(
        category=result["category"],
        category_score=result["category_score"],
        category_alternatives=[PathfinderCategoryAlternative(**a) for a in result["category_alternatives"]],
        pathfinder_score=result["pathfinder_score"],
        caveat=result["caveat"],
        model_version=pathfinder.pathfinder_model_version(payload.sensor_type),
    )
