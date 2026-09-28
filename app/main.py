from fastapi import Depends, FastAPI, HTTPException, status

from app import model
from app.schemas import ClassifyRequest, ClassifyResponse, ConfidenceAlternative
from app.security import require_service_key
from app.spectral import SpectrumRangeError

app = FastAPI(title="MDMIS ML Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "mdmis-ml-service", "model_version": model.model_version()}


@app.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(require_service_key)])
def classify(payload: ClassifyRequest):
    if len(payload.x_values) != len(payload.intensities):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "x_values and intensities must be the same length.")
    try:
        result = model.classify(payload.x_values, payload.intensities)
    except SpectrumRangeError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    return ClassifyResponse(
        mineral_type=result["mineral_type"],
        confidence_score=result["confidence_score"],
        confidence_alternatives=[ConfidenceAlternative(**a) for a in result["alternatives"]],
        model_version=model.model_version(),
    )
