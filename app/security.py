from fastapi import Header, HTTPException, status

from app.config import settings


async def require_service_key(x_ml_service_key: str = Header(...)) -> None:
    if x_ml_service_key != settings.service_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid service key.")
