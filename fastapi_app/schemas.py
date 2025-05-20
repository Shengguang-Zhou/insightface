from pydantic import BaseModel
from typing import List, Optional

class MatchRequest(BaseModel):
    image1: str
    image2: str
    model: Optional[str] = None

class RecognizeRequest(BaseModel):
    image: str
    model: Optional[str] = None

class CameraRequest(BaseModel):
    camera_id: str
    image: str
    rtsp: Optional[str] = None
    model: Optional[str] = None

class CameraBatchRequest(BaseModel):
    cameras: List[CameraRequest]

class RegisterRequest(BaseModel):
    """Request model for adding a new face to the database."""
    name: str
    image: str
    model: Optional[str] = None

class RegisterResponse(BaseModel):
    """Response returned after a successful registration."""
    id: int
    name: str

class MatchResponse(BaseModel):
    similarity: float

class RecognizeResponse(BaseModel):
    name: str | None
    distance: float | None

class CameraResponse(BaseModel):
    camera_id: str
    rtsp: Optional[str] = None
    result: RecognizeResponse
