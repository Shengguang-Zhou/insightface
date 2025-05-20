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
    model: Optional[str] = None

class CameraBatchRequest(BaseModel):
    cameras: List[CameraRequest]

class MatchResponse(BaseModel):
    similarity: float

class RecognizeResponse(BaseModel):
    name: str | None
    distance: float | None

class CameraResponse(BaseModel):
    camera_id: str
    result: RecognizeResponse
