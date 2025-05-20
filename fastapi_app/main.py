from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import asyncio

from .database import get_db
from .models import Base, Face
from .face import model_manager
from .face import SUPPORTED_MODELS
from .schemas import (
    MatchRequest,
    RecognizeRequest,
    CameraBatchRequest,
    RegisterRequest,
    RegisterResponse,
    MatchResponse,
    RecognizeResponse,
    CameraResponse,
)

app = FastAPI(title="InsightFace API")


@app.get("/models", response_model=list[str])
async def list_models():
    """Return built-in model pack names."""
    return SUPPORTED_MODELS

@app.on_event("startup")
async def on_startup():
    async with get_db() as session:
        await session.run_sync(Base.metadata.create_all)

@app.post("/match", response_model=MatchResponse)
async def match_faces(req: MatchRequest):
    img1 = await model_manager.get_faces(req.image1, req.model)
    img2 = await model_manager.get_faces(req.image2, req.model)
    if not img1 or not img2:
        return MatchResponse(similarity=0.0)
    emb1 = img1[0].normed_embedding
    emb2 = img2[0].normed_embedding
    sim = float(np.dot(emb1, emb2))
    return MatchResponse(similarity=sim)


@app.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new face in the database."""
    faces = await model_manager.get_faces(req.image, req.model)
    if not faces:
        return RegisterResponse(id=-1, name=req.name)
    emb = faces[0].normed_embedding.tobytes()
    face = Face(name=req.name, embedding=emb)
    db.add(face)
    await db.commit()
    await db.refresh(face)
    return RegisterResponse(id=face.id, name=face.name)

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest, db: AsyncSession = Depends(get_db)):
    faces = await model_manager.get_faces(req.image, req.model)
    if not faces:
        return RecognizeResponse(name=None, distance=None)
    emb = faces[0].normed_embedding.tobytes()
    result = await db.execute(
        "SELECT name, embedding FROM faces"
    )
    best_name = None
    best_dist = 1e9
    for name, emb_db in result:
        emb_db = np.frombuffer(emb_db, dtype=np.float32)
        dist = np.linalg.norm(emb_db - faces[0].normed_embedding)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_name is None:
        return RecognizeResponse(name=None, distance=None)
    return RecognizeResponse(name=best_name, distance=float(best_dist))

@app.post("/process", response_model=list[CameraResponse])
async def process(req: CameraBatchRequest, db: AsyncSession = Depends(get_db)):
    async def handle_camera(cam):
        rec = await recognize(
            RecognizeRequest(image=cam.image, model=cam.model), db
        )
        return CameraResponse(camera_id=cam.camera_id, rtsp=cam.rtsp, result=rec)

    tasks = [handle_camera(cam) for cam in req.cameras]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app.main:app", host="0.0.0.0", port=8000)
