from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from .database import get_db
from .models import Base, Face
from .face import model_manager
from .schemas import (
    MatchRequest,
    RecognizeRequest,
    CameraBatchRequest,
    MatchResponse,
    RecognizeResponse,
    CameraResponse,
)

app = FastAPI(title="InsightFace API")

@app.on_event("startup")
async def on_startup():
    async with get_db() as session:
        await session.run_sync(Base.metadata.create_all)

@app.post("/match", response_model=MatchResponse)
async def match_faces(req: MatchRequest):
    model = await model_manager.get_model(req.model)
    img1 = model.get(req.image1)
    img2 = model.get(req.image2)
    if not img1.faces or not img2.faces:
        return MatchResponse(similarity=0.0)
    emb1 = img1.faces[0].normed_embedding
    emb2 = img2.faces[0].normed_embedding
    sim = float(np.dot(emb1, emb2))
    return MatchResponse(similarity=sim)

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest, db: AsyncSession = Depends(get_db)):
    model = await model_manager.get_model(req.model)
    img = model.get(req.image)
    if not img.faces:
        return RecognizeResponse(name=None, distance=None)
    emb = img.faces[0].normed_embedding.tobytes()
    result = await db.execute(
        "SELECT name, embedding FROM faces"
    )
    best_name = None
    best_dist = 1e9
    for name, emb_db in result:
        emb_db = np.frombuffer(emb_db, dtype=np.float32)
        dist = np.linalg.norm(emb_db - img.faces[0].normed_embedding)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_name is None:
        return RecognizeResponse(name=None, distance=None)
    return RecognizeResponse(name=best_name, distance=float(best_dist))

@app.post("/process", response_model=list[CameraResponse])
async def process(req: CameraBatchRequest, db: AsyncSession = Depends(get_db)):
    responses = []
    for cam in req.cameras:
        rec = await recognize(RecognizeRequest(image=cam.image, model=cam.model), db)
        responses.append(CameraResponse(camera_id=cam.camera_id, result=rec))
    return responses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app.main:app", host="0.0.0.0", port=8000)
