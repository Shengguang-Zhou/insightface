import asyncio
from insightface.app import FaceAnalysis
from functools import lru_cache
from typing import Dict

from .settings import settings

class FaceModelManager:
    def __init__(self):
        self._models: Dict[str, FaceAnalysis] = {}
        self._lock = asyncio.Lock()

    async def get_model(self, model_name: str | None = None) -> FaceAnalysis:
        name = model_name or settings.default_model
        async with self._lock:
            if name not in self._models:
                model = FaceAnalysis(name)
                model.prepare(ctx_id=0, det_size=(640, 640))
                self._models[name] = model
            return self._models[name]

model_manager = FaceModelManager()
