import asyncio
from insightface.app import FaceAnalysis
from functools import lru_cache
from typing import Dict, List

from .settings import settings

SUPPORTED_MODELS: List[str] = [
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc",
]


class FaceModelManager:
    """Lazy loader and cache for ``FaceAnalysis`` models."""

    def __init__(self) -> None:
        # Map model name/path to loaded ``FaceAnalysis`` instance
        self._models: Dict[str, FaceAnalysis] = {}
        self._lock = asyncio.Lock()

    async def get_model(self, model_name: str | None = None) -> FaceAnalysis:
        name = model_name or settings.default_model
        async with self._lock:
            if name not in self._models:
                model = FaceAnalysis(name)
                model.prepare(
                    ctx_id=settings.ctx_id,
                    det_size=settings.det_size,
                )
                self._models[name] = model
            return self._models[name]

    async def get_faces(self, image: str, model_name: str | None = None):
        """Asynchronously run face detection and recognition."""
        model = await self.get_model(model_name)
        return await asyncio.to_thread(model.get, image)

model_manager = FaceModelManager()
