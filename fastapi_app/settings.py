import yaml
from pathlib import Path
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).with_name('config.yaml')

class Settings(BaseModel):
    default_model: str = 'buffalo_l'
    database_url: str = 'sqlite+aiosqlite:///./faces.db'


def load_settings() -> Settings:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()

settings = load_settings()
