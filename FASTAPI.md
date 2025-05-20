# FastAPI InsightFace API

This directory provides a minimal FastAPI application for face matching and recognition using InsightFace.

## Configuration

`fastapi_app/config.yaml` contains the default model, database URL and model parameters:

```yaml
default_model: buffalo_l        # model name or path
database_url: sqlite+aiosqlite:///./faces.db
ctx_id: 0                       # -1 for CPU, 0 for first GPU
det_size: [640, 640]            # detection resolution
```

You can overwrite the model by providing a `model` field in each request.

## Running

```bash
python -m fastapi_app.main
```

## API Endpoints

- `GET /models` – list available model packs
- `POST /match` – compare two images
- `POST /register` – store an image in the database
- `POST /recognize` – recognize a single face from the database
- `POST /process` – batch recognition for multiple cameras (supports concurrent processing)

All endpoints accept an optional `model` field to specify a different model. For camera processing
the request and response include an optional `rtsp` field so the caller can track which camera
source produced the result.

The built‑in model pack names are:

```text
antelopev2
buffalo_l
buffalo_m
buffalo_s
buffalo_sc
```
