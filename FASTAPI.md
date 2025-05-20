# FastAPI InsightFace API

This directory provides a minimal FastAPI application for face matching and recognition using InsightFace.

## Configuration

`fastapi_app/config.yaml` contains the default model and database URL. You can change these values or pass a different model via API requests.

## Running

```bash
python -m fastapi_app.main
```

## API Endpoints

- `POST /match` – compare two images
- `POST /recognize` – recognize a single face from the database
- `POST /process` – batch recognition for multiple cameras

All endpoints accept an optional `model` field to specify a different model.
