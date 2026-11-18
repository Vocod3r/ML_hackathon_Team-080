# PathSnap

PathSnap matches GPS observations to roads in a trip-specific road network. The project includes data preparation, model training, HMM route decoding, and a FastAPI service with Folium maps.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put trip data in the expected layout, for example:

```text
data/student_test/00000007/00000007.track
data/student_test/00000007/00000007.nodes
data/student_test/00000007/00000007.arcs
data/student_test/00000007/00000007.route
```

Configure trip and artifact paths in `configs/api.yaml`. Paths may be absolute or relative to the repository root. If no trained model is available, the API uses a nearest-arc fallback model.

## Run the API

From the repository root:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The server listens on all network interfaces, so another device on the same network can connect using this computer's IP address.

## Endpoints

- `GET /health` — service health
- `GET /trips` — list available trip IDs
- `POST /predict` — predict and evaluate one trip

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"00000007"}'
```

The response includes GPS coordinates, predicted arcs, server-side metrics, elapsed time, and rendered map HTML in `route_geojson`.

Interactive API documentation is available at `http://localhost:8000/docs`.
