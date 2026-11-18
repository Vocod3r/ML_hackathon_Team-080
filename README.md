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
data/raw/student_test/00000007/00000007.track
data/raw/student_test/00000007/00000007.nodes
data/raw/student_test/00000007/00000007.arcs
data/raw/student_test/00000007/00000007.route
```

Configure trip and artifact paths in `configs/api.yaml`. Paths may be absolute or relative to the repository root. If no trained model is available, the API uses a nearest-arc fallback model.

## Run the API

From the repository root:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The server listens on all network interfaces, so another device on the same network can connect using this computer's LAN IP address. Allow inbound TCP port 8000 in the computer's firewall, restricted to the local network where possible.

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

## Run the UI

In a second terminal, from the repository root:

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

Open the Streamlit URL printed in the terminal. The UI defaults to `http://localhost:8000`; set `PATHSNAP_API_URL` or edit the API URL in the sidebar to connect to another host.

## Two-laptop setup

1. On Laptop 1, start the API with the command above and find its LAN IP address (`ip addr` on Linux, `ipconfig` on Windows, or `ifconfig` on macOS).
2. Allow inbound TCP port 8000 through Laptop 1's firewall for the local network.
3. On Laptop 2, install the project dependencies and run `streamlit run ui/app.py`.
4. In the UI sidebar, set the API URL to `http://<Laptop-1-LAN-IP>:8000`.

If hosting the API on EC2 instead, add an inbound security-group rule for TCP 8000 limited to the client IP range, and use the instance address as the API URL. Avoid exposing the evaluation API publicly without access controls.
