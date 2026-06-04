#!/bin/sh
# Block until Qdrant accepts TCP connections, then launch the API.
# Eliminates the boot race where the API imports clients.py (which connects
# to Qdrant) before the qdrant container is ready.
python - <<'PY'
import os, socket, time, urllib.parse

url = os.getenv("QDRANT_URL", "http://qdrant:6333")
p = urllib.parse.urlparse(url)
host, port = p.hostname or "qdrant", p.port or 6333

for _ in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"Qdrant reachable at {host}:{port}")
            break
    except OSError:
        print(f"Waiting for Qdrant at {host}:{port}...")
        time.sleep(2)
else:
    raise SystemExit("Qdrant not reachable after 120s, giving up")
PY

exec uvicorn api:api --host 0.0.0.0 --port 8000
