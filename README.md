`uv run main.py` to launch the server.

basic curl commands:
Upload a file:
```bash
curl -X POST localhost:8000/upload -F "file=@path"
```

Create a processing job:
```bash
curl -X POST localhost:8000/job -H "Content-Type: application/json"
-d '{"files": [FILE_ID], "model": "TreeNetDenoise"}'
```