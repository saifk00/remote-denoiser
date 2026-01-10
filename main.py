from uuid import uuid4
import fastapi
from fastapi import Request, UploadFile, File, FastAPI
from dataclasses import dataclass
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
import os
import uvicorn
import threading
import hashlib

from worker import ProcessConfig, Worker
from database import init_database, find_by_hash, db_transaction, insert_hash_record
from config import get_data_dir

data_dir = get_data_dir()

class WorkerRegistry:
    def __init__(self, models: list[str]) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, Worker] = [{model: Worker(model)} for model in models]

    def get(self, model: str) -> Worker:
        return self._workers[model]

AVAILABLE_MODELS = ["TreeNetDenoise", "DeepSharpen", "TreeNetDenoiseSuperLight"]
worker_registry = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_registry
    init_database()
    worker_registry = WorkerRegistry(AVAILABLE_MODELS)
    yield

app = fastapi.FastAPI(lifespan=lifespan)

@dataclass
class RawForgeArgs:
    model: str
    in_file: str
    out_file: str

def process(args: RawForgeArgs):
    config = ProcessConfig(
        in_file=args.in_file,
        out_file=args.out_file
    )

    worker_registry.get(args.model).submit(config)

class JobConfig(BaseModel):
    files: list[str]  # list of files by their id as returned by /upload
    model: str = "TreeNetDenoise"  # model name to use

@app.post("/job")
async def create_job(job: JobConfig):
    for file_id in job.files:
        in_file = str(data_dir / file_id / "image.dng")
        out_file = str(data_dir / file_id / "denoised.dng")
        args = RawForgeArgs(model=job.model, in_file=in_file, out_file=out_file)
        process(args)
    return {"status": "jobs created"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_size = len(file_bytes)
    filename = file.filename or "unknown"

    file_hash = hashlib.blake2b(file_bytes, digest_size=32).hexdigest()

    existing = find_by_hash(file_hash)
    if existing:
        return {"file_id": existing["file_id"]}

    file_id = uuid4().hex

    with db_transaction() as conn:
        os.makedirs(data_dir / file_id, exist_ok=True)
        file_path = data_dir / file_id / "image.dng"
        with open(str(file_path), "wb") as f:
            f.write(file_bytes)

        insert_hash_record(conn, file_hash, file_id, file_size, filename)

    return {"file_id": file_id}

@app.get("/job/{file_id}/download")
async def download_file(file_id: str):
    file_path = data_dir / file_id / "denoised.dng"
    return fastapi.FileResponse(str(file_path))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
