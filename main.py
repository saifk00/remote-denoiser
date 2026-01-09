from uuid import uuid4
import fastapi
from fastapi import Request, UploadFile, File
from dataclasses import dataclass
from pydantic import BaseModel
import os
import uvicorn
import threading

from worker import ProcessConfig, Worker
app = fastapi.FastAPI()

class WorkerRegistry:
    def __init__(self, models: list[str]) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, Worker] = [{model: Worker(model)} for model in models]

    def get(self, model: str) -> Worker:
        return self._workers[model]

AVAILABLE_MODELS = ["TreeNetDenoise", "DeepSharpen", "TreeNetDenoiseSuperLight"]
worker_registry = WorkerRegistry(AVAILABLE_MODELS)

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
        in_file = f"data/{file_id}/image.dng"
        out_file = f"data/{file_id}/denoised.dng"
        args = RawForgeArgs(model=job.model, in_file=in_file, out_file=out_file)
        process(args)
    return {"status": "jobs created"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = uuid4().hex
    os.makedirs(f"data/{file_id}", exist_ok=True)
    file_path = f"data/{file_id}/image.dng"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"file_id": file_id}

@app.get("/job/{file_id}/download")
async def download_file(file_id: str):
    file_path = f"data/{file_id}/denoised.dng"
    return fastapi.FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
