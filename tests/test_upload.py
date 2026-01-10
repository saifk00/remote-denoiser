import json
import os
import uuid
from urllib import request
import concurrent.futures


def _build_multipart(field_name: str, filename: str, data: bytes, boundary: str) -> bytes:
    """Construct a multipart/form-data body for a single file field."""
    sep = f"--{boundary}\r\n".encode()
    ending = f"--{boundary}--\r\n".encode()

    headers = (
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()

    body = bytearray()
    body.extend(sep)
    body.extend(headers)
    body.extend(data)
    body.extend(b"\r\n")
    body.extend(ending)
    return bytes(body)


def _upload_bytes(api_server: str, payload: bytes, filename: str = "test.bin") -> dict:
    """Helper function to upload bytes to /upload endpoint and return JSON response."""
    url = api_server + "/upload"
    boundary = uuid.uuid4().hex
    body = _build_multipart("file", filename, payload, boundary)

    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with request.urlopen(req, timeout=10) as resp:
        assert resp.getcode() == 200
        data = resp.read()
        return json.loads(data.decode())


def test_upload_random_bytes(api_server):
    """Upload random bytes to /upload and assert we get a 200 and a file_id in response."""
    payload = os.urandom(1024)
    resp = _upload_bytes(api_server, payload)
    assert "file_id" in resp, f"Response JSON missing file_id: {resp}"


def test_upload_deduplicate_identical_files(api_server):
    """Upload same file twice, should get same file_id."""
    payload = os.urandom(1024)

    resp1 = _upload_bytes(api_server, payload, "file1.dng")
    file_id_1 = resp1["file_id"]
    assert file_id_1

    resp2 = _upload_bytes(api_server, payload, "file2.dng")
    file_id_2 = resp2["file_id"]

    assert file_id_1 == file_id_2, "Identical files should return the same file_id"


def test_upload_different_files_get_different_ids(api_server):
    """Upload different files, should get different file_ids."""
    payload1 = os.urandom(1024)
    payload2 = os.urandom(1024)

    resp1 = _upload_bytes(api_server, payload1)
    resp2 = _upload_bytes(api_server, payload2)

    assert resp1["file_id"] != resp2["file_id"], "Different files should get different file_ids"


def test_deduped_file_only_creates_one_directory(api_server):
    """Verify that deduplication doesn't create duplicate directories."""
    payload = os.urandom(2048)

    resp1 = _upload_bytes(api_server, payload)
    file_id = resp1["file_id"]

    resp2 = _upload_bytes(api_server, payload)
    assert resp2["file_id"] == file_id

    file_path = f"data/{file_id}/image.dng"
    assert os.path.exists(file_path), f"File should exist at {file_path}"

    with open(file_path, "rb") as f:
        stored_content = f.read()
    assert stored_content == payload, "Stored content should match uploaded payload"


def test_concurrent_identical_uploads(api_server):
    """Upload same file from 10 threads simultaneously."""
    payload = os.urandom(4096)
    file_ids = []

    def upload():
        resp = _upload_bytes(api_server, payload)
        file_ids.append(resp["file_id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(upload) for _ in range(10)]
        concurrent.futures.wait(futures)

    assert len(set(file_ids)) == 1, f"All concurrent uploads should return the same file_id, got {set(file_ids)}"
