import json
import os
import uuid
from urllib import request


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


def test_upload_random_bytes(api_server):
    """Upload random bytes to /upload and assert we get a 200 and a file_id in response."""
    url = api_server + "/upload"

    # Random payload
    payload = os.urandom(1024)
    boundary = uuid.uuid4().hex
    body = _build_multipart("file", "test.bin", payload, boundary)

    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with request.urlopen(req, timeout=10) as resp:
        code = resp.getcode()
        assert code == 200, f"Unexpected response code: {code}"
        data = resp.read()
        # parse JSON body
        try:
            obj = json.loads(data.decode())
        except Exception:
            raise AssertionError(f"Response is not valid JSON: {data!r}")

    assert "file_id" in obj, f"Response JSON missing file_id: {obj}"
