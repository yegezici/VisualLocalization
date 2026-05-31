import json
from io import BytesIO
import struct
from typing import Dict, Tuple

import numpy as np


HEADER = struct.Struct('!III')
RESPONSE_SIZE = struct.Struct('!I')
MAX_RECV_CHUNK = 1024 * 1024


def recvall(sock, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(min(remaining, MAX_RECV_CHUNK))
        if not chunk:
            raise ConnectionError('Peer disconnected before the full message was received.')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def recv_exact_into(sock, buffer) -> None:
    view = memoryview(buffer).cast('B')
    offset = 0
    remaining = len(view)
    while remaining:
        end = offset + min(remaining, MAX_RECV_CHUNK)
        received = sock.recv_into(view[offset:end])
        if received == 0:
            raise ConnectionError('Peer disconnected before the full message was received.')
        offset += received
        remaining -= received


def discard_bytes(sock, size: int) -> None:
    scratch = bytearray(min(MAX_RECV_CHUNK, max(1, int(size))))
    remaining = int(size)
    while remaining:
        view = memoryview(scratch)[:min(remaining, len(scratch))]
        received = sock.recv_into(view)
        if received == 0:
            raise ConnectionError('Peer disconnected before the full message was received.')
        remaining -= received


def read_rgb_frame_header(sock, max_image_bytes: int) -> Dict[str, int]:
    header = recvall(sock, HEADER.size)
    width, height, payload_size = HEADER.unpack(header)

    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid frame size: {width}x{height}')
    if payload_size <= 0 or payload_size > max_image_bytes:
        raise ValueError(f'Invalid payload size: {payload_size}')

    expected = int(width) * int(height) * 3
    if payload_size != expected:
        raise ValueError(
            f'Payload size mismatch: got {payload_size}, expected {expected} '
            f'for {width}x{height} RGB'
        )

    return {'width': int(width), 'height': int(height), 'payload_size': int(payload_size)}


def read_rgb_payload(sock, meta: Dict[str, int]) -> np.ndarray:
    rgb = np.empty((meta['height'], meta['width'], 3), dtype=np.uint8)
    recv_exact_into(sock, rgb)
    return rgb


def read_rgb_frame(sock, max_image_bytes: int) -> Tuple[np.ndarray, Dict[str, int]]:
    meta = read_rgb_frame_header(sock, max_image_bytes)
    rgb = read_rgb_payload(sock, meta)
    return rgb, meta


def send_json(sock, payload: Dict[str, object]) -> None:
    data = json.dumps(payload).encode('utf-8')
    sock.sendall(RESPONSE_SIZE.pack(len(data)) + data)


def send_rgb_frame(sock, rgb: np.ndarray) -> None:
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 RGB uint8 image, got shape={arr.shape}')
    height, width, _ = arr.shape
    payload = arr.tobytes()
    sock.sendall(HEADER.pack(int(width), int(height), len(payload)) + payload)


def read_json_response(sock) -> Dict[str, object]:
    response_size = RESPONSE_SIZE.unpack(recvall(sock, RESPONSE_SIZE.size))[0]
    return json.loads(recvall(sock, response_size).decode('utf-8'))


def send_npz_response(sock, metadata: Dict[str, object], arrays: Dict[str, np.ndarray]) -> None:
    """Send JSON metadata followed by a length-prefixed NPZ payload."""
    meta_data = json.dumps(metadata).encode('utf-8')
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    sock.sendall(
        RESPONSE_SIZE.pack(len(meta_data))
        + meta_data
        + RESPONSE_SIZE.pack(len(payload))
        + payload
    )


def read_npz_response(sock) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Read a JSON metadata block and a length-prefixed NPZ payload."""
    meta_size = RESPONSE_SIZE.unpack(recvall(sock, RESPONSE_SIZE.size))[0]
    metadata = json.loads(recvall(sock, meta_size).decode('utf-8'))
    payload_size = RESPONSE_SIZE.unpack(recvall(sock, RESPONSE_SIZE.size))[0]
    payload = recvall(sock, payload_size)
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return metadata, arrays
