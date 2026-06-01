#!/usr/bin/env python

"""TCP visual localization server for live CARLA frames.

Run this script in a modern Python environment with torch, hloc, pycolmap,
and h5py installed. The CARLA Python 3.6 process sends RGB frames over a
small stdlib TCP protocol and receives estimated CARLA x/y/z + heading.
"""

import argparse
import json
import socketserver
import struct
import traceback

import numpy as np

from carla_live_localization import LiveCarlaLocalizer


HEADER = struct.Struct('!III')
MAX_IMAGE_BYTES = 64 * 1024 * 1024


def recvall(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError('Client disconnected before sending the full request.')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def send_json(sock, payload):
    data = json.dumps(payload).encode('utf-8')
    sock.sendall(struct.pack('!I', len(data)) + data)


class LocalizationRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            header = recvall(self.request, HEADER.size)
            width, height, payload_size = HEADER.unpack(header)
            if payload_size <= 0 or payload_size > MAX_IMAGE_BYTES:
                raise ValueError('Invalid image payload size: {}'.format(payload_size))
            expected = int(width) * int(height) * 3
            if payload_size != expected:
                raise ValueError(
                    'Payload size mismatch: got {} bytes, expected {} for {}x{} RGB'.format(
                        payload_size, expected, width, height
                    )
                )

            payload = recvall(self.request, payload_size)
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()
            result = self.server.localizer.localize_xyz_heading(rgb)
            send_json(self.request, result)
        except Exception as exc:
            traceback.print_exc()
            send_json(self.request, {'success': False, 'error': str(exc)})


class LocalizationServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, localizer):
        socketserver.TCPServer.__init__(self, server_address, handler_cls)
        self.localizer = localizer


def parse_args():
    parser = argparse.ArgumentParser(description='Live visual localization TCP server')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--bundle', default='sim-19-may-bundle')
    parser.add_argument('--num-loc', type=int, default=10)
    parser.add_argument('--max-error', type=float, default=12.0)
    parser.add_argument('--retrieval-max-size', type=int, default=1024,
                        help='Resize the longest side before NetVLAD retrieval to reduce GPU memory (default: 1024)')
    parser.add_argument('--device', default=None, help='torch device override, e.g. cuda or cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    localizer = LiveCarlaLocalizer(
        bundle_root=args.bundle,
        num_loc=args.num_loc,
        max_error=args.max_error,
        retrieval_max_size=args.retrieval_max_size,
        device=args.device,
    )
    server = LocalizationServer((args.host, args.port), LocalizationRequestHandler, localizer)
    print('[Localization Server] listening on {}:{} bundle={}'.format(args.host, args.port, args.bundle))
    try:
        server.serve_forever()
    finally:
        localizer.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
