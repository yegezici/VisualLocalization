#!/usr/bin/env python3

import argparse
import socketserver
import threading
import time
import traceback

from protocol import discard_bytes, read_rgb_frame_header, read_rgb_payload, send_json


class LocalizationHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        started = time.time()
        try:
            self.request.settimeout(self.server.request_timeout)
            meta = read_rgb_frame_header(self.request, self.server.max_image_bytes)

            if self.server.localizer is None:
                discard_bytes(self.request, meta['payload_size'])
                send_json(self.request, {
                    'success': False,
                    'error': 'received frame; localization disabled by --dry-run',
                    'width': meta['width'],
                    'height': meta['height'],
                })
                return

            if not self.server.localization_lock.acquire(False):
                discard_bytes(self.request, meta['payload_size'])
                send_json(self.request, {
                    'success': False,
                    'error': 'localization server busy; try again after the current request finishes',
                    'width': meta['width'],
                    'height': meta['height'],
                })
                print('[Busy] rejected request while localization is running', flush=True)
                return

            try:
                rgb = read_rgb_payload(self.request, meta)
                elapsed_ms = (time.time() - started) * 1000.0
                print(
                    '[Request] '
                    f'client={self.client_address[0]}:{self.client_address[1]} '
                    f'frame={meta["width"]}x{meta["height"]} '
                    f'bytes={meta["payload_size"]} '
                    f'recv_ms={elapsed_ms:.1f}',
                    flush=True,
                )

                loc_started = time.time()
                result = self.server.localizer.localize_xyz_heading(rgb)
                result['width'] = meta['width']
                result['height'] = meta['height']
                result['server_elapsed_ms'] = round((time.time() - started) * 1000.0, 1)
                result['localization_elapsed_ms'] = round((time.time() - loc_started) * 1000.0, 1)
                print(
                    '[Result] '
                    f'success={result.get("success")} '
                    f'inliers={result.get("num_inliers", 0)} '
                    f'corr={result.get("num_correspondences", 0)} '
                    f'retrieved={result.get("retrieved", [])} '
                    f'loc_ms={result["localization_elapsed_ms"]}',
                    flush=True,
                )
                try:
                    send_json(self.request, result)
                except (BrokenPipeError, ConnectionResetError, ConnectionError, TimeoutError) as exc:
                    print(f'[Response Error] client disconnected before response was sent: {exc}', flush=True)
            finally:
                self.server.localization_lock.release()
        except Exception as exc:
            traceback.print_exc()
            try:
                send_json(self.request, {'success': False, 'error': str(exc)})
            except Exception:
                pass


class ThreadedLocalizationServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_cls,
        max_image_bytes: int,
        request_timeout: float,
        localizer,
    ):
        super().__init__(server_address, handler_cls)
        self.max_image_bytes = int(max_image_bytes)
        self.request_timeout = float(request_timeout)
        self.localizer = localizer
        self.localization_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Jetson Nano CARLA localization TCP server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--bundle', default='../../datasets/train_dataset_585_24_may-bundle')
    parser.add_argument('--max-image-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--request-timeout', type=float, default=30.0)
    parser.add_argument('--retrieval-max-size', type=int, default=512)
    parser.add_argument('--local-max-size', type=int, default=512)
    parser.add_argument('--max-keypoints', type=int, default=1024)
    parser.add_argument('--num-loc', type=int, default=5)
    parser.add_argument('--max-error', type=float, default=12.0)
    parser.add_argument('--min-inliers', type=int, default=15,
                        help='Reject pose estimates with fewer PnP inliers (default: 15)')
    parser.add_argument('--device', default=None, help='torch device override, e.g. cuda or cpu')
    parser.add_argument('--dry-run', action='store_true',
                        help='Receive frames and return a test response without loading localization models')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    localizer = None
    if args.dry_run:
        print('[Localization Disabled] running in --dry-run mode', flush=True)
    else:
        from nano_localizer import LiveCarlaLocalizer

        localizer = LiveCarlaLocalizer(
            bundle_root=args.bundle,
            num_loc=args.num_loc,
            max_error=args.max_error,
            min_inliers=args.min_inliers,
            retrieval_max_size=args.retrieval_max_size,
            local_max_size=args.local_max_size,
            max_keypoints=args.max_keypoints,
            device=args.device,
        )

    server = ThreadedLocalizationServer(
        (args.host, args.port),
        LocalizationHandler,
        max_image_bytes=args.max_image_bytes,
        request_timeout=args.request_timeout,
        localizer=localizer,
    )
    print(
        '[Server Ready] '
        f'listening={args.host}:{args.port} '
        f'bundle={args.bundle} '
        f'max_image_bytes={args.max_image_bytes}',
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[Server] stopped by user', flush=True)
    finally:
        if localizer is not None:
            localizer.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
