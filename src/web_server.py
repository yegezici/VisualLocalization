#!/usr/bin/env python

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / 'web' / 'static'
INDEX_FILE = STATIC_DIR / 'index.html'

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


class ProcessState:
    def __init__(self) -> None:
        self.visual_localization: Optional[subprocess.Popen] = None
        self.localization_server: Optional[subprocess.Popen] = None
        self.carla_sim: Optional[subprocess.Popen] = None


state = ProcessState()


class WebSocketManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload)
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                await self.disconnect(ws)


ws_manager = WebSocketManager()
last_telemetry: Dict[str, Any] = {}


@app.get('/')
async def index() -> FileResponse:
    return FileResponse(str(INDEX_FILE))


@app.get('/api/status')
async def get_status() -> JSONResponse:
    return JSONResponse({
        'visual_localization_running': state.visual_localization is not None and state.visual_localization.poll() is None,
        'localization_server_running': state.localization_server is not None and state.localization_server.poll() is None,
        'carla_sim_running': state.carla_sim is not None and state.carla_sim.poll() is None,
    })


@app.post('/api/telemetry')
async def post_telemetry(payload: Dict[str, Any]) -> JSONResponse:
    last_telemetry.update(payload)
    await ws_manager.broadcast(payload)
    return JSONResponse({'ok': True})


def build_python_command(script_name: str, args: List[str]) -> List[str]:
    script_path = ROOT_DIR / script_name
    return [sys.executable, str(script_path), *args]


def terminate_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == 'nt':
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def launch_carla_sim(exe_path: str, args: List[str]) -> subprocess.Popen:
    if not exe_path:
        raise ValueError('CARLA executable path is required.')
    exe = Path(exe_path)
    if not exe.exists():
        raise FileNotFoundError(exe)
    command = [str(exe), *args]
    return subprocess.Popen(
        command,
        cwd=str(exe.parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )


@app.post('/api/start')
async def start_processes(payload: Dict[str, Any]) -> JSONResponse:
    start_visual = bool(payload.get('start_visual', True))
    start_localization_server = bool(payload.get('start_localization_server', True))

    if start_visual and (state.visual_localization is None or state.visual_localization.poll() is not None):
        args = [
            '--host', str(payload.get('carla_host', '127.0.0.1')),
            '--port', str(payload.get('carla_port', 2000)),
            '--loc-host', str(payload.get('loc_host', '127.0.0.1')),
            '--loc-port', str(payload.get('loc_port', 5555)),
            '--web-enabled',
            '--web-host', str(payload.get('web_host', '127.0.0.1')),
            '--web-port', str(payload.get('web_port', 8000)),
            '--web-rate', str(payload.get('web_rate', 6.0)),
        ]
        if payload.get('no_preview_window'):
            args.append('--no-preview-window')
        state.visual_localization = subprocess.Popen(
            build_python_command('visual_localization.py', args),
            cwd=str(ROOT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        )

    if start_localization_server and (state.localization_server is None or state.localization_server.poll() is not None):
        args = [
            '--host', str(payload.get('localization_host', '127.0.0.1')),
            '--port', str(payload.get('localization_port', 5555)),
            '--bundle-root', str(payload.get('bundle_root', 'sim-19-may-bundle')),
        ]
        state.localization_server = subprocess.Popen(
            build_python_command('localization_server.py', args),
            cwd=str(ROOT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        )

    return JSONResponse({'ok': True})


@app.post('/api/launch-carla')
async def launch_carla(payload: Dict[str, Any]) -> JSONResponse:
    if state.carla_sim is not None and state.carla_sim.poll() is None:
        return JSONResponse({'ok': True, 'already_running': True})

    exe_path = str(payload.get('carla_exe', '')).strip()
    args_raw = str(payload.get('carla_args', '')).strip()
    args = shlex.split(args_raw)
    try:
        state.carla_sim = launch_carla_sim(exe_path, args)
    except Exception as exc:
        return JSONResponse({'ok': False, 'error': str(exc)})

    return JSONResponse({'ok': True})


@app.post('/api/stop')
async def stop_processes() -> JSONResponse:
    terminate_process(state.visual_localization)
    terminate_process(state.localization_server)
    terminate_process(state.carla_sim)
    state.visual_localization = None
    state.localization_server = None
    state.carla_sim = None
    return JSONResponse({'ok': True})


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws_manager.connect(ws)
    try:
        if last_telemetry:
            await ws.send_text(json.dumps(last_telemetry))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)
    except Exception:
        await ws_manager.disconnect(ws)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Web controller for CARLA visual localization demo')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('uvicorn is required: pip install uvicorn[standard]') from exc
    uvicorn.run('web_server:app', host=args.host, port=args.port, reload=False)


if __name__ == '__main__':
    main()
