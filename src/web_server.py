#!/usr/bin/env python

import argparse
import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
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

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('web_server')


class ProcessState:
    def __init__(self) -> None:
        self.visual_localization: Optional[subprocess.Popen] = None
        self.localization_server: Optional[subprocess.Popen] = None
        self.carla_sim: Optional[subprocess.Popen] = None
        self.extra_scripts: List[subprocess.Popen] = []


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
        'carla_rpc_up': is_tcp_port_open('127.0.0.1', 2000),
    })


@app.post('/api/telemetry')
async def post_telemetry(payload: Dict[str, Any]) -> JSONResponse:
    last_telemetry.update(payload)
    await ws_manager.broadcast(payload)
    return JSONResponse({'ok': True})


def build_python_command(script_name: str, args: List[str]) -> List[str]:
    script_path = ROOT_DIR / script_name
    return [sys.executable, str(script_path), *args]


def build_python_command_for_path(script_path: str, args: List[str]) -> List[str]:
    return [sys.executable, script_path, *args]


def _stream_process_output(proc: subprocess.Popen, name: str) -> None:
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ''):
        line = line.rstrip()
        if line:
            logger.info('%s | %s', name, line)


def start_process_logger(proc: subprocess.Popen, name: str) -> None:
    thread = threading.Thread(target=_stream_process_output, args=(proc, name), daemon=True)
    thread.start()


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


def terminate_process_list(processes: List[subprocess.Popen]) -> None:
    for proc in list(processes):
        terminate_process(proc)
    processes.clear()


def terminate_stale_script_process(script_path: Path) -> None:
    if os.name == 'nt':
        return
    target = str(script_path)
    try:
        output = subprocess.check_output(
            ['ps', '-eo', 'pid=,args='],
            universal_newlines=True,
        )
    except Exception:
        return

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(' ')
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid() or target not in command:
            continue
        logger.info('Stopping stale script process: %s', command)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def launch_carla_sim(exe_path: str, args: List[str]) -> subprocess.Popen:
    if not exe_path:
        raise ValueError('CARLA executable path is required.')
    exe_path = resolve_carla_exe_path(exe_path)
    exe = Path(exe_path)
    if not exe.exists():
        raise FileNotFoundError(exe)
    command = [str(exe), *args]
    proc = subprocess.Popen(
        command,
        cwd=str(exe.parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    start_process_logger(proc, 'carla')
    return proc


def normalize_windows_path_for_wsl(path: str) -> str:
    if not path or os.name == 'nt':
        return path
    if path.startswith('/'):
        return path
    if len(path) >= 2 and path[1] == ':' and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].lstrip('\\/')
        rest = rest.replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return path


def resolve_carla_exe_path(exe_path: str) -> str:
    exe_path = exe_path.strip()
    if not exe_path:
        return exe_path

    if 'fakepath' in exe_path.lower():
        exe_path = os.path.basename(exe_path)

    candidate = Path(exe_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())

    which_path = shutil.which(exe_path)
    if which_path:
        return which_path

    carla_root = os.environ.get('CARLA_ROOT') or os.environ.get('CARLA_EXE')
    if carla_root:
        root_path = Path(carla_root)
        if root_path.is_dir():
            root_candidate = root_path / exe_path
            if root_candidate.exists():
                return str(root_candidate)
        elif root_path.exists():
            return str(root_path)

    return exe_path


def is_tcp_port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


async def wait_for_tcp_service(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            await asyncio.sleep(0.5)
    return False


async def wait_for_carla_process(
    proc: subprocess.Popen,
    host: str = '127.0.0.1',
    port: int = 2000,
    timeout_s: float = 45.0,
) -> Optional[str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f'CARLA exited early with code {proc.returncode}. Check the CARLA log above.'
        if is_tcp_port_open(host, port):
            return None
        await asyncio.sleep(0.5)
    return f'CARLA RPC did not become available at {host}:{port} within {int(timeout_s)} seconds.'


@app.post('/api/start')
async def start_processes(payload: Dict[str, Any]) -> JSONResponse:
    start_visual = bool(payload.get('start_visual', True))

    if start_visual:
        carla_host = str(payload.get('carla_host', '127.0.0.1'))
        carla_port = int(payload.get('carla_port', 2000))
        carla_ready = await wait_for_tcp_service(carla_host, carla_port, timeout_s=5.0)
        if not carla_ready:
            logger.error('CARLA RPC is not available at %s:%s', carla_host, carla_port)
            return JSONResponse({
                'ok': False,
                'error': 'CARLA RPC is not available. Start CARLA successfully before starting visual localization.',
            })

        logger.info('CARLA RPC is available at %s:%s', carla_host, carla_port)

        if state.visual_localization is not None and state.visual_localization.poll() is None:
            terminate_process(state.visual_localization)
            state.visual_localization = None
        args = [
            '--host', carla_host,
            '--port', str(carla_port),
            '--loc-host', str(payload.get('loc_host', '127.0.0.1')),
            '--loc-port', str(payload.get('loc_port', 5555)),
            '--web-enabled',
            '--web-host', str(payload.get('web_host', '127.0.0.1')),
            '--web-port', str(payload.get('web_port', 8000)),
            '--web-rate', str(payload.get('web_rate', 6.0)),
        ]
        if 'preview_fullscreen' in payload:
            if payload.get('preview_fullscreen'):
                args.append('--fs')
            else:
                args.append('--no-fs')
        if payload.get('no_preview_window'):
            args.append('--no-preview-window')
        state.visual_localization = subprocess.Popen(
            build_python_command('visual_localization.py', args),
            cwd=str(ROOT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        start_process_logger(state.visual_localization, 'visual_localization')

        terminate_stale_script_process(ROOT_DIR / 'carla_examples' / 'generate_traffic.py')
        terminate_process_list(state.extra_scripts)
        extra_scripts = [
            (ROOT_DIR / 'carla_examples' / 'add_buildings_to_map.py', ['--no-debug-boxes']),
        ]
        for script_path, script_args in extra_scripts:
            proc = subprocess.Popen(
                build_python_command_for_path(str(script_path), script_args),
                cwd=str(Path(script_path).parent),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            state.extra_scripts.append(proc)
            start_process_logger(proc, Path(script_path).name)



    return JSONResponse({'ok': True})


@app.post('/api/launch-carla')
async def launch_carla(payload: Dict[str, Any]) -> JSONResponse:
    if state.carla_sim is not None and state.carla_sim.poll() is None:
        return JSONResponse({'ok': True, 'already_running': True})
    if state.carla_sim is not None and state.carla_sim.poll() is not None:
        state.carla_sim = None

    if is_tcp_port_open('127.0.0.1', 2000):
        logger.info('CARLA RPC port 2000 is already in use; treating CARLA as already running.')
        terminate_stale_script_process(ROOT_DIR / 'carla_examples' / 'generate_traffic.py')
        return JSONResponse({'ok': True, 'already_running': True, 'external': True})

    exe_path = str(payload.get('carla_exe', '')).strip()
    exe_path = normalize_windows_path_for_wsl(exe_path)
    args_raw = str(payload.get('carla_args', '')).strip()
    args = shlex.split(args_raw) if args_raw else ['-quality-level=Low']
    terminate_stale_script_process(ROOT_DIR / 'carla_examples' / 'generate_traffic.py')
    try:
        state.carla_sim = launch_carla_sim(exe_path, args)
    except Exception as exc:
        return JSONResponse({'ok': False, 'error': str(exc)})

    launch_error = await wait_for_carla_process(state.carla_sim)
    if launch_error is not None:
        terminate_process(state.carla_sim)
        state.carla_sim = None
        return JSONResponse({'ok': False, 'error': launch_error})

    return JSONResponse({'ok': True})


@app.post('/api/stop')
async def stop_processes() -> JSONResponse:
    terminate_process(state.visual_localization)
    terminate_process(state.localization_server)
    terminate_process(state.carla_sim)
    terminate_process_list(state.extra_scripts)
    terminate_stale_script_process(ROOT_DIR / 'carla_examples' / 'generate_traffic.py')
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
