#!/usr/bin/env python

# Copyright (c) 2026
#
# Multi-camera CARLA dataset capture for HLoc/COLMAP workflows.

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    egg_pattern = os.path.join(
        script_dir,
        '..',
        'carla',
        'dist',
        'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64',
        ),
    )
    sys.path.append(glob.glob(egg_pattern)[0])
except IndexError:
    pass

import carla

try:
    import pygame  # type: ignore[import-not-found]
    from pygame.locals import K_a  # type: ignore[import-not-found]
    from pygame.locals import K_d  # type: ignore[import-not-found]
    from pygame.locals import K_ESCAPE  # type: ignore[import-not-found]
    from pygame.locals import K_q  # type: ignore[import-not-found]
    from pygame.locals import K_s  # type: ignore[import-not-found]
    from pygame.locals import K_SPACE  # type: ignore[import-not-found]
    from pygame.locals import K_w  # type: ignore[import-not-found]
except ImportError as exc:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed') from exc


DIAGONAL_CAMERA_NAMES = ['front', 'right', 'left', 'back']
CARDINAL_CAMERA_NAMES = ['front', 'back']
DEFAULT_SPAWN_LAT = 40.90676579983358
DEFAULT_SPAWN_LON = 29.155049985719415


def parse_res(value: str) -> Tuple[int, int]:
    parts = value.lower().split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Resolution must be WIDTHxHEIGHT')
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect CARLA camera data for HLoc/COLMAP')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--res', default='1920x1080', type=parse_res,
                        help='Per-camera resolution WIDTHxHEIGHT (default: 1920x1080)')
    parser.add_argument('--fov', type=float, default=None,
                        help='Horizontal field-of-view in degrees (default: 95.0, or 120.0 in cardinal mode)')
    parser.add_argument('--output', default='_out/hloc_capture',
                        help='Dataset output directory')
    parser.add_argument('--vehicle-filter', default='vehicle.*',
                        help='Blueprint filter applied before vehicle-type selection')
    parser.add_argument('--vehicle-type', default='sedan', choices=['sedan', 'any'],
                        help='Vehicle class to spawn (default: sedan)')
    parser.add_argument('--camera-rig', default='single', choices=['single', 'diagonal-corners', 'cardinal'],
                        help='Camera placement strategy (default: single)')
    parser.add_argument('--display-scale', type=float, default=0.6,
                        help='Scale factor for the live Pygame preview only; saved images keep --res (default: 0.6)')
    parser.add_argument('--preview-every', type=int, default=1,
                        help='Update the live preview every N render-loop frames; saved images are unaffected (default: 1)')
    parser.add_argument('--preview-layout', default='single', choices=['single', 'grid'],
                        help='Live preview layout. single is faster and better for presentations (default: single)')
    parser.add_argument('--preview-camera', default='front',
                        help='Camera shown when --preview-layout single is used (default: front)')
    parser.add_argument('--smooth-preview', action='store_true',
                        help='Use higher-quality preview scaling. Slower than the default nearest-neighbor scaling.')
    parser.add_argument('--spawn-lat', type=float, default=DEFAULT_SPAWN_LAT,
                        help=f'Latitude used to choose the nearest vehicle spawn point (default: {DEFAULT_SPAWN_LAT})')
    parser.add_argument('--spawn-lon', type=float, default=DEFAULT_SPAWN_LON,
                        help=f'Longitude used to choose the nearest vehicle spawn point (default: {DEFAULT_SPAWN_LON})')
    parser.add_argument('--random-spawn', action='store_true',
                        help='Use a random vehicle spawn point instead of the nearest point to --spawn-lat/--spawn-lon')
    return parser.parse_args()


def pick_vehicle_blueprint(bp_lib: carla.BlueprintLibrary, vehicle_filter: str, vehicle_type: str) -> carla.ActorBlueprint:
    candidates = list(bp_lib.filter(vehicle_filter))
    if not candidates:
        raise RuntimeError(f'No vehicle blueprint matches filter: {vehicle_filter}')

    if vehicle_type == 'any':
        return random.choice(candidates)

    # Prefer known sedan models if they are present in this CARLA build.
    preferred_sedans = [
        'vehicle.tesla.model3',
        'vehicle.audi.tt',
        'vehicle.mercedes.coupe',
        'vehicle.lincoln.mkz_2020',
        'vehicle.lincoln.mkz2017',
    ]
    by_id = {bp.id: bp for bp in candidates}
    for sedan_id in preferred_sedans:
        if sedan_id in by_id:
            return by_id[sedan_id]

    # Fallback: pick a 4-wheel passenger-car-like blueprint and avoid obvious non-sedans.
    excluded_tokens = ('truck', 'bus', 'firetruck', 'ambulance', 'van', 'bike', 'motorcycle')
    sedan_like = []
    for bp in candidates:
        type_id = bp.id.lower()
        if any(token in type_id for token in excluded_tokens):
            continue
        if bp.has_attribute('number_of_wheels') and bp.get_attribute('number_of_wheels').as_int() != 4:
            continue
        sedan_like.append(bp)

    if sedan_like:
        return random.choice(sedan_like)

    return random.choice(candidates)


def geodetic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat * 0.5) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon * 0.5) ** 2
    )
    return earth_radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def choose_spawn_point(
    world_map: carla.Map,
    spawn_points: List[carla.Transform],
    target_lat: float,
    target_lon: float,
    random_spawn: bool,
) -> carla.Transform:
    if random_spawn:
        spawn_point = random.choice(spawn_points)
        geo = world_map.transform_to_geolocation(spawn_point.location)
        print(
            f'[INFO] Random spawn point: x={spawn_point.location.x:.2f} '
            f'y={spawn_point.location.y:.2f} yaw={spawn_point.rotation.yaw:.2f} '
            f'lat={geo.latitude:.8f} lon={geo.longitude:.8f}'
        )
        return spawn_point

    best_spawn = None
    best_geo = None
    best_distance = float('inf')

    for spawn_point in spawn_points:
        geo = world_map.transform_to_geolocation(spawn_point.location)
        distance = geodetic_distance_m(target_lat, target_lon, geo.latitude, geo.longitude)
        if distance < best_distance:
            best_spawn = spawn_point
            best_geo = geo
            best_distance = distance

    if best_spawn is None or best_geo is None:
        raise RuntimeError('Could not choose a vehicle spawn point.')

    print(
        f'[INFO] Nearest spawn point to lat={target_lat:.8f} lon={target_lon:.8f}: '
        f'x={best_spawn.location.x:.2f} y={best_spawn.location.y:.2f} '
        f'yaw={best_spawn.rotation.yaw:.2f} '
        f'lat={best_geo.latitude:.8f} lon={best_geo.longitude:.8f} '
        f'distance={best_distance:.2f}m'
    )
    return best_spawn


def get_active_camera_names(rig_mode: str) -> List[str]:
    if rig_mode == 'single':
        return ['front']
    if rig_mode == 'cardinal':
        return CARDINAL_CAMERA_NAMES
    return DIAGONAL_CAMERA_NAMES


def ensure_dirs(root_dir: str, camera_names: List[str]) -> Dict[str, str]:
    images_root = os.path.join(root_dir, 'images')
    metadata_root = os.path.join(root_dir, 'metadata')
    sparse_root = os.path.join(root_dir, 'sparse', '0')
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(metadata_root, exist_ok=True)
    os.makedirs(sparse_root, exist_ok=True)

    per_camera_dirs = {}
    for name in camera_names:
        camera_dir = os.path.join(images_root, name)
        os.makedirs(camera_dir, exist_ok=True)
        per_camera_dirs[name] = camera_dir

    return {
        'images_root': images_root,
        'metadata_root': metadata_root,
        'sparse_root': sparse_root,
        'poses_json': os.path.join(metadata_root, 'poses.json'),
        'poses_detailed_json': os.path.join(metadata_root, 'poses_detailed.json'),
        'intrinsics_json': os.path.join(metadata_root, 'intrinsics_pinhole.json'),
        'captures_json': os.path.join(metadata_root, 'captures_manifest.json'),
        **{f'image_dir_{k}': v for k, v in per_camera_dirs.items()},
    }


def make_camera_transforms(rig_mode: str) -> Dict[str, carla.Transform]:
    z_height = 2.2

    if rig_mode == 'single':
        return {
            'front': carla.Transform(carla.Location(x=1.6, y=0.0, z=z_height), carla.Rotation(pitch=16.0, yaw=0.0)),
        }

    if rig_mode == 'cardinal':
        return {
            # Front/back-only setup with a slight upward tilt.
            'front': carla.Transform(carla.Location(x=1.6, y=0.0, z=z_height), carla.Rotation(pitch=16.0, yaw=0.0)),
            'back': carla.Transform(carla.Location(x=-1.2, y=0.0, z=z_height), carla.Rotation(pitch=16.0, yaw=180.0)),
        }

    # Default diagonal-corner rig: reduces very-close side facade dominance on narrow roads.
    return {
        'front': carla.Transform(carla.Location(x=1.55, y=-0.55, z=z_height), carla.Rotation(pitch=12.0, yaw=-35.0)),
        'right': carla.Transform(carla.Location(x=1.55, y=0.55, z=z_height), carla.Rotation(pitch=12.0, yaw=35.0)),
        'left': carla.Transform(carla.Location(x=-1.25, y=-0.55, z=z_height), carla.Rotation(pitch=12.0, yaw=-145.0)),
        'back': carla.Transform(carla.Location(x=-1.25, y=0.55, z=z_height), carla.Rotation(pitch=12.0, yaw=145.0)),
    }


def compute_pinhole_intrinsics(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    focal = width / (2.0 * math.tan(fov_deg * math.pi / 360.0))
    return {
        'fx': focal,
        'fy': focal,
        'cx': width / 2.0,
        'cy': height / 2.0,
    }


def write_intrinsics_json(path: str, width: int, height: int, fov_deg: float) -> None:
    intr = compute_pinhole_intrinsics(width, height, fov_deg)
    cameras = [{
        'camera_id': 1,
        'camera_name': 'shared_camera',
        'model': 'PINHOLE',
        'width': width,
        'height': height,
        'params': [intr['fx'], intr['fy'], intr['cx'], intr['cy']],
        'fov_deg': fov_deg,
    }]

    payload = {
        'format': 'colmap_hloc_pinhole',
        'note': 'CARLA pinhole cameras using one shared camera model/intrinsics',
        'formula': 'focal = width / (2 * tan(FOV * pi / 360)); fx=fy=focal; cx=width/2; cy=height/2',
        'shared_camera': True,
        'cameras': cameras,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def image_to_rgb_array(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1]


def get_image_sensor_transform(image: carla.Image, sensor: carla.Sensor) -> carla.Transform:
    image_transform = getattr(image, 'transform', None)
    if image_transform is not None:
        return image_transform
    return sensor.get_transform()


def apply_vehicle_control(vehicle: carla.Vehicle) -> None:
    keys = pygame.key.get_pressed()
    control = carla.VehicleControl()

    if keys[K_w]:
        control.reverse = False
        control.throttle = 0.6
    elif keys[K_s]:
        control.reverse = True
        control.throttle = 0.45

    if keys[K_a]:
        control.steer = -0.5
    elif keys[K_d]:
        control.steer = 0.5

    vehicle.apply_control(control)


def rotmat_to_quaternion_wxyz(rot: np.ndarray) -> List[float]:
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def carla_transform_to_converted_pose(tf: carla.Transform) -> Dict[str, object]:
    t_cw_lh = np.array(tf.get_matrix(), dtype=np.float64)
    r_cw_lh = t_cw_lh[:3, :3]
    p_cw_lh = t_cw_lh[:3, 3]

    # CARLA left-handed (x forward, y right, z up) ->
    # right-handed camera-style (x right, y down, z forward).
    s = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    r_cw_rh = s @ r_cw_lh @ np.linalg.inv(s)
    p_cw_rh = s @ p_cw_lh

    t_cw_rh = np.eye(4, dtype=np.float64)
    t_cw_rh[:3, :3] = r_cw_rh
    t_cw_rh[:3, 3] = p_cw_rh

    r_wc_rh = r_cw_rh.T
    p_wc_rh = -r_wc_rh @ p_cw_rh

    t_wc_rh = np.eye(4, dtype=np.float64)
    t_wc_rh[:3, :3] = r_wc_rh
    t_wc_rh[:3, 3] = p_wc_rh

    qw, qx, qy, qz = rotmat_to_quaternion_wxyz(r_wc_rh)

    return {
        'carla_left_handed': {
            'matrix_camera_to_world': t_cw_lh.tolist(),
            'location_xyz_m': [float(tf.location.x), float(tf.location.y), float(tf.location.z)],
            'rotation_rpy_deg': [float(tf.rotation.roll), float(tf.rotation.pitch), float(tf.rotation.yaw)],
        },
        'converted_right_handed': {
            'axes': {
                'x': 'right',
                'y': 'down',
                'z': 'forward',
            },
            'conversion_matrix_S': s.tolist(),
            'matrix_camera_to_world': t_cw_rh.tolist(),
            'matrix_world_to_camera': t_wc_rh.tolist(),
            'quaternion_wxyz': [float(qw), float(qx), float(qy), float(qz)],
            'translation_xyz_m': [float(p_wc_rh[0]), float(p_wc_rh[1]), float(p_wc_rh[2])],
        },
    }


def save_json(path: str, payload: object) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def make_simple_colmap_pose_record(
    world: carla.World,
    capture_id: int,
    world_frame: int,
    timestamp: float,
    camera_name: str,
    camera_id: int,
    image_rel: str,
    image_frame: int,
    camera_tf: carla.Transform,
) -> Dict[str, object]:
    converted = carla_transform_to_converted_pose(camera_tf)['converted_right_handed']
    geo = world.get_map().transform_to_geolocation(camera_tf.location)

    return {
        'capture_id': capture_id,
        'world_frame': world_frame,
        'timestamp_sec': timestamp,
        'camera_name': camera_name,
        'camera_id': camera_id,
        'camera_model': 'PINHOLE',
        'image_name': image_rel,
        'image_frame': image_frame,
        # Geodetic reference for downstream GIS alignment.
        'lat': float(geo.latitude),
        'lon': float(geo.longitude),
        'alt': float(geo.altitude),
        # Raw CARLA orientation for easy inspection and debug.
        'roll_deg': float(camera_tf.rotation.roll),
        'pitch_deg': float(camera_tf.rotation.pitch),
        'yaw_deg': float(camera_tf.rotation.yaw),
        # COLMAP images.txt compatible extrinsics (world -> camera).
        'qw': float(converted['quaternion_wxyz'][0]),
        'qx': float(converted['quaternion_wxyz'][1]),
        'qy': float(converted['quaternion_wxyz'][2]),
        'qz': float(converted['quaternion_wxyz'][3]),
        'tx': float(converted['translation_xyz_m'][0]),
        'ty': float(converted['translation_xyz_m'][1]),
        'tz': float(converted['translation_xyz_m'][2]),
    }


def main() -> None:
    args = parse_args()
    active_camera_names = get_active_camera_names(args.camera_rig)
    effective_fov = args.fov if args.fov is not None else (120.0 if args.camera_rig == 'cardinal' else 95.0)
    width, height = args.res
    if args.display_scale <= 0.0:
        raise ValueError('--display-scale must be greater than 0')
    if args.preview_every <= 0:
        raise ValueError('--preview-every must be greater than 0')
    if args.preview_camera not in active_camera_names:
        raise ValueError(f'--preview-camera must be one of: {", ".join(active_camera_names)}')
    preview_width = max(1, int(width * args.display_scale))
    preview_height = max(1, int(height * args.display_scale))

    out = ensure_dirs(args.output, active_camera_names)
    write_intrinsics_json(out['intrinsics_json'], width, height, effective_fov)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    actors: List[carla.Actor] = []
    sensors: Dict[str, carla.Sensor] = {}
    latest_images: Dict[str, carla.Image] = {}

    display = None
    capture_records = []
    pose_records = []
    pose_records_detailed = []

    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = pick_vehicle_blueprint(bp_lib, args.vehicle_filter, args.vehicle_type)
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('No spawn points available in current map.')

        spawn_point = choose_spawn_point(
            world.get_map(),
            spawn_points,
            args.spawn_lat,
            args.spawn_lon,
            args.random_spawn,
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors.append(vehicle)
        print(f'[INFO] Spawned vehicle: {vehicle.type_id} (vehicle_type={args.vehicle_type})')

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(effective_fov))

        camera_transforms = make_camera_transforms(args.camera_rig)
        print(f'[INFO] Camera rig mode: {args.camera_rig} | cameras={active_camera_names} | fov={effective_fov}')
        for name in active_camera_names:
            sensor = world.spawn_actor(cam_bp, camera_transforms[name], attach_to=vehicle)
            sensors[name] = sensor
            actors.append(sensor)
            sensor.listen(lambda data, camera_name=name: latest_images.__setitem__(camera_name, data))

        pygame.init()
        pygame.font.init()
        if args.preview_layout == 'single':
            display = pygame.display.set_mode((preview_width, preview_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        elif len(active_camera_names) == 2:
            display = pygame.display.set_mode((preview_width * 2, preview_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            display = pygame.display.set_mode((preview_width * 2, preview_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA Async HLoc Capture')
        font = pygame.font.Font(None, 26)
        clock = pygame.time.Clock()

        running = True
        capture_requested = False
        capture_id = 0
        preview_counter = 0

        print(f'Controls: W/A/S/D to drive, SPACE to capture latest {len(active_camera_names)} camera image(s), Q or ESC to quit.')

        while running:
            clock.tick(30)
            preview_counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_SPACE:
                        capture_requested = True

            apply_vehicle_control(vehicle)

            if preview_counter % args.preview_every == 0:
                if args.preview_layout == 'single':
                    tile_order = [(args.preview_camera, (0, 0))]
                elif len(active_camera_names) == 2:
                    tile_order = [
                        ('front', (0, 0)),
                        ('back', (preview_width, 0)),
                    ]
                else:
                    tile_order = [
                        ('front', (0, 0)),
                        ('right', (preview_width, 0)),
                        ('left', (0, preview_height)),
                        ('back', (preview_width, preview_height)),
                    ]
                for name, (ox, oy) in tile_order:
                    image = latest_images.get(name)
                    if image is None:
                        continue

                    rgb_array = image_to_rgb_array(image)
                    surface = pygame.surfarray.make_surface(np.ascontiguousarray(rgb_array.swapaxes(0, 1)))
                    if args.display_scale != 1.0:
                        scale_fn = pygame.transform.smoothscale if args.smooth_preview else pygame.transform.scale
                        surface = scale_fn(surface, (preview_width, preview_height))
                    display.blit(surface, (ox, oy))

                preview_image = latest_images.get(args.preview_camera)
                preview_frame = preview_image.frame if preview_image is not None else 'waiting'
                hud_text = f'{args.preview_camera if args.preview_layout == "single" else args.preview_layout} | Frame: {preview_frame} | Captures: {capture_id} | SPACE: capture'
                hud_surface = font.render(hud_text, True, (255, 255, 255))
                display.blit(hud_surface, (10, 10))
                pygame.display.flip()

            if not capture_requested:
                continue

            snapshot = world.get_snapshot()
            current_world_frame = int(snapshot.frame)
            current_timestamp = float(snapshot.timestamp.elapsed_seconds)
            frame_images = {}
            missing_cameras = []
            for name in active_camera_names:
                image = latest_images.get(name)
                if image is None:
                    missing_cameras.append(name)
                else:
                    frame_images[name] = image

            if missing_cameras:
                print(f'[WARN] Capture skipped; waiting for camera image(s): {", ".join(missing_cameras)}')
                capture_requested = False
                continue

            capture_world_frame = int(frame_images[active_camera_names[0]].frame)
            capture_timestamp = float(frame_images[active_camera_names[0]].timestamp)
            capture_id += 1
            capture_tag = f'{capture_id:06d}'
            image_files = {}
            per_camera_pose = {}

            for camera_index, name in enumerate(active_camera_names, start=1):
                image = frame_images[name]
                image_name = f'{capture_tag}_{name}_f{image.frame:08d}.png'
                image_rel = f'images/{name}/{image_name}'
                image_abs = os.path.join(args.output, image_rel)
                image.save_to_disk(image_abs)
                image_files[name] = image_rel

                camera_tf = get_image_sensor_transform(image, sensors[name])

                per_camera_pose[name] = {
                    'camera_id': 1,
                    'image': image_rel,
                    'frame': int(image.frame),
                    'timestamp_sec': float(image.timestamp),
                    **carla_transform_to_converted_pose(camera_tf),
                }

                pose_records.append(
                    make_simple_colmap_pose_record(
                        world=world,
                        capture_id=capture_id,
                        world_frame=int(image.frame),
                        timestamp=float(image.timestamp),
                        camera_name=name,
                        camera_id=1,
                        image_rel=image_rel,
                        image_frame=int(image.frame),
                        camera_tf=camera_tf,
                    )
                )

            capture_records.append({
                'capture_id': capture_id,
                'world_frame': capture_world_frame,
                'timestamp_sec': capture_timestamp,
                'current_world_frame_at_keypress': current_world_frame,
                'current_timestamp_sec_at_keypress': current_timestamp,
                'images': image_files,
            })

            pose_records_detailed.append({
                'capture_id': capture_id,
                'world_frame': capture_world_frame,
                'timestamp_sec': capture_timestamp,
                'current_world_frame_at_keypress': current_world_frame,
                'current_timestamp_sec_at_keypress': current_timestamp,
                'cameras': per_camera_pose,
            })

            save_json(out['captures_json'], capture_records)
            save_json(out['poses_json'], pose_records)
            save_json(out['poses_detailed_json'], pose_records_detailed)

            print(
                f'[CAPTURE] id={capture_id} image_frame={capture_world_frame} '
                f'current_world_frame={current_world_frame} saved {len(active_camera_names)} images + poses'
            )
            capture_requested = False

    finally:
        if display is not None:
            pygame.quit()

        for sensor in sensors.values():
            if sensor is not None:
                sensor.stop()

        for actor in actors:
            if actor is not None:
                actor.destroy()


if __name__ == '__main__':
    main()
