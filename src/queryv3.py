from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

from hloc.utils.io import read_image

from simulation_pose_utils import (
    load_json,
    load_pose_records,
    metadata_pose_center,
    quat_wxyz_to_rotmat,
)


JETSON_SERVER_DIR = Path(__file__).resolve().parent / 'JetsonServer'
if str(JETSON_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(JETSON_SERVER_DIR))

from nano_localizer import COLMAP_TO_CARLA_S, LiveCarlaLocalizer  # noqa: E402


# Dataset with query metadata.
dataset_root = Path('../datasets/test_dataset_108_24_may')
intrinsics_json = dataset_root / 'metadata/intrinsics_pinhole.json'

# Query image folder. Can contain any camera as long as basenames exist in poses.json.
query_source_dir = dataset_root / 'test'
query_glob = '*.png'
max_queries = 200
random_seed = 42

# Known-pose map bundle created by reconstruction_known_pose.ipynb.
map_bundle_root = Path('../datasets/train_dataset_585_24_may-bundle')
sfm_model_root = map_bundle_root / 'sfm'
db_features = map_bundle_root / 'features.h5'
db_global_features = map_bundle_root / 'global-feats-netvlad.h5'

# Query outputs.
results_dir = map_bundle_root / 'query_batch_results_v4'
results_dir.mkdir(parents=True, exist_ok=True)

# Localization parameters. These mirror src/JetsonServer/server.py defaults.
num_loc = 5
max_error = 12
min_inliers = 15
retrieval_max_size = 512
local_max_size = 640
max_keypoints = 768
netvlad_dtype = 'auto'
reference_feature_cache_size = 16
device = None


def wrap_angle_deg(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0


def heading_error_deg(est_heading, gt_heading):
    return abs(wrap_angle_deg(float(est_heading) - float(gt_heading)))


def colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh):
    r_cw_rh = np.asarray(r_wc_rh, dtype=np.float64).T
    r_cw_lh = COLMAP_TO_CARLA_S @ r_cw_rh @ np.linalg.inv(COLMAP_TO_CARLA_S)
    return wrap_angle_deg(np.degrees(np.arctan2(r_cw_lh[1, 0], r_cw_lh[0, 0])))


def metadata_record_to_carla_yaw_deg(record):
    r_wc_rh = quat_wxyz_to_rotmat([record['qw'], record['qx'], record['qy'], record['qz']])
    return colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh)


def metadata_record_to_carla_center(record):
    center_colmap = metadata_pose_center(record)
    return COLMAP_TO_CARLA_S @ center_colmap


def carla_position_error_m(estimated, ground_truth):
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    return float(np.linalg.norm(estimated - ground_truth))


gpu_available = torch.cuda.is_available()
print(f'CUDA available: {gpu_available}')
if gpu_available:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: No GPU detected. Running on CPU will be significantly slower.')

for path in [query_source_dir, sfm_model_root, db_features, db_global_features]:
    if not path.exists():
        raise FileNotFoundError(path)

intrinsics_cfg = load_json(intrinsics_json)['cameras'][0]
print(f'Dataset: {dataset_root}')
print(f'Query dir: {query_source_dir}')
print(f'Map bundle: {map_bundle_root}')
print(f'SfM model: {sfm_model_root}')
print(f'Intrinsics: {intrinsics_cfg["model"]} {intrinsics_cfg["width"]}x{intrinsics_cfg["height"]} params={intrinsics_cfg["params"]}')

all_pose_records, pose_by_name, _ = load_pose_records(dataset_root, camera_names=None)
query_paths_all = sorted(p for p in query_source_dir.glob(query_glob) if p.is_file())
if max_queries is not None and max_queries < len(query_paths_all):
    rng = random.Random(random_seed)
    query_paths = sorted(rng.sample(query_paths_all, max_queries))
else:
    query_paths = query_paths_all

missing_metadata = [p.name for p in query_paths if p.name not in pose_by_name]
if missing_metadata:
    raise ValueError(f'{len(missing_metadata)} query images have no poses.json metadata. Examples: {missing_metadata[:10]}')

query_camera_distribution = {}
for path in query_paths:
    cam = pose_by_name[path.name]['camera_name']
    query_camera_distribution[cam] = query_camera_distribution.get(cam, 0) + 1

print(f'All query images found: {len(query_paths_all)}')
print(f'Queries selected: {len(query_paths)}')
print(f'Random seed: {random_seed if max_queries is not None else None}')
print(f'Query camera distribution: {query_camera_distribution}')

metadata_yaw_errors = []
for query_path in query_paths:
    rec = pose_by_name[query_path.name]
    metadata_yaw_errors.append(heading_error_deg(metadata_record_to_carla_yaw_deg(rec), rec['yaw_deg']))
metadata_yaw_errors = np.array(metadata_yaw_errors, dtype=np.float64)
print('Metadata quaternion -> CARLA yaw sanity check')
print(f'  count: {len(metadata_yaw_errors)}')
print(f'  mean error: {metadata_yaw_errors.mean():.9f} deg')
print(f'  max error: {metadata_yaw_errors.max():.9f} deg')
if metadata_yaw_errors.max() >= 1e-3:
    raise AssertionError('Metadata quaternion to CARLA yaw conversion sanity check failed.')

print('\nLoading live-server localizer once...')
localizer = LiveCarlaLocalizer(
    bundle_root=str(map_bundle_root),
    num_loc=num_loc,
    max_error=max_error,
    min_inliers=min_inliers,
    retrieval_max_size=retrieval_max_size,
    local_max_size=local_max_size,
    max_keypoints=max_keypoints,
    device=device,
    netvlad_dtype=netvlad_dtype,
    reference_feature_cache_size=reference_feature_cache_size,
)

localization_results = []
inference_times_ms = []

print('\nRunning server-style in-memory batch localization...')
try:
    for idx, query_path in enumerate(query_paths, start=1):
        query_basename = query_path.name
        query_gt = pose_by_name[query_basename]
        gt_center_carla = metadata_record_to_carla_center(query_gt)

        print(f'\n[{idx}/{len(query_paths)}] {query_basename} camera={query_gt["camera_name"]}')

        try:
            rgb = read_image(query_path, grayscale=False)
            start_time = time.perf_counter()
            loc = localizer.localize_xyz_heading(rgb)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            result = {
                'image_name': query_basename,
                'camera_name': query_gt['camera_name'],
                'capture_id': int(query_gt['capture_id']),
                'success': bool(loc.get('success', False)),
                'pose_estimated': bool(loc.get('pose_estimated', False)),
                'num_inliers': int(loc.get('num_inliers', 0)),
                'num_correspondences': int(loc.get('num_correspondences', 0)),
                'retrieved': loc.get('retrieved', []),
                'inference_time_ms': elapsed_ms,
                'timings_ms': loc.get('timings_ms', {}),
            }

            if not loc.get('success'):
                result['error'] = loc.get('error', 'unknown localization error')
                localization_results.append(result)
                print(
                    f"  failed error={result['error']} "
                    f"corr={result['num_correspondences']} time={elapsed_ms:.2f} ms"
                )
                continue

            estimated_center_carla = np.array([loc['x'], loc['y'], loc['z']], dtype=np.float64)
            position_error_m = carla_position_error_m(estimated_center_carla, gt_center_carla)
            estimated_heading = float(loc['heading_deg'])
            gt_heading = float(query_gt['yaw_deg'])
            heading_error = heading_error_deg(estimated_heading, gt_heading)

            result.update({
                'position_error_m': position_error_m,
                'estimated_center_x': float(estimated_center_carla[0]),
                'estimated_center_y': float(estimated_center_carla[1]),
                'estimated_center_z': float(estimated_center_carla[2]),
                'gt_center_x': float(gt_center_carla[0]),
                'gt_center_y': float(gt_center_carla[1]),
                'gt_center_z': float(gt_center_carla[2]),
                'estimated_heading_deg': estimated_heading,
                'ground_truth_yaw_deg': gt_heading,
                'heading_error_deg': float(heading_error),
                'inliers_by_ref': loc.get('inliers_by_ref', {}),
                'match_stats': loc.get('match_stats', []),
            })
            localization_results.append(result)
            inference_times_ms.append(elapsed_ms)
            print(
                f"  success corr={result['num_correspondences']} "
                f"inliers={result['num_inliers']} pos_err={position_error_m:.3f} m "
                f"heading_err={heading_error:.2f} deg time={elapsed_ms:.2f} ms"
            )

        except Exception as exc:
            print(f'  error: {exc}')
            localization_results.append({
                'image_name': query_basename,
                'camera_name': query_gt['camera_name'],
                'capture_id': int(query_gt['capture_id']),
                'success': False,
                'error': str(exc),
            })
finally:
    localizer.shutdown()

print('\nBatch localization finished')

results_json = results_dir / 'queryv3_server_style_results.json'
results_csv = results_dir / 'queryv3_server_style_results.csv'
results_json.write_text(json.dumps(localization_results, indent=2), encoding='utf-8')
pd.DataFrame(localization_results).to_csv(results_csv, index=False)
print(f'Results JSON: {results_json}')
print(f'Results CSV: {results_csv}')

if inference_times_ms:
    mean_time = np.mean(inference_times_ms)
    std_time = np.std(inference_times_ms)
    min_time = np.min(inference_times_ms)
    max_time = np.max(inference_times_ms)
    print('\n--- Inference Time Metrics (server-style full pipeline per image) ---')
    print(f'Mean Inference Time: {mean_time:.2f} ms')
    print(f'Std Deviation:       {std_time:.2f} ms')
    print(f'Min / Max Time:      {min_time:.2f} ms / {max_time:.2f} ms')
