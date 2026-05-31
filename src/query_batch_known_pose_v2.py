#!/usr/bin/env python
# coding: utf-8

# # Batch Query Test Against Known-Pose Map
# 
# This notebook localizes test/query images against the known-pose reconstruction produced by `reconstruction_known_pose.ipynb`. Query images can come from any camera as long as their basenames exist in `metadata/poses.json`.
# 
# The map and query cameras do not need to share the same camera_name. Ground truth is matched by image basename.

# ## 1. Configuration

# In[ ]:


from pathlib import Path
from datetime import datetime
import copy
import gc
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import pycolmap
import torch

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.parsers import parse_retrieval

from simulation_pose_utils import (
    image_basename,
    load_json,
    load_pose_records,
    metadata_pose_center,
    pycolmap_transform_rt,
    quat_wxyz_to_rotmat,
)

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


# Dataset with query metadata.
dataset_root = Path('../datasets/test_dataset_108_24_May')
intrinsics_json = dataset_root / 'metadata/intrinsics_pinhole.json'

# Query image folder. Can contain any camera as long as basenames exist in poses.json.
query_source_dir = dataset_root / 'test'
query_glob = '*.png'
max_queries = int(os.environ.get('VL_MAX_QUERIES', '200'))
random_seed = 42

# Known-pose map bundle created by reconstruction_known_pose.ipynb.
map_bundle_root = Path('../datasets/train_dataset_585_24_may-bundle')
sfm_model_root = map_bundle_root / 'sfm'
db_features = map_bundle_root / 'features.h5'
db_global_features = map_bundle_root / 'global-feats-netvlad.h5'

# Query outputs.
results_dir = map_bundle_root / 'query_batch_results_v2_26_may-nano'
query_cache_dir = map_bundle_root / 'query_batch_v2_26_may_cache-nano'
results_dir.mkdir(parents=True, exist_ok=True)
query_cache_dir.mkdir(parents=True, exist_ok=True)

# Localization parameters.
num_loc = int(os.environ.get('VL_NUM_LOC', '5'))
max_error = 12
overwrite_query_features = env_flag('VL_OVERWRITE_QUERY_FEATURES', False)
extract_device = os.environ.get('VL_EXTRACT_DEVICE', 'cpu').strip().lower()
if extract_device not in {'cpu', 'cuda', 'auto'}:
    raise ValueError('VL_EXTRACT_DEVICE must be one of: cpu, cuda, auto')

feature_conf = copy.deepcopy(extract_features.confs['superpoint_max'])
feature_conf['model']['max_keypoints'] = int(os.environ.get('VL_MAX_KEYPOINTS', '768'))
feature_conf['preprocessing']['resize_max'] = int(os.environ.get('VL_LOCAL_RESIZE_MAX', '640'))

retrieval_conf = copy.deepcopy(extract_features.confs['netvlad'])
retrieval_conf['preprocessing']['resize_max'] = int(os.environ.get('VL_RETRIEVAL_RESIZE_MAX', '512'))

matcher_conf = copy.deepcopy(match_features.confs['superpoint+lightglue'])

torch.set_num_threads(int(os.environ.get('VL_TORCH_THREADS', '1')))

# Windows/sandbox-safe HLoc execution: avoid multiprocessing DataLoader workers.
# Keep class-compatible with Kornia's DataLoader[Any] annotation.
_original_dataloader = torch.utils.data.DataLoader
class _SingleProcessDataLoader(_original_dataloader):
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, *args, **kwargs):
        kwargs['num_workers'] = 0
        kwargs['pin_memory'] = False
        super().__init__(*args, **kwargs)
torch.utils.data.DataLoader = _SingleProcessDataLoader

for path in [query_source_dir, sfm_model_root, db_features, db_global_features]:
    if not path.exists():
        raise FileNotFoundError(path)

intrinsics_cfg = load_json(intrinsics_json)['cameras'][0]
print(f'Dataset: {dataset_root}')
print(f'Query dir: {query_source_dir}')
print(f'Map bundle: {map_bundle_root}')
print(f'SfM model: {sfm_model_root}')
print(f'Intrinsics: {intrinsics_cfg["model"]} {intrinsics_cfg["width"]}x{intrinsics_cfg["height"]} params={intrinsics_cfg["params"]}')
print(
    'Nano profile: '
    f'num_loc={num_loc}, '
    f'local_resize_max={feature_conf["preprocessing"]["resize_max"]}, '
    f'max_keypoints={feature_conf["model"]["max_keypoints"]}, '
	    f'retrieval_resize_max={retrieval_conf["preprocessing"]["resize_max"]}, '
	    f'torch_threads={torch.get_num_threads()}, '
	    f'extract_device={extract_device}, '
	    f'overwrite_query_features={overwrite_query_features}'
	)

# ## 2. Load Map and Query Ground Truth

# In[3]:


model = pycolmap.Reconstruction(sfm_model_root)
print(model.summary())

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
for p in query_paths:
    cam = pose_by_name[p.name]['camera_name']
    query_camera_distribution[cam] = query_camera_distribution.get(cam, 0) + 1

references = sorted(image.name for image in model.images.values())
print(f'All query images found: {len(query_paths_all)}')
print(f'Queries selected: {len(query_paths)}')
print(f'Random seed: {random_seed if max_queries is not None else None}')
print(f'Query camera distribution: {query_camera_distribution}')
print(f'Reference images in map: {len(references)}')

# ## 3. Helper Functions

# In[4]:


def extract_on_cpu(conf, image_root, image_list, feature_path, overwrite=True):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return extract_features.main(
            conf,
            image_root,
            image_list=image_list,
            feature_path=feature_path,
            overwrite=overwrite,
        )
    finally:
        torch.cuda.is_available = original


def extract_query_features(conf, image_root, image_list, feature_path, overwrite=True):
    if extract_device == 'cpu':
        return extract_on_cpu(conf, image_root, image_list, feature_path, overwrite=overwrite)

    if extract_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('VL_EXTRACT_DEVICE=cuda requested but CUDA is not available')

    return extract_features.main(
        conf,
        image_root,
        image_list=image_list,
        feature_path=feature_path,
        overwrite=overwrite,
    )


def resolve_reference_ids(model, names):
    ref_ids = []
    missing = []
    for name in names:
        image = model.find_image_with_name(name)
        if image is None:
            basename = image_basename(name)
            for candidate in model.images.values():
                if image_basename(candidate.name) == basename:
                    image = candidate
                    break
        if image is None:
            missing.append(name)
        else:
            ref_ids.append(image.image_id)
    if missing:
        raise ValueError('Retrieved images not registered in model: ' + ', '.join(missing))
    return ref_ids


CARLA_TO_COLMAP_S = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)
COLMAP_TO_CARLA_S = np.linalg.inv(CARLA_TO_COLMAP_S)


def wrap_angle_deg(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0


def heading_error_deg(est_heading, gt_heading):
    return abs(wrap_angle_deg(float(est_heading) - float(gt_heading)))


def colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh):
    # PnP/pycolmap returns COLMAP right-handed world-to-camera rotation.
    # Convert it back to CARLA left-handed camera-to-world rotation before extracting yaw.
    r_cw_rh = np.asarray(r_wc_rh, dtype=np.float64).T
    r_cw_lh = COLMAP_TO_CARLA_S @ r_cw_rh @ CARLA_TO_COLMAP_S
    return wrap_angle_deg(np.degrees(np.arctan2(r_cw_lh[1, 0], r_cw_lh[0, 0])))


def metadata_record_to_carla_yaw_deg(record):
    r_wc_rh = quat_wxyz_to_rotmat([record['qw'], record['qx'], record['qy'], record['qz']])
    return colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh)


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


localizer_conf = {
    'estimation': {'ransac': {'max_error': max_error}},
    'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
}


class CompatQueryLocalizer(QueryLocalizer):
    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera, **kwargs):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        if points2D.shape[0] == 0:
            return None

        if hasattr(pycolmap, 'estimate_and_refine_absolute_pose'):
            return pycolmap.estimate_and_refine_absolute_pose(
                points2D,
                points3D,
                query_camera,
                estimation_options=self.config.get('estimation', {}),
                refinement_options=self.config.get('refinement', {}),
            )

        if hasattr(pycolmap, 'absolute_pose_estimation'):
            ret = pycolmap.absolute_pose_estimation(
                points2D,
                points3D,
                query_camera,
                estimation_options=self.config.get('estimation', {}),
            )
            if isinstance(ret, dict) and ret.get('success') is False:
                return None
            return ret

        raise RuntimeError(
            'Unsupported pycolmap version: missing estimate_and_refine_absolute_pose '
            'and absolute_pose_estimation'
        )


localizer = CompatQueryLocalizer(model, localizer_conf)


def qvec_tvec_to_rt(qvec, tvec):
    return (
        quat_wxyz_to_rotmat(np.asarray(qvec, dtype=np.float64).reshape(4)),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )


def pose_result_to_rt(ret):
    if isinstance(ret, dict):
        if 'cam_from_world' in ret:
            return pycolmap_transform_rt(ret['cam_from_world'])
        if 'qvec' in ret and 'tvec' in ret:
            return qvec_tvec_to_rt(ret['qvec'], ret['tvec'])
    if hasattr(ret, 'cam_from_world'):
        cam_from_world = ret.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        return pycolmap_transform_rt(cam_from_world)
    if hasattr(ret, 'qvec') and hasattr(ret, 'tvec'):
        return qvec_tvec_to_rt(ret.qvec, ret.tvec)
    raise RuntimeError(f'Unsupported pose result format: {type(ret)}')


def pose_result_num_inliers(ret):
    if isinstance(ret, dict):
        if 'num_inliers' in ret:
            return int(ret['num_inliers'])
        if 'inliers' in ret:
            return int(np.count_nonzero(ret['inliers']))
    if hasattr(ret, 'num_inliers'):
        return int(ret.num_inliers)
    if hasattr(ret, 'inliers'):
        return int(np.count_nonzero(ret.inliers))
    return 0


camera = pycolmap.Camera(
    model=intrinsics_cfg['model'],
    width=int(intrinsics_cfg['width']),
    height=int(intrinsics_cfg['height']),
    params=np.array(intrinsics_cfg['params'], dtype=float),
)

# ## 4. Batch Query Localization

# In[5]:


localization_results = []

for idx, query_path in enumerate(query_paths, start=1):
    query_basename = query_path.name
    query_gt = pose_by_name[query_basename]
    query_gt_center = metadata_pose_center(query_gt)
    query_dst = query_cache_dir / query_basename
    shutil.copy2(query_path, query_dst)
    query_rel = f'{query_cache_dir.name}/{query_basename}'

    print(f'\n[{idx}/{len(query_paths)}] {query_basename} camera={query_gt["camera_name"]}')

    stem = Path(query_basename).stem
    query_features = results_dir / f'{stem}-features.h5'
    query_global_features = results_dir / f'{stem}-global-feats-netvlad.h5'
    query_matches = results_dir / f'{stem}-matches.h5'
    loc_pairs = results_dir / f'{stem}-pairs-query-netvlad.txt'

    try:
        extract_query_features(retrieval_conf, map_bundle_root, [query_rel], query_global_features, overwrite=overwrite_query_features)
        extract_query_features(feature_conf, map_bundle_root, [query_rel], query_features, overwrite=overwrite_query_features)

        pairs_from_retrieval.main(
            descriptors=query_global_features,
            output=loc_pairs,
            num_matched=min(num_loc, len(references)),
            query_list=[query_rel],
            db_list=references,
            db_descriptors=db_global_features,
        )

        match_features.main(
            matcher_conf,
            loc_pairs,
            features=query_features,
            features_ref=db_features,
            matches=query_matches,
            overwrite=True,
        )

        retrieval_dict = parse_retrieval(loc_pairs)
        retrieved_names = retrieval_dict[query_rel]
        ref_ids = resolve_reference_ids(model, retrieved_names)

        ret, log = pose_from_cluster(localizer, query_rel, camera, ref_ids, query_features, query_matches)
        if ret is None:
            print('  pose estimation failed')
            localization_results.append({
                'image_name': query_basename,
                'camera_name': query_gt['camera_name'],
                'success': False,
                'error': 'pose_from_cluster returned None',
                'retrieved': retrieved_names,
            })
            continue

        R_wc_rh, tvec = pose_result_to_rt(ret)
        estimated_center = -R_wc_rh.T @ tvec
        position_error_m = float(np.linalg.norm(estimated_center - query_gt_center))

        estimated_heading = colmap_world_to_camera_to_carla_yaw_deg(R_wc_rh)
        gt_heading = float(query_gt['yaw_deg'])
        heading_error = heading_error_deg(estimated_heading, gt_heading)

        result = {
            'image_name': query_basename,
            'camera_name': query_gt['camera_name'],
            'capture_id': int(query_gt['capture_id']),
            'success': True,
            'num_inliers': pose_result_num_inliers(ret),
            'num_matches': int(log.get('num_matches', 0)),
            'position_error_m': position_error_m,
            'estimated_center_x': float(estimated_center[0]),
            'estimated_center_y': float(estimated_center[1]),
            'estimated_center_z': float(estimated_center[2]),
            'gt_center_x': float(query_gt_center[0]),
            'gt_center_y': float(query_gt_center[1]),
            'gt_center_z': float(query_gt_center[2]),
            'estimated_heading_deg': estimated_heading,
            'ground_truth_yaw_deg': gt_heading,
            'heading_error_deg': float(heading_error),
            'retrieved': retrieved_names,
        }
        localization_results.append(result)
        print(
            f"  success matches={result['num_matches']} inliers={result['num_inliers']} "
            f"pos_err={position_error_m:.3f} m heading_err={heading_error:.2f} deg"
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print('\nBatch localization finished')

# ## 5. Results Summary

# In[6]:


results_df = pd.DataFrame(localization_results)
for column in [
    'image_name',
    'camera_name',
    'capture_id',
    'success',
    'error',
    'num_inliers',
    'num_matches',
    'position_error_m',
    'heading_error_deg',
]:
    if column not in results_df.columns:
        results_df[column] = np.nan
display(results_df)

successful = results_df[results_df['success'] == True]
failed = results_df[results_df['success'] != True]

summary = {
    'timestamp': datetime.now().isoformat(timespec='seconds'),
    'dataset_root': str(dataset_root),
    'query_source_dir': str(query_source_dir),
    'map_bundle_root': str(map_bundle_root),
    'sfm_model_root': str(sfm_model_root),
    'total_queries': int(len(results_df)),
    'successful_queries': int(len(successful)),
    'failed_queries': int(len(failed)),
    'success_rate': float(len(successful) / max(1, len(results_df))),
    'num_retrieved': num_loc,
    'ransac_max_error_px': max_error,
}

if len(successful) > 0:
    summary['position_error_m'] = {
        'mean': float(successful['position_error_m'].mean()),
        'median': float(successful['position_error_m'].median()),
        'max': float(successful['position_error_m'].max()),
    }
    summary['inliers'] = {
        'mean': float(successful['num_inliers'].mean()),
        'median': float(successful['num_inliers'].median()),
        'min': int(successful['num_inliers'].min()),
    }
    if 'num_matches' in successful:
        summary['matches'] = {
            'mean': float(successful['num_matches'].mean()),
            'median': float(successful['num_matches'].median()),
            'min': int(successful['num_matches'].min()),
        }
    if 'heading_error_deg' in successful:
        summary['heading_error_deg'] = {
            'mean': float(successful['heading_error_deg'].mean()),
            'median': float(successful['heading_error_deg'].median()),
            'max': float(successful['heading_error_deg'].max()),
        }

print(json.dumps(summary, indent=2))

details_csv = results_dir / 'query_batch_details.csv'
summary_json = results_dir / 'query_batch_summary.json'
results_df.to_csv(details_csv, index=False)
summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print(f'Details CSV: {details_csv}')
print(f'Summary JSON: {summary_json}')

# ## 6. VisualLocalization.net Threshold Format

# In[7]:


benchmark_thresholds = [
    (0.25, 2.0),
    (0.50, 5.0),
    (5.00, 10.0),
]

total_queries = len(results_df)
successful_eval = successful.dropna(subset=['position_error_m', 'heading_error_deg']).copy()

benchmark_parts = []
benchmark_rows = []
for pos_thr, rot_thr in benchmark_thresholds:
    passed = successful_eval[
        (successful_eval['position_error_m'] <= pos_thr)
        & (successful_eval['heading_error_deg'] <= rot_thr)
    ]
    count = int(len(passed))
    percent_total = 100.0 * count / max(1, total_queries)
    percent_successful = 100.0 * count / max(1, len(successful_eval))
    benchmark_parts.append(f'{percent_total:.1f}')
    benchmark_rows.append({
        'position_threshold_m': pos_thr,
        'heading_threshold_deg': rot_thr,
        'passed': count,
        'total_queries': int(total_queries),
        'successful_evaluated_queries': int(len(successful_eval)),
        'percent_of_all_queries': percent_total,
        'percent_of_successful_queries': percent_successful,
    })

benchmark_df = pd.DataFrame(benchmark_rows)
print('VisualLocalization.net style localization scores')
print('All conditions: (0.25m, 2 deg) / (0.5m, 5 deg) / (5m, 10 deg)')
print('All conditions: ' + ' / '.join(benchmark_parts))
display(benchmark_df)

benchmark_json = results_dir / 'query_batch_visual_localization_thresholds.json'
benchmark_json.write_text(json.dumps(benchmark_rows, indent=2), encoding='utf-8')
print(f'Benchmark thresholds JSON: {benchmark_json}')

# ## 7. Worst Successful Queries

# In[8]:


if len(successful) > 0:
    worst = successful.sort_values('position_error_m', ascending=False).head(10)
    display(worst[['image_name', 'camera_name', 'capture_id', 'num_matches', 'num_inliers', 'position_error_m', 'heading_error_deg']])
else:
    print('No successful queries to inspect.')

if len(failed) > 0:
    print('Failed queries:')
    failed_columns = ['image_name', 'camera_name', 'capture_id', 'error']
    print(failed[failed_columns].to_string(index=False))
