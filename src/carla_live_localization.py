#!/usr/bin/env python

"""In-memory live visual localization for the async CARLA capture loop.

This module intentionally does not write query images, features, pairs, or
matches to disk. It loads the map/retrieval/local feature assets once, then
localizes copied RGB frames in a background thread.
"""

import copy
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


CARLA_TO_COLMAP_S = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)
COLMAP_TO_CARLA_S = np.linalg.inv(CARLA_TO_COLMAP_S)


def wrap_angle_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh: np.ndarray) -> float:
    """Convert PyCOLMAP's right-handed world->camera rotation to CARLA yaw."""
    r_cw_rh = np.asarray(r_wc_rh, dtype=np.float64).T
    r_cw_lh = COLMAP_TO_CARLA_S @ r_cw_rh @ CARLA_TO_COLMAP_S
    return wrap_angle_deg(math.degrees(math.atan2(r_cw_lh[1, 0], r_cw_lh[0, 0])))


def carla_image_rgb_array(image) -> np.ndarray:
    """Convert a CARLA BGRA image object to contiguous RGB uint8."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def rgb_to_gray_tensor(rgb_array: np.ndarray, torch, device) :
    """Convert HxWx3 uint8 RGB to a 1x1xHxW float tensor in [0, 1]."""
    rgb = np.asarray(rgb_array)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'Expected RGB array with shape HxWx3, got {rgb.shape}')
    gray = (
        0.2989 * rgb[:, :, 0].astype(np.float32)
        + 0.5870 * rgb[:, :, 1].astype(np.float32)
        + 0.1140 * rgb[:, :, 2].astype(np.float32)
    ) / 255.0
    gray = np.ascontiguousarray(gray[None, None, :, :])
    return torch.from_numpy(gray).to(device=device, non_blocking=True)


def rgb_to_color_tensor(rgb_array: np.ndarray, torch, device):
    """Convert HxWx3 uint8 RGB to a 1x3xHxW float tensor in [0, 1]."""
    rgb = np.asarray(rgb_array)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'Expected RGB array with shape HxWx3, got {rgb.shape}')
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return torch.from_numpy(chw).to(device=device, non_blocking=True)


def resize_tensor_max(image_tensor, max_size: int, torch):
    if max_size <= 0:
        return image_tensor
    height, width = image_tensor.shape[-2:]
    scale = float(max_size) / float(max(height, width))
    if scale >= 1.0:
        return image_tensor
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return torch.nn.functional.interpolate(
        image_tensor,
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False,
    )


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _as_numpy(value[0])
        value = [_as_numpy(item) for item in value]
    return np.asarray(value)


def _strip_batch(pred: Dict[str, object]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in pred.items():
        arr = _as_numpy(value)
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        out[key] = arr
    return out


def _normalize_descriptors(desc: np.ndarray, n_keypoints: int) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2:
        raise ValueError(f'Descriptors must be 2D, got {desc.shape}')
    if desc.shape[0] != n_keypoints and desc.shape[1] == n_keypoints:
        desc = desc.T
    return np.ascontiguousarray(desc)


def _read_h5_feature_group(group) -> Dict[str, np.ndarray]:
    data = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            data[key] = value[()]
    return data


def _load_h5_tree(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    items: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(path, 'r') as h5:
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and any(isinstance(v, h5py.Dataset) for v in obj.values()):
                items[name] = _read_h5_feature_group(obj)
        h5.visititems(visit)
    return items


def _load_global_descriptors(path: Path) -> Dict[str, np.ndarray]:
    raw = _load_h5_tree(path)
    descriptors = {}
    for name, values in raw.items():
        if 'global_descriptor' in values:
            desc = values['global_descriptor']
        elif 'descriptor' in values:
            desc = values['descriptor']
        else:
            continue
        desc = np.asarray(desc, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(desc)
        descriptors[name] = desc / max(norm, 1e-12)
    return descriptors


def _image_name_candidates(name: str) -> List[str]:
    p = Path(name)
    return [name, p.as_posix(), p.name, f'images/{p.name}', f'images/front/{p.name}']


def _find_loaded_name(name: str, loaded: Dict[str, object]) -> Optional[str]:
    for candidate in _image_name_candidates(name):
        if candidate in loaded:
            return candidate
    suffix = '/' + Path(name).name
    for candidate in loaded:
        if candidate.endswith(suffix):
            return candidate
    return None


class LiveCarlaLocalizer:
    """Preloaded, thread-backed HLoc/PyCOLMAP localizer for live CARLA frames."""

    def __init__(
        self,
        bundle_root: str = 'sim-19-may-bundle',
        num_loc: int = 10,
        max_error: float = 12.0,
        max_workers: int = 1,
        retrieval_max_size: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        self.bundle_root = Path(bundle_root)
        self.sfm_root = self.bundle_root / 'sfm'
        self.db_features_path = self.bundle_root / 'features.h5'
        self.db_global_path = self.bundle_root / 'global-feats-netvlad.h5'
        self.num_loc = int(num_loc)
        self.max_error = float(max_error)
        self.retrieval_max_size = int(retrieval_max_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='carla-localizer')
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.busy = False

        self._import_runtime_dependencies()
        self.device = self.torch.device(device or ('cuda' if self.torch.cuda.is_available() else 'cpu'))
        self._load_map()
        self._load_reference_features()
        self._load_models()
        print(
            f'[Localization Ready] refs={len(self.reference_names)} '
            f'points3D={len(self.reconstruction.points3D)} device={self.device} '
            f'retrieval_max_size={self.retrieval_max_size}'
        )

    def _import_runtime_dependencies(self) -> None:
        try:
            import pycolmap
            import torch
            from hloc import extract_features, match_features
            from hloc.utils.base_model import dynamic_load
            import hloc.extractors as extractors
            import hloc.matchers as matchers
        except ImportError as exc:
            raise RuntimeError(
                'Live localization requires pycolmap, torch, and hloc in the Python environment.'
            ) from exc

        self.pycolmap = pycolmap
        self.torch = torch
        self.extract_features = extract_features
        self.match_features = match_features
        self.dynamic_load = dynamic_load
        self.extractors = extractors
        self.matchers = matchers

    def _load_map(self) -> None:
        for path in [self.sfm_root, self.db_features_path, self.db_global_path]:
            if not path.exists():
                raise FileNotFoundError(path)
        self.reconstruction = self.pycolmap.Reconstruction(self.sfm_root)
        self.camera = next(iter(self.reconstruction.cameras.values()))
        self.reference_names = [image.name for image in self.reconstruction.images.values()]
        self.image_name_to_id = {image.name: image_id for image_id, image in self.reconstruction.images.items()}

    def _load_reference_features(self) -> None:
        self.db_features = _load_h5_tree(self.db_features_path)
        self.db_global = _load_global_descriptors(self.db_global_path)
        if not self.db_features:
            raise RuntimeError(f'No reference local features found in {self.db_features_path}')
        if not self.db_global:
            raise RuntimeError(f'No reference global descriptors found in {self.db_global_path}')

        self.reference_global = []
        self.reference_global_names = []
        for name in self.reference_names:
            key = _find_loaded_name(name, self.db_global)
            if key is None:
                continue
            self.reference_global_names.append(name)
            self.reference_global.append(self.db_global[key])
        if not self.reference_global:
            raise RuntimeError('No registered images have matching NetVLAD descriptors.')
        self.reference_global = np.stack(self.reference_global, axis=0).astype(np.float32)

    def _load_models(self) -> None:
        self._add_hloc_third_party_paths()

        feature_conf = copy.deepcopy(self.extract_features.confs['superpoint_max'])
        retrieval_conf = copy.deepcopy(self.extract_features.confs['netvlad'])
        matcher_conf = copy.deepcopy(self.match_features.confs['superpoint+lightglue'])

        try:
            SuperPoint = self.dynamic_load(self.extractors, feature_conf['model']['name'])
            NetVLAD = self.dynamic_load(self.extractors, retrieval_conf['model']['name'])
            LightGlue = self.dynamic_load(self.matchers, matcher_conf['model']['name'])
        except ModuleNotFoundError as exc:
            if exc.name == 'SuperGluePretrainedNetwork':
                raise RuntimeError(
                    'HLoc cannot find SuperGluePretrainedNetwork. Install HLoc from a recursive clone, e.g. '
                    '"git clone --recursive https://github.com/cvg/Hierarchical-Localization.git" then '
                    '"pip install -e Hierarchical-Localization", or set HLOC_THIRD_PARTY to the directory '
                    'that contains the SuperGluePretrainedNetwork folder.'
                ) from exc
            raise

        self.superpoint = SuperPoint(feature_conf['model']).eval().to(self.device)
        self.netvlad = NetVLAD(retrieval_conf['model']).eval().to(self.device)
        self.lightglue = LightGlue(matcher_conf['model']).eval().to(self.device)
        self.reference_global_gpu = self.torch.from_numpy(self.reference_global).to(self.device)

    def _add_hloc_third_party_paths(self) -> None:
        candidates = []
        env_path = os.environ.get('HLOC_THIRD_PARTY')
        if env_path:
            candidates.append(Path(env_path))

        hloc_root = Path(self.extract_features.__file__).resolve().parent
        candidates.extend([
            hloc_root / 'third_party',
            hloc_root.parent / 'third_party',
            Path.cwd() / 'third_party',
            Path.cwd() / 'Hierarchical-Localization' / 'third_party',
        ])

        for path in candidates:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def submit(self, rgb_array: np.ndarray, carla_world_or_map) -> bool:
        """Submit a copied RGB frame for background localization.

        Returns False if a localization job is already running.
        """
        with self.lock:
            if self.busy:
                print('[Localization Busy] Previous localization is still running.')
                return False
            self.busy = True

        frame = np.ascontiguousarray(rgb_array.copy())
        carla_map = carla_world_or_map.get_map() if hasattr(carla_world_or_map, 'get_map') else carla_world_or_map
        print('[Lozalization Started]')
        future = self.executor.submit(self._localize_and_print, frame, carla_map)
        future.add_done_callback(self._mark_done)
        return True

    def _mark_done(self, future) -> None:
        with self.lock:
            self.busy = False
        exc = future.exception()
        if exc is not None:
            print(f'[Localization Error] {exc}')

    def _localize_and_print(self, rgb_array: np.ndarray, carla_map) -> None:
        result = self.localize(rgb_array, carla_map)
        if not result['success']:
            print(f'[Localization Failed] {result["error"]}')
            return
        print(
            '[Localization Result] '
            f'lat={result["latitude"]:.8f} lon={result["longitude"]:.8f} '
            f'alt={result["altitude"]:.3f} heading={result["heading_deg"]:.2f}deg '
            f'inliers={result["num_inliers"]} refs={result["retrieved"]}'
        )

    def localize(self, rgb_array: np.ndarray, carla_map) -> Dict[str, object]:
        result = self.localize_xyz_heading(rgb_array)
        if not result['success']:
            return result

        import carla
        geo = carla_map.transform_to_geolocation(
            carla.Location(x=float(result['x']), y=float(result['y']), z=float(result['z']))
        )
        result.update({
            'latitude': float(geo.latitude),
            'longitude': float(geo.longitude),
            'altitude': float(geo.altitude),
        })
        return result

    def localize_xyz_heading(self, rgb_array: np.ndarray) -> Dict[str, object]:
        with self.inference_lock:
            return self._localize_xyz_heading_impl(rgb_array)


    def _localize_xyz_heading_impl(self, rgb_array: np.ndarray) -> Dict[str, object]:
        with self.torch.inference_mode():
            color_tensor = rgb_to_color_tensor(rgb_array, self.torch, self.device)
            color_tensor = resize_tensor_max(color_tensor, self.retrieval_max_size, self.torch)
            gray_tensor = rgb_to_gray_tensor(rgb_array, self.torch, self.device)
            query_global = self._extract_global(color_tensor)
            query_features = self._extract_local(gray_tensor)

        retrieved = self._retrieve(query_global)
        points2d, points3d = self._match_to_3d(query_features, retrieved)
        if len(points2d) < 6:
            return {
                'success': False,
                'error': f'Not enough 2D-3D correspondences: {len(points2d)}',
                'retrieved': retrieved,
            }

        ret = self._estimate_pose(points2d, points3d, rgb_array.shape[1], rgb_array.shape[0])
        if ret is None:
            return {'success': False, 'error': 'PyCOLMAP pose estimation failed', 'retrieved': retrieved}

        r_wc_rh, tvec = self._cam_from_world_rt(ret['cam_from_world'])
        center_colmap = -r_wc_rh.T @ tvec
        center_carla = COLMAP_TO_CARLA_S @ center_colmap
        heading = colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh)

        return {
            'success': True,
            'heading_deg': float(heading),
            'x': float(center_carla[0]),
            'y': float(center_carla[1]),
            'z': float(center_carla[2]),
            'num_inliers': int(ret.get('num_inliers', len(points2d))),
            'retrieved': retrieved,
        }

    def _extract_global(self, image_tensor: object) -> np.ndarray:
        pred = self.netvlad({'image': image_tensor})
        desc = pred.get('global_descriptor', pred.get('descriptor'))
        desc = _as_numpy(desc).reshape(-1).astype(np.float32)
        return desc / max(np.linalg.norm(desc), 1e-12)

    def _extract_local(self, image_tensor: object) -> Dict[str, np.ndarray]:
        pred = _strip_batch(self.superpoint({'image': image_tensor}))
        pred['keypoints'] = np.asarray(pred['keypoints'], dtype=np.float32)
        pred['descriptors'] = _normalize_descriptors(pred['descriptors'], len(pred['keypoints']))
        if 'scores' in pred:
            pred['scores'] = np.asarray(pred['scores'], dtype=np.float32)
        pred['image_size'] = np.array([image_tensor.shape[-1], image_tensor.shape[-2]], dtype=np.float32)
        return pred

    def _retrieve(self, query_global: np.ndarray) -> List[str]:
        with self.torch.inference_mode():
            q_gpu = self.torch.from_numpy(query_global.astype(np.float32)).to(self.device)
            sims = (self.reference_global_gpu @ q_gpu).cpu().numpy()
        order = np.argsort(-sims)[:self.num_loc]
        return [self.reference_global_names[int(i)] for i in order]

    def _match_to_3d(
        self,
        query_features: Dict[str, np.ndarray],
        retrieved_names: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_points: List[np.ndarray] = []
        xyz_points: List[np.ndarray] = []
        used = set()

        for ref_name in retrieved_names:
            ref_id = self.image_name_to_id.get(ref_name)
            if ref_id is None:
                continue
            ref_image = self.reconstruction.images[ref_id]
            ref_key = _find_loaded_name(ref_name, self.db_features)
            if ref_key is None:
                continue
            ref_features = self._prepare_reference_features(self.db_features[ref_key])
            matches = self._match_pair(query_features, ref_features)

            ref_point3d_ids = self._image_point3d_ids(ref_image)
            for q_idx, r_idx in matches:
                if r_idx < 0 or r_idx >= len(ref_point3d_ids):
                    continue
                point3d_id = int(ref_point3d_ids[r_idx])
                if point3d_id == -1 or point3d_id not in self.reconstruction.points3D:
                    continue
                if q_idx in used:
                    continue
                used.add(int(q_idx))
                q_points.append(query_features['keypoints'][q_idx] + 0.5)
                xyz_points.append(np.asarray(self.reconstruction.points3D[point3d_id].xyz, dtype=np.float64))

        if not q_points:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        return np.asarray(q_points, dtype=np.float64), np.asarray(xyz_points, dtype=np.float64)

    def _prepare_reference_features(self, values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        keypoints = np.asarray(values['keypoints'], dtype=np.float32)
        features = {
            'keypoints': keypoints,
            'descriptors': _normalize_descriptors(values['descriptors'], len(keypoints)),
        }
        if 'scores' in values:
            features['scores'] = np.asarray(values['scores'], dtype=np.float32)
        features['image_size'] = np.array([self.camera.width, self.camera.height], dtype=np.float32)
        return features

    def _match_pair(self, query: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> np.ndarray:
        data = {}
        data.update(self._features_to_torch(query, suffix='0'))
        data.update(self._features_to_torch(ref, suffix='1'))
        with self.torch.inference_mode():
            pred = _strip_batch(self.lightglue(data))

        if 'matches' in pred:
            matches = np.asarray(pred['matches'], dtype=np.int64)
            if matches.ndim == 2 and matches.shape[1] == 2:
                return matches

        if 'matches0' not in pred:
            raise RuntimeError(f'LightGlue output does not contain matches or matches0: {sorted(pred.keys())}')
        matches0 = np.asarray(pred['matches0'], dtype=np.int64).reshape(-1)
        q_idx = np.nonzero(matches0 >= 0)[0]
        return np.stack([q_idx, matches0[q_idx]], axis=1).astype(np.int64)

    def _features_to_torch(self, features: Dict[str, np.ndarray], suffix: str) -> Dict[str, object]:
        data = {}
        for key in ['keypoints', 'descriptors', 'scores']:
            if key not in features:
                continue
            value = np.asarray(features[key])
            if key == 'descriptors' and value.ndim == 2 and value.shape[0] == len(features['keypoints']):
                value = value.T
            tensor = self.torch.from_numpy(np.ascontiguousarray(value)).float().to(self.device)
            data[key + suffix] = tensor.unsqueeze(0)

        image_size = np.asarray(features['image_size'], dtype=np.int64)
        width, height = int(image_size[0]), int(image_size[1])
        data['image' + suffix] = self.torch.empty((1, 1, height, width), device=self.device)
        return data

    def _image_point3d_ids(self, image) -> np.ndarray:
        ids = []
        for point2d in image.points2D:
            point3d_id = getattr(point2d, 'point3D_id', -1)
            if point3d_id is None:
                ids.append(-1)
                continue
            try:
                point3d_id = int(point3d_id)
            except OverflowError:
                ids.append(-1)
                continue
            if point3d_id < 0 or point3d_id > 9223372036854775807:
                ids.append(-1)
            else:
                ids.append(point3d_id)
        return np.asarray(ids, dtype=np.int64)

    def _estimate_pose(self, points2d: np.ndarray, points3d: np.ndarray, width: int, height: int):
        camera = self._scaled_camera(width, height)
        return self.pycolmap.estimate_and_refine_absolute_pose(
            points2d,
            points3d,
            camera,
            estimation_options={'ransac': {'max_error': self.max_error}},
            refinement_options={'refine_focal_length': False, 'refine_extra_params': False},
        )

    def _scaled_camera(self, width: int, height: int):
        if int(width) == int(self.camera.width) and int(height) == int(self.camera.height):
            return self.camera
        params = np.array(self.camera.params, dtype=np.float64).copy()
        sx = float(width) / float(self.camera.width)
        sy = float(height) / float(self.camera.height)
        if len(params) >= 4:
            params[0] *= sx
            params[1] *= sy
            params[2] *= sx
            params[3] *= sy
        return self.pycolmap.Camera(
            model=self.camera.model_name,
            width=int(width),
            height=int(height),
            params=params,
        )

    def _cam_from_world_rt(self, cam_from_world) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(cam_from_world, 'rotation') and hasattr(cam_from_world.rotation, 'matrix'):
            r_wc = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
            tvec = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
            return r_wc, tvec
        mat = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        return mat[:3, :3], mat[:3, 3]

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


def submit_latest_image_for_localization(localizer: LiveCarlaLocalizer, latest_image, world) -> bool:
    """Convenience function for the CARLA Pygame KEYDOWN/K_l handler."""
    if latest_image is None:
        print('[Localization Skipped] No camera image received yet.')
        return False
    return localizer.submit(carla_image_rgb_array(latest_image), world)
