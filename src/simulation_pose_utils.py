from __future__ import annotations

import json
import math
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


FRONT_CAMERA = "front"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_basename(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file or header is incomplete: {path}")
    return struct.unpack(">II", header[16:24])


def quat_wxyz_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in qvec]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def camera_center_from_qt(qvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    r_cw = quat_wxyz_to_rotmat(qvec)
    t_cw = np.asarray(tvec, dtype=np.float64)
    return -r_cw.T @ t_cw


def metadata_pose_center(rec: dict) -> np.ndarray:
    return camera_center_from_qt(
        [rec["qw"], rec["qx"], rec["qy"], rec["qz"]],
        [rec["tx"], rec["ty"], rec["tz"]],
    )


def normalize_camera_names(camera_names: Optional[Iterable[str]] = None) -> Optional[set]:
    if camera_names is None:
        return None
    normalized = {str(name).strip().lower() for name in camera_names if str(name).strip()}
    return normalized or None


def load_pose_records(
    dataset_root: Path,
    camera_names: Optional[Iterable[str]] = None,
) -> Tuple[List[dict], Dict[str, dict], Dict[Tuple[int, str], dict]]:
    poses = load_json(dataset_root / "metadata" / "poses.json")
    detailed = load_json(dataset_root / "metadata" / "poses_detailed.json")
    selected_cameras = normalize_camera_names(camera_names)

    selected_poses = []
    for pose in poses:
        camera_name = str(pose.get("camera_name", "")).lower()
        if selected_cameras is not None and camera_name not in selected_cameras:
            continue
        selected_poses.append(pose)

    by_name = {image_basename(p["image_name"]): p for p in selected_poses}

    detailed_by_capture_camera = {}
    for cap in detailed:
        cameras = cap.get("cameras", {})
        for camera_name, payload in cameras.items():
            norm_camera = str(camera_name).lower()
            if selected_cameras is not None and norm_camera not in selected_cameras:
                continue
            detailed_by_capture_camera[(int(cap["capture_id"]), norm_camera)] = payload

    return selected_poses, by_name, detailed_by_capture_camera


def load_front_pose_records(dataset_root: Path) -> Tuple[List[dict], Dict[str, dict], Dict[int, dict]]:
    front_poses, by_name, detailed_by_capture_camera = load_pose_records(dataset_root, [FRONT_CAMERA])
    detailed_by_capture = {
        capture_id: payload
        for (capture_id, camera_name), payload in detailed_by_capture_camera.items()
        if camera_name == FRONT_CAMERA
    }

    return front_poses, by_name, detailed_by_capture


def validate_camera_dataset(
    dataset_root: Path,
    image_dir: Optional[Path] = None,
    camera_names: Optional[Iterable[str]] = None,
    image_glob: str = "*.png",
) -> dict:
    images_dir = image_dir or dataset_root / "images"
    image_paths = sorted(images_dir.glob(image_glob))
    image_names = [p.name for p in image_paths]
    poses, pose_by_name, detailed_by_capture_camera = load_pose_records(dataset_root, camera_names)

    intrinsics_payload = load_json(dataset_root / "metadata" / "intrinsics_pinhole.json")
    camera = intrinsics_payload["cameras"][0]
    expected_size = (int(camera["width"]), int(camera["height"]))
    actual_size = read_png_size(image_paths[0]) if image_paths else None
    params = [float(v) for v in camera["params"]]
    fov_deg = camera.get("fov_deg")
    expected_focal = None
    focal_formula_error = None
    if fov_deg is not None and len(params) >= 2:
        expected_focal = expected_size[0] / (2.0 * math.tan(float(fov_deg) * math.pi / 360.0))
        focal_formula_error = max(abs(params[0] - expected_focal), abs(params[1] - expected_focal))

    pose_names = [image_basename(p["image_name"]) for p in poses]
    duplicate_pose_names = sorted(name for name, count in Counter(pose_names).items() if count > 1)
    duplicate_image_names = sorted(name for name, count in Counter(image_names).items() if count > 1)

    missing_images = sorted(set(pose_names) - set(image_names))
    extra_images = sorted(set(image_names) - set(pose_names))
    path_mismatches = sorted(
        p["image_name"]
        for p in poses
        if str(p["image_name"]).replace("\\", "/") != f"images/{image_basename(p['image_name'])}"
    )

    detailed_missing = sorted(
        int(p["capture_id"])
        for p in poses
        if (int(p["capture_id"]), str(p.get("camera_name", "")).lower()) not in detailed_by_capture_camera
    )
    camera_counts = Counter(str(p.get("camera_name", "")).lower() for p in poses)

    return {
        "dataset_root": str(dataset_root),
        "image_dir": str(images_dir),
        "image_count": len(image_paths),
        "pose_count": len(poses),
        "detailed_pose_count": len(detailed_by_capture_camera),
        "camera_counts": dict(sorted(camera_counts.items())),
        # Backward-compatible keys used by older notebook cells.
        "front_pose_count": len(poses),
        "detailed_front_count": len(detailed_by_capture_camera),
        "intrinsics_model": camera["model"],
        "intrinsics_size": expected_size,
        "first_image_size": actual_size,
        "intrinsics_match_first_image": actual_size == expected_size,
        "fov_deg": fov_deg,
        "expected_focal_from_fov": expected_focal,
        "focal_formula_error": focal_formula_error,
        "missing_images": missing_images,
        "extra_images": extra_images,
        "duplicate_pose_names": duplicate_pose_names,
        "duplicate_image_names": duplicate_image_names,
        "metadata_path_mismatches": path_mismatches,
        "detailed_missing_capture_ids": detailed_missing,
        "pose_by_name": pose_by_name,
        "image_names": image_names,
    }


def validate_front_dataset(dataset_root: Path) -> dict:
    return validate_camera_dataset(
        dataset_root=dataset_root,
        image_dir=dataset_root / "images",
        camera_names=[FRONT_CAMERA],
        image_glob="*_front_*.png",
    )


def _max_abs(a: np.ndarray) -> float:
    return float(np.max(np.abs(a))) if a.size else 0.0


def audit_front_pose_matrices(dataset_root: Path) -> Tuple[dict, List[dict]]:
    poses, _, detailed_by_capture = load_front_pose_records(dataset_root)
    rows = []

    for rec in poses:
        capture_id = int(rec["capture_id"])
        detailed = detailed_by_capture[capture_id]["converted_right_handed"]
        t_cw = np.asarray(detailed["matrix_camera_to_world"], dtype=np.float64)
        t_wc = np.asarray(detailed["matrix_world_to_camera"], dtype=np.float64)
        q = np.asarray(detailed["quaternion_wxyz"], dtype=np.float64)
        t = np.asarray(detailed["translation_xyz_m"], dtype=np.float64)

        simple_q = np.array([rec["qw"], rec["qx"], rec["qy"], rec["qz"]], dtype=np.float64)
        simple_t = np.array([rec["tx"], rec["ty"], rec["tz"]], dtype=np.float64)
        r_from_q = quat_wxyz_to_rotmat(q)
        center_from_qt = camera_center_from_qt(q, t)
        center_from_matrix = t_cw[:3, 3]

        rows.append(
            {
                "capture_id": capture_id,
                "image_name": image_basename(rec["image_name"]),
                "quat_norm_error": abs(float(np.linalg.norm(simple_q)) - 1.0),
                "poses_vs_detailed_q_max_abs": _max_abs(simple_q - q),
                "poses_vs_detailed_t_max_abs": _max_abs(simple_t - t),
                "twc_inverse_error": _max_abs(t_wc @ t_cw - np.eye(4)),
                "quat_vs_twc_rotation_error": _max_abs(r_from_q - t_wc[:3, :3]),
                "translation_vs_twc_error": _max_abs(t - t_wc[:3, 3]),
                "center_reprojection_error_m": float(np.linalg.norm(center_from_qt - center_from_matrix)),
                "raw_carla_x": float(rec.get("lat", 0.0)),
                "alt_m": float(rec["alt"]),
                "center_x": float(center_from_matrix[0]),
                "center_y": float(center_from_matrix[1]),
                "center_z": float(center_from_matrix[2]),
                "tx": float(simple_t[0]),
                "ty": float(simple_t[1]),
                "tz": float(simple_t[2]),
                "yaw_deg": float(rec["yaw_deg"]),
            }
        )

    maxes = {
        "count": len(rows),
        "max_quat_norm_error": max((r["quat_norm_error"] for r in rows), default=0.0),
        "max_poses_vs_detailed_q_abs": max((r["poses_vs_detailed_q_max_abs"] for r in rows), default=0.0),
        "max_poses_vs_detailed_t_abs": max((r["poses_vs_detailed_t_max_abs"] for r in rows), default=0.0),
        "max_twc_inverse_error": max((r["twc_inverse_error"] for r in rows), default=0.0),
        "max_quat_vs_twc_rotation_error": max((r["quat_vs_twc_rotation_error"] for r in rows), default=0.0),
        "max_translation_vs_twc_error": max((r["translation_vs_twc_error"] for r in rows), default=0.0),
        "max_center_reprojection_error_m": max((r["center_reprojection_error_m"] for r in rows), default=0.0),
    }
    return maxes, rows


def print_dataset_report(report: dict, max_preview: int = 5) -> None:
    print(f"Dataset: {report['dataset_root']}")
    print(f"Front images: {report['image_count']}")
    print(f"Front pose records: {report['front_pose_count']}")
    print(f"Detailed front records: {report['detailed_front_count']}")
    print(
        "Intrinsics: "
        f"{report['intrinsics_model']} {report['intrinsics_size']} "
        f"fov={report['fov_deg']} | first image size={report['first_image_size']} "
        f"| match={report['intrinsics_match_first_image']}"
    )
    if report["focal_formula_error"] is not None:
        print(
            "FOV focal check: "
            f"expected={report['expected_focal_from_fov']:.9f} "
            f"max_fx_fy_error={report['focal_formula_error']:.12g}"
        )

    checks = [
        ("missing_images", "Pose records without image"),
        ("extra_images", "Images without pose record"),
        ("duplicate_pose_names", "Duplicate pose names"),
        ("duplicate_image_names", "Duplicate image names"),
        ("metadata_path_mismatches", "Metadata path layout mismatches"),
        ("detailed_missing_capture_ids", "Missing detailed front captures"),
    ]
    for key, label in checks:
        values = report[key]
        print(f"{label}: {len(values)}")
        if values:
            print("  examples:", values[:max_preview])


def print_matrix_audit(maxes: dict) -> None:
    print("Matrix/pose audit max errors")
    for key, value in maxes.items():
        print(f"  {key}: {value}")
    print("Reminder: tx/ty/tz are COLMAP world-to-camera translation, not camera position.")


def get_pycolmap_cam_from_world(image):
    cam_from_world = getattr(image, "cam_from_world", None)
    if cam_from_world is not None:
        return cam_from_world() if callable(cam_from_world) else cam_from_world
    return None


def _call_or_value(obj):
    return obj() if callable(obj) else obj


def pycolmap_transform_rt(transform) -> Tuple[np.ndarray, np.ndarray]:
    """Return R,t for a pycolmap cam_from_world transform without assuming quaternion order."""
    matrix_attr = getattr(transform, "matrix", None)
    if matrix_attr is not None:
        matrix = np.asarray(_call_or_value(matrix_attr), dtype=np.float64)
        if matrix.shape == (3, 4):
            return matrix[:3, :3], matrix[:3, 3]
        if matrix.shape == (4, 4):
            return matrix[:3, :3], matrix[:3, 3]

    rotation = getattr(transform, "rotation", None)
    if rotation is None:
        raise ValueError("Cannot read rotation from pycolmap transform.")

    rotation_matrix = None
    for attr_name in ("matrix", "as_matrix"):
        attr = getattr(rotation, attr_name, None)
        if attr is not None:
            rotation_matrix = np.asarray(_call_or_value(attr), dtype=np.float64)
            break
    if rotation_matrix is None:
        raise ValueError(
            "Cannot read rotation matrix from pycolmap transform. "
            "Avoiding rotation.quat because its component order depends on pycolmap API."
        )

    translation = getattr(transform, "translation", None)
    if translation is None:
        t_attr = getattr(transform, "t", None)
        if t_attr is None:
            raise ValueError("Cannot read translation from pycolmap transform.")
        translation = t_attr
    translation = np.asarray(_call_or_value(translation), dtype=np.float64)

    return rotation_matrix, translation


def pycolmap_image_qt(image) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        return np.asarray(image.qvec, dtype=np.float64), np.asarray(image.tvec, dtype=np.float64)

    cam_from_world = get_pycolmap_cam_from_world(image)
    if cam_from_world is None:
        raise ValueError(f"Cannot read pose for image {getattr(image, 'name', '<unknown>')}")

    rotation = getattr(cam_from_world, "rotation", None)
    if rotation is None:
        raise ValueError(f"Cannot read rotation for image {getattr(image, 'name', '<unknown>')}")
    quat = getattr(rotation, "quat", None)
    quat = quat() if callable(quat) else quat
    if quat is None:
        raise ValueError(f"Cannot read quaternion for image {getattr(image, 'name', '<unknown>')}")

    translation = getattr(cam_from_world, "translation", None)
    translation = translation() if callable(translation) else translation
    if translation is None:
        raise ValueError(f"Cannot read translation for image {getattr(image, 'name', '<unknown>')}")

    return np.asarray(quat, dtype=np.float64), np.asarray(translation, dtype=np.float64)


def pycolmap_image_center(image) -> np.ndarray:
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        qvec, tvec = pycolmap_image_qt(image)
        return camera_center_from_qt(qvec, tvec)

    cam_from_world = get_pycolmap_cam_from_world(image)
    if cam_from_world is None:
        raise ValueError(f"Cannot read pose for image {getattr(image, 'name', '<unknown>')}")
    r_cw, t_cw = pycolmap_transform_rt(cam_from_world)
    return -r_cw.T @ t_cw


def find_model_image_by_basename(model, basename: str):
    for image in model.images.values():
        if image_basename(image.name) == basename:
            return image
    return None


def estimate_similarity_umeyama(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both have shape (N, 3).")
    if source.shape[0] < 3:
        raise ValueError(f"At least 3 points are required for Sim(3), got {source.shape[0]}.")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    cov = (tgt_centered.T @ src_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[-1, -1] = -1.0
    rotation = u @ s_fix @ vt

    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ s_fix) / variance)
    translation = tgt_mean - scale * rotation @ src_mean

    transformed = apply_similarity(source, scale, rotation, translation)
    errors = np.linalg.norm(transformed - target, axis=1)
    return scale, rotation, translation, errors


def apply_similarity(points_xyz: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    return scale * (points @ rotation.T) + np.asarray(translation, dtype=np.float64)


def build_alignment_from_registered_images(model, pose_by_name: Dict[str, dict]) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    pairs = collect_registered_alignment_pairs(model, pose_by_name)
    colmap_centers = [p["colmap_center"] for p in pairs]
    metadata_centers = [p["metadata_center"] for p in pairs]
    names = [p["name"] for p in pairs]

    if len(colmap_centers) < 3:
        raise ValueError(f"Need at least 3 registered images with metadata, got {len(colmap_centers)}.")

    scale, rotation, translation, errors = estimate_similarity_umeyama(
        np.asarray(colmap_centers), np.asarray(metadata_centers)
    )
    stats = {
        "matched_images": len(names),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "names": names,
    }
    return scale, rotation, translation, stats


def collect_registered_alignment_pairs(model, pose_by_name: Dict[str, dict]) -> List[dict]:
    pairs = []
    for image in model.images.values():
        basename = image_basename(image.name)
        rec = pose_by_name.get(basename)
        if rec is None:
            continue
        pairs.append(
            {
                "capture_id": int(rec["capture_id"]),
                "name": basename,
                "model_name": image.name,
                "image_id": int(image.image_id),
                "colmap_center": pycolmap_image_center(image),
                "metadata_center": metadata_pose_center(rec),
            }
        )
    return sorted(pairs, key=lambda p: p["capture_id"])


def estimate_alignment_for_pairs(pairs: Sequence[dict]) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    if len(pairs) < 3:
        raise ValueError(f"Need at least 3 pairs for Sim(3), got {len(pairs)}.")
    colmap_centers = np.asarray([p["colmap_center"] for p in pairs], dtype=np.float64)
    metadata_centers = np.asarray([p["metadata_center"] for p in pairs], dtype=np.float64)
    scale, rotation, translation, errors = estimate_similarity_umeyama(colmap_centers, metadata_centers)
    stats = {
        "matched_images": len(pairs),
        "capture_min": int(min(p["capture_id"] for p in pairs)),
        "capture_max": int(max(p["capture_id"] for p in pairs)),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "names": [p["name"] for p in pairs],
    }
    return scale, rotation, translation, stats


def alignment_window_around_capture(
    pairs: Sequence[dict],
    capture_id: int,
    half_window: int = 20,
    min_pairs: int = 8,
    exclude_name: Optional[str] = None,
) -> List[dict]:
    selected = [
        p
        for p in pairs
        if abs(int(p["capture_id"]) - int(capture_id)) <= half_window
        and (exclude_name is None or p["name"] != exclude_name)
    ]
    if len(selected) >= min_pairs:
        return selected

    candidates = [p for p in pairs if exclude_name is None or p["name"] != exclude_name]
    candidates = sorted(candidates, key=lambda p: abs(int(p["capture_id"]) - int(capture_id)))
    return candidates[:max(min_pairs, 3)]


def summarize_alignment_windows(
    pairs: Sequence[dict],
    window_size: int = 40,
) -> List[dict]:
    if not pairs:
        return []
    capture_ids = [int(p["capture_id"]) for p in pairs]
    start = min(capture_ids)
    end = max(capture_ids)
    rows = []
    lo = start
    while lo <= end:
        hi = min(end, lo + window_size - 1)
        chunk = [p for p in pairs if lo <= int(p["capture_id"]) <= hi]
        if len(chunk) >= 3:
            _, _, _, stats = estimate_alignment_for_pairs(chunk)
            rows.append(stats)
        lo = hi + 1
    return rows


def build_known_pose_triangulated_model(
    dataset_root: Path,
    feature_bundle_root: Path,
    output_root: Path,
    overwrite: bool = False,
):
    """Build a triangulated COLMAP model using metadata q/t as fixed camera poses.

    The source database/features/matches/images are read from feature_bundle_root.
    The generated model is written to output_root/sfm.
    """
    import pycolmap

    if output_root.exists():
        if not overwrite:
            return pycolmap.Reconstruction(output_root / "sfm")
        shutil.rmtree(output_root)

    sparse_input = output_root / "sparse_input"
    sfm_output = output_root / "sfm"
    sparse_input.mkdir(parents=True, exist_ok=True)
    sfm_output.mkdir(parents=True, exist_ok=True)

    report = validate_front_dataset(dataset_root)
    pose_by_name = report["pose_by_name"]
    intrinsics = load_json(dataset_root / "metadata" / "intrinsics_pinhole.json")["cameras"][0]

    params = " ".join(str(float(v)) for v in intrinsics["params"])
    (sparse_input / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 {intrinsics['model']} {int(intrinsics['width'])} {int(intrinsics['height'])} {params}\n",
        encoding="utf-8",
    )

    with (sparse_input / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for image_id, name in enumerate(sorted(pose_by_name), start=1):
            rec = pose_by_name[name]
            f.write(
                f"{image_id} {rec['qw']} {rec['qx']} {rec['qy']} {rec['qz']} "
                f"{rec['tx']} {rec['ty']} {rec['tz']} 1 images/{name}\n\n"
            )

    (sparse_input / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n",
        encoding="utf-8",
    )

    reconstruction = pycolmap.Reconstruction(sparse_input)
    return pycolmap.triangulate_points(
        reconstruction,
        str(feature_bundle_root / "database.db"),
        str(feature_bundle_root),
        str(sfm_output),
        clear_points=True,
        refine_intrinsics=False,
    )
