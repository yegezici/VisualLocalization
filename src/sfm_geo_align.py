"""Align a COLMAP/pycolmap SfM model to GPS (lat/lon/alt) using a similarity transform.

This script is designed for your CSV format:
  filename,lat,lon,meters,degrees,datetime

We use:
- lat, lon in decimal degrees
- meters as altitude (meters above sea level)

What it does:
1) Reads SfM camera centers from a reconstruction (COLMAP model).
2) Converts GPS points to a local ENU frame (meters).
3) Fits a Sim(3) transform: enu ~= s * R * C_sfm + t.
4) Optionally converts a HLoc poses.txt (cam_from_world) to lat/lon/alt.

Notes:
- The "degrees" column looks like compass heading; it is currently ignored.
- This does not modify the SfM model; it writes alignment parameters to JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


# WGS84 constants
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_B = _A * (1.0 - _F)
_EP2 = (_A * _A - _B * _B) / (_B * _B)


@dataclass(frozen=True)
class LLA:
	lat_deg: float
	lon_deg: float
	alt_m: float


def _deg2rad(x: float) -> float:
	return x * math.pi / 180.0


def _rad2deg(x: float) -> float:
	return x * 180.0 / math.pi


def lla_to_ecef(p: LLA) -> np.ndarray:
	lat = _deg2rad(p.lat_deg)
	lon = _deg2rad(p.lon_deg)
	slat = math.sin(lat)
	clat = math.cos(lat)
	slon = math.sin(lon)
	clon = math.cos(lon)
	N = _A / math.sqrt(1.0 - _E2 * slat * slat)
	x = (N + p.alt_m) * clat * clon
	y = (N + p.alt_m) * clat * slon
	z = (N * (1.0 - _E2) + p.alt_m) * slat
	return np.array([x, y, z], dtype=float)


def ecef_to_lla(xyz: np.ndarray) -> LLA:
	x, y, z = map(float, xyz)
	lon = math.atan2(y, x)
	p = math.hypot(x, y)
	if p < 1e-9:
		lat = math.copysign(math.pi / 2.0, z)
		N = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
		alt = abs(z) - N * (1.0 - _E2)
		return LLA(_rad2deg(lat), _rad2deg(lon), alt)

	theta = math.atan2(z * _A, p * _B)
	st = math.sin(theta)
	ct = math.cos(theta)
	lat = math.atan2(z + _EP2 * _B * st**3, p - _E2 * _A * ct**3)
	slat = math.sin(lat)
	N = _A / math.sqrt(1.0 - _E2 * slat * slat)
	alt = p / math.cos(lat) - N
	return LLA(_rad2deg(lat), _rad2deg(lon), alt)


def ecef_to_enu(ecef: np.ndarray, origin: LLA) -> np.ndarray:
	lat0 = _deg2rad(origin.lat_deg)
	lon0 = _deg2rad(origin.lon_deg)
	slat0 = math.sin(lat0)
	clat0 = math.cos(lat0)
	slon0 = math.sin(lon0)
	clon0 = math.cos(lon0)

	ecef0 = lla_to_ecef(origin)
	d = (ecef - ecef0).astype(float)

	R = np.array(
		[
			[-slon0, clon0, 0.0],
			[-slat0 * clon0, -slat0 * slon0, clat0],
			[clat0 * clon0, clat0 * slon0, slat0],
		],
		dtype=float,
	)
	return R @ d


def enu_to_ecef(enu: np.ndarray, origin: LLA) -> np.ndarray:
	lat0 = _deg2rad(origin.lat_deg)
	lon0 = _deg2rad(origin.lon_deg)
	slat0 = math.sin(lat0)
	clat0 = math.cos(lat0)
	slon0 = math.sin(lon0)
	clon0 = math.cos(lon0)

	ecef0 = lla_to_ecef(origin)
	R = np.array(
		[
			[-slon0, clon0, 0.0],
			[-slat0 * clon0, -slat0 * slon0, clat0],
			[clat0 * clon0, clat0 * slon0, slat0],
		],
		dtype=float,
	)
	return ecef0 + R.T @ enu


def mean_lla(values: list[LLA]) -> LLA:
	if not values:
		raise ValueError("No GPS values")
	return LLA(
		lat_deg=float(np.mean([v.lat_deg for v in values])),
		lon_deg=float(np.mean([v.lon_deg for v in values])),
		alt_m=float(np.mean([v.alt_m for v in values])),
	)


def _to_float(s: str) -> float:
	return float(str(s).strip().replace(",", "."))


def read_coordinates_csv(path: Path) -> dict[str, LLA]:
	"""Read your 5-column CSV and return mapping filename -> LLA."""
	with path.open(newline="") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames:
			raise ValueError("CSV has no header")
		fields = {h.strip().lower(): h for h in reader.fieldnames}

		def col(*names: str) -> str:
			for n in names:
				if n in fields:
					return fields[n]
			raise ValueError(f"Missing required column(s): {names}. Got: {reader.fieldnames}")

		c_name = col("filename", "name")
		c_lat = col("lat", "latitude")
		c_lon = col("lon", "lng", "longitude")
		c_alt = col("meters", "alt", "altitude", "height")

		gps: dict[str, LLA] = {}
		for row in reader:
			name = (row.get(c_name) or "").strip()
			if not name:
				continue
			lat = _to_float(row[c_lat])
			lon = _to_float(row[c_lon])
			alt = _to_float(row[c_alt])
			gps[name] = LLA(lat, lon, alt)
		return gps


def camera_center_from_image(img: pycolmap.Image) -> np.ndarray:
	pose = img.cam_from_world()
	try:
		c = pose.inverse().translation
		return np.array([c[0], c[1], c[2]], dtype=float)
	except Exception:
		R = np.array(pose.rotation.matrix(), dtype=float)
		t = np.array(pose.translation, dtype=float)
		return -R.T @ t


def umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
	"""Fit Y ≈ s R X + t (least squares). X,Y are Nx3."""
	if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
		raise ValueError("X and Y must be Nx3 with same shape")
	n = X.shape[0]
	if n < 3:
		raise ValueError("Need at least 3 correspondences")

	mx = X.mean(axis=0)
	my = Y.mean(axis=0)
	Xc = X - mx
	Yc = Y - my

	cov = (Yc.T @ Xc) / float(n)
	U, D, Vt = np.linalg.svd(cov)

	S = np.eye(3)
	if np.linalg.det(U) * np.linalg.det(Vt) < 0:
		S[2, 2] = -1.0

	R = U @ S @ Vt
	var_x = float((Xc**2).sum() / n)
	s = float((D * np.diag(S)).sum() / var_x)
	t = my - s * (R @ mx)
	return s, R, t


def robust_fit_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Two-pass robust fit: fit once, drop large residuals, fit again."""
	s, R, t = umeyama_sim3(X, Y)
	pred = (s * (X @ R.T)) + t
	res = np.linalg.norm(pred - Y, axis=1)

	med = float(np.median(res))
	thr = max(3.0 * med, 5.0)
	inliers = res <= thr
	if inliers.sum() >= 3:
		s, R, t = umeyama_sim3(X[inliers], Y[inliers])
		pred = (s * (X @ R.T)) + t
		res = np.linalg.norm(pred - Y, axis=1)
	return s, R, t, inliers, res


def read_pose_file_centers(path: Path) -> list[tuple[str, np.ndarray]]:
	centers = []
	for line in path.read_text().splitlines():
		line = line.strip()
		if not line:
			continue
		parts = line.split()
		if len(parts) != 8:
			raise ValueError(f"Unexpected pose format (expected 8 tokens): {line}")
		name = parts[0]
		qw, qx, qy, qz = map(float, parts[1:5])
		tx, ty, tz = map(float, parts[5:8])

		# Quaternion (w,x,y,z) -> R
		n = qw * qw + qx * qx + qy * qy + qz * qz
		if n == 0.0:
			R = np.eye(3, dtype=float)
		else:
			s2 = 2.0 / n
			wx, wy, wz = s2 * qw * qx, s2 * qw * qy, s2 * qw * qz
			xx, xy, xz = s2 * qx * qx, s2 * qx * qy, s2 * qx * qz
			yy, yz, zz = s2 * qy * qy, s2 * qy * qz, s2 * qz * qz
			R = np.array(
				[
					[1.0 - (yy + zz), xy - wz, xz + wy],
					[xy + wz, 1.0 - (xx + zz), yz - wx],
					[xz - wy, yz + wx, 1.0 - (xx + yy)],
				],
				dtype=float,
			)
		t = np.array([tx, ty, tz], dtype=float)
		C = -R.T @ t
		centers.append((name, C))
	return centers


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--sfm",
		type=Path,
		required=True,
		help="SfM model dir (contains cameras.bin/images.bin/points3D.bin)",
	)
	ap.add_argument(
		"--csv",
		type=Path,
		required=True,
		help="coordinates_5cols.csv (filename,lat,lon,meters,degrees,datetime)",
	)
	ap.add_argument(
		"--out",
		type=Path,
		default=Path("outputs/kulliye/geo_align.json"),
		help="Output JSON path",
	)
	ap.add_argument(
		"--poses",
		type=Path,
		default=None,
		help="Optional poses.txt (HLoc) to convert to lat/lon/alt",
	)
	ap.add_argument(
		"--poses_out",
		type=Path,
		default=None,
		help="Output CSV for converted poses (default: poses.geo.csv)",
	)
	args = ap.parse_args()

	gps = read_coordinates_csv(args.csv)
	rec = pycolmap.Reconstruction(args.sfm)
	if len(rec.images) == 0:
		raise RuntimeError("SfM model has 0 images")

	names: list[str] = []
	Xs = []
	Ls: list[LLA] = []
	for img in rec.images.values():
		if not img.name:
			continue
		if img.name not in gps:
			continue
		names.append(img.name)
		Xs.append(camera_center_from_image(img))
		Ls.append(gps[img.name])

	if len(Xs) < 3:
		raise RuntimeError(
			f"Need >=3 GPS-tagged images present in SfM. Found {len(Xs)}."
		)

	origin = mean_lla(Ls)
	X = np.stack(Xs, axis=0)
	Y = np.stack([ecef_to_enu(lla_to_ecef(gps[n]), origin) for n in names], axis=0)

	s, R, t, inliers, res = robust_fit_sim3(X, Y)
	in_res = res[inliers] if np.any(inliers) else res
	rmse = float(math.sqrt(float(np.mean(in_res**2))))

	out = {
		"sfm": str(args.sfm),
		"csv": str(args.csv),
		"origin": {"lat": origin.lat_deg, "lon": origin.lon_deg, "alt": origin.alt_m},
		"sim3": {"scale": float(s), "R": R.tolist(), "t": t.tolist()},
		"stats": {
			"num_used": int(len(names)),
			"num_inliers": int(int(inliers.sum())),
			"rmse_m": rmse,
			"median_m": float(np.median(in_res)),
			"max_m": float(np.max(in_res)),
		},
		"residuals_m": {names[i]: float(res[i]) for i in range(len(names))},
		"inliers": {names[i]: bool(inliers[i]) for i in range(len(names))},
	}

	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(out, indent=2))
	print(f"Wrote alignment: {args.out}")
	print(
		f"RMSE (inliers): {rmse:.2f} m | inliers {int(inliers.sum())}/{len(names)} | median {out['stats']['median_m']:.2f} m"
	)

	if args.poses is not None:
		poses_out = args.poses_out
		if poses_out is None:
			poses_out = args.poses.with_suffix(".geo.csv")
		centers = read_pose_file_centers(args.poses)
		with poses_out.open("w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["name", "lat", "lon", "alt", "east_m", "north_m", "up_m"])
			for name, C_sfm in centers:
				enu = (s * (C_sfm @ R.T)) + t
				ecef = enu_to_ecef(enu, origin)
				lla = ecef_to_lla(ecef)
				w.writerow([name, lla.lat_deg, lla.lon_deg, lla.alt_m, enu[0], enu[1], enu[2]])
		print(f"Wrote converted poses: {poses_out}")


if __name__ == "__main__":
	main()
