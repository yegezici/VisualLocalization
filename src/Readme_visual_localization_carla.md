# CARLA Live Visual Localization

This setup runs CARLA data capture and visual localization as two separate processes:

- `visual_localization.py` runs in the CARLA Python 3.6 environment.
- `localization_server.py` runs in a modern Python environment with PyTorch, HLoc, LightGlue, NetVLAD, SuperPoint, PyCOLMAP, and h5py.

The split is intentional. CARLA's Python API is tied to Python 3.6, while the visual localization stack works much better in a newer Python environment.

## Files

- `visual_localization.py`
  - CARLA + Pygame async client.
  - Captures and displays the live single-camera stream.
  - On `L`, sends the current RGB frame to the localization server over TCP.
  - Converts returned CARLA `x/y/z` to latitude, longitude, altitude.
  - Prints estimated pose, ground truth pose, position error, and heading error.

- `localization_server.py`
  - TCP server for visual localization.
  - Loads the map bundle and neural models once at startup.
  - Receives RGB frames from `visual_localization.py`.
  - Runs in-memory NetVLAD retrieval, SuperPoint extraction, LightGlue matching, and PyCOLMAP PnP.
  - Returns estimated CARLA `x/y/z`, heading, inlier count, and retrieved references.

- `carla_live_localization.py`
  - Shared localization implementation used by the server.
  - Contains COLMAP/CARLA coordinate conversion logic.
  - Does not write query images, query features, matches, or pairs to disk.

- `sim-19-may-bundle/`
  - Known-pose reconstruction bundle.
  - Expected files include:
    - `sfm/`
    - `features.h5`
    - `global-feats-netvlad.h5`
    - `metadata.json`

## Environment 1: CARLA Client

Use your existing CARLA Python 3.6 environment.

```bash
conda activate carla3.6
cd /media/ilker/ubuntu-disk1/Carla_Intersection/PythonAPI/examples
```

This environment only needs CARLA, pygame, and numpy. It does not need PyTorch, HLoc, PyCOLMAP, or h5py.

Run the CARLA-side script after starting the localization server:

```bash
python visual_localization.py --loc-host 127.0.0.1 --loc-port 5555
```

Useful options:

```bash
python visual_localization.py \
  --camera-rig single \
  --preview-layout single \
  --preview-camera front \
  --res 1920x1080 \
  --loc-host 127.0.0.1 \
  --loc-port 5555 \
  --loc-timeout 120
```

Controls:

- `W/A/S/D`: drive the vehicle.
- `SPACE`: save the latest camera image and pose metadata.
- `L`: localize the current camera frame.
- `Q` or `ESC`: quit.

## Environment 2: Visual Localization Server

Use a modern Python environment. Python 3.10 is recommended.

```bash
conda create -n visloc python=3.10 -y
conda activate visloc
cd /media/ilker/ubuntu-disk1/Carla_Intersection/PythonAPI/examples
```

Install core dependencies:

```bash
pip install -r requirements_localization_server.txt
```

If you need a CUDA-specific PyTorch wheel, install PyTorch first using the matching command from the PyTorch website, then install the remaining requirements.

Install HLoc from a recursive clone. This is important because SuperPoint depends on HLoc's third-party `SuperGluePretrainedNetwork` folder.

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
pip uninstall -y hloc
pip install -e Hierarchical-Localization
export HLOC_THIRD_PARTY=$PWD/Hierarchical-Localization/third_party
```

Run the server:

```bash
python localization_server.py --bundle sim-19-may-bundle --host 127.0.0.1 --port 5555 --retrieval-max-size 1024
```

If GPU memory is tight, reduce the retrieval image size:

```bash
python localization_server.py --bundle sim-19-may-bundle --host 127.0.0.1 --port 5555 --retrieval-max-size 768
```

Server options:

- `--bundle`: path to the map bundle. Default: `sim-19-may-bundle`.
- `--host`: TCP bind host. Default: `127.0.0.1`.
- `--port`: TCP port. Default: `5555`.
- `--num-loc`: number of retrieved reference images. Default: `10`.
- `--max-error`: PyCOLMAP RANSAC reprojection threshold. Default: `12.0`.
- `--retrieval-max-size`: longest image side used for NetVLAD retrieval. Default: `1024`.
- `--device`: force `cuda` or `cpu`.

## Run Order

Start CARLA first, then start the localization server, then start the CARLA client script.

Terminal 1, localization server:

```bash
conda activate visloc
cd /media/ilker/ubuntu-disk1/Carla_Intersection/PythonAPI/examples
export HLOC_THIRD_PARTY=$PWD/Hierarchical-Localization/third_party
python localization_server.py --bundle sim-19-may-bundle --host 127.0.0.1 --port 5555 --retrieval-max-size 1024
```

Terminal 2, CARLA client:

```bash
conda activate carla3.6
cd /media/ilker/ubuntu-disk1/Carla_Intersection/PythonAPI/examples
python visual_localization.py --loc-host 127.0.0.1 --loc-port 5555
```

Press `L` in the Pygame window to run localization.

## How It Works

`visual_localization.py` runs CARLA in asynchronous mode. Sensor callbacks do not block the simulator; each camera callback only overwrites the latest received image. The Pygame loop runs at 30 FPS and reads the latest image when it needs to render or localize.

When `L` is pressed:

1. The CARLA process copies the latest camera frame.
2. It also reads the ground-truth camera transform associated with that image.
3. It sends the RGB frame to `localization_server.py` over a small TCP protocol:
   - width
   - height
   - raw RGB bytes
4. The CARLA/Pygame loop continues running while the request is handled in a background thread.

The localization server:

1. Loads `pycolmap.Reconstruction` from `sim-19-may-bundle/sfm` once at startup.
2. Loads reference local features from `features.h5` once.
3. Loads reference NetVLAD descriptors from `global-feats-netvlad.h5` once.
4. Loads SuperPoint, NetVLAD, and LightGlue once into memory/VRAM.
5. Converts the received RGB frame to tensors in memory:
   - RGB tensor for NetVLAD retrieval.
   - grayscale tensor for SuperPoint local features.
6. Retrieves the nearest reference images with NetVLAD.
7. Matches query SuperPoint features to reference SuperPoint features with LightGlue.
8. Builds 2D-3D correspondences from matched reference keypoints and COLMAP point IDs.
9. Runs PyCOLMAP absolute pose estimation/refinement.
10. Converts the COLMAP right-handed pose back to CARLA left-handed coordinates.
11. Returns estimated CARLA `x/y/z`, heading, inlier count, and retrieved references.

Back in `visual_localization.py`:

1. The returned CARLA `x/y/z` is converted to latitude, longitude, and altitude with:

```python
world.get_map().transform_to_geolocation(carla.Location(x, y, z))
```

2. Ground truth is computed from the image-aligned CARLA camera transform.
3. The script prints:
   - estimated localization result
   - ground truth
   - position error in meters
   - heading error in degrees
   - `dx/dy/dz`

Example output:

```text
[Lozalization Started]
[Localization Result] lat=40.90674868 lon=29.15503751 alt=2.205 heading=113.04deg x=6.025 y=4.133 z=2.205 inliers=1177
[Localization Ground Truth] lat=40.90674870 lon=29.15503748 alt=2.205 heading=112.91deg x=6.028 y=4.130 z=2.205
[Localization Error] position=0.004m heading=0.13deg dx=-0.003 dy=0.003 dz=0.000
```

## Coordinate Conversion

The localization server uses the same CARLA-to-COLMAP conversion matrix as the notebook:

```python
CARLA_TO_COLMAP_S = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)
```

PyCOLMAP returns a right-handed world-to-camera pose. The server converts this back to CARLA's left-handed camera-to-world convention before extracting heading:

```python
r_cw_rh = r_wc_rh.T
r_cw_lh = COLMAP_TO_CARLA_S @ r_cw_rh @ CARLA_TO_COLMAP_S
heading = atan2(r_cw_lh[1, 0], r_cw_lh[0, 0])
```

Position is converted back from COLMAP world coordinates to CARLA world coordinates before being returned to the CARLA process.

## Performance Notes

- The neural models are loaded once at server startup.
- Query features and matches are never written to disk.
- The CARLA process never imports PyTorch, HLoc, PyCOLMAP, or h5py.
- NetVLAD retrieval uses `--retrieval-max-size` to control GPU memory.
- SuperPoint still runs on the full-resolution frame for accurate local features.
- The server serializes inference with a lock to avoid overlapping CUDA allocations.
- CUDA cache is cleared after each request to reduce repeated-localization OOM risk.

## Troubleshooting

Missing `SuperGluePretrainedNetwork`:

```text
ModuleNotFoundError: No module named 'SuperGluePretrainedNetwork'
```

Fix:

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
pip uninstall -y hloc
pip install -e Hierarchical-Localization
export HLOC_THIRD_PARTY=$PWD/Hierarchical-Localization/third_party
```

CUDA out of memory:

```bash
python localization_server.py --bundle sim-19-may-bundle --retrieval-max-size 768
```

If needed, go lower:

```bash
python localization_server.py --bundle sim-19-may-bundle --retrieval-max-size 640
```

Server not reachable from CARLA client:

- Make sure the server is running before pressing `L`.
- Make sure both scripts use the same host and port.
- Default is `127.0.0.1:5555`.

No localization result:

- Check that `sim-19-may-bundle` contains `sfm/`, `features.h5`, and `global-feats-netvlad.h5`.
- Check that the live camera view overlaps with the mapped area.
- Increase `--num-loc` if retrieval is weak.
- Increase `--max-error` if PnP has too few inliers.

