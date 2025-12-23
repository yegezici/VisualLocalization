"""Minimal single-image localization against an existing SfM model.

Goals:
- Do NOT re-process the mapping dataset.
- Reuse mapping outputs: SfM model + DB local features + DB global descriptors.
- Only compute query descriptors/features if they are missing (unless OVERWRITE=True).
"""

from pathlib import Path
import shutil
import uuid

import pycolmap

from hloc import extract_features, localize_sfm, match_features, pairs_from_retrieval


# --- CONFIGURATION (edit these) ---
outputs = Path('outputs/kulliye/')
sfm_dir = outputs / 'sfm'
query_dir = Path('query')
query_name = 'IMG_20251212_153550.jpg'
top_k = 50
OVERWRITE = False
KEEP_CACHE = False  # if False, only the pose output is kept

# Must match your mapping pipeline
feature_conf = extract_features.confs['aliked-n16']
retrieval_conf = extract_features.confs['netvlad']
matcher_conf = match_features.confs['aliked+lightglue']

# Mapping artifacts (must already exist)
features_db = outputs / f"{feature_conf['output']}.h5"
global_db = outputs / f"{retrieval_conf['output']}.h5"

# Output directory
loc_outputs = outputs / 'localization'
loc_outputs.mkdir(exist_ok=True)
results = loc_outputs / 'poses.txt'


def _select_sfm_model(root: Path) -> Path:
    models_dir = root / 'models'
    if not models_dir.exists():
        return root
    best_dir = None
    best_n = -1
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir():
            continue
        required = ['cameras.bin', 'images.bin', 'points3D.bin']
        if not all((sub / f).exists() for f in required):
            continue
        try:
            rec = pycolmap.Reconstruction(sub)
        except Exception:
            continue
        if len(rec.images) > best_n:
            best_n = len(rec.images)
            best_dir = sub
    return best_dir or root


def main():
    if not sfm_dir.exists():
        raise FileNotFoundError(f"SfM model not found: {sfm_dir}")
    if not features_db.exists():
        raise FileNotFoundError(f"DB local features missing: {features_db}")
    if not global_db.exists():
        raise FileNotFoundError(
            f"DB global descriptors missing: {global_db}\n"
            "Run mapping once with NetVLAD to create it (global-feats-netvlad.h5)."
        )

    query_path = query_dir / query_name
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    sfm_model = _select_sfm_model(sfm_dir)
    ref = pycolmap.Reconstruction(sfm_model)
    if len(ref.images) == 0 or len(ref.cameras) == 0:
        raise RuntimeError("Reference reconstruction is empty.")

    db_names = [img.name for img in ref.images.values()]
    k = min(top_k, len(db_names))

    # Put intermediate artifacts into a temp directory unless the user wants caching.
    # This keeps the workspace clean and prevents accumulation across different queries.
    qtag = Path(query_name).stem
    if KEEP_CACHE:
        work_dir = loc_outputs
        global_query = work_dir / f'global_query_{qtag}.h5'
        features_query = work_dir / f'query_features_{qtag}.h5'
        retrieval_pairs = work_dir / f'retrieval_{qtag}_top{k}.txt'
        matches_query_db = work_dir / f'query_db_matches_{qtag}.h5'
        queries_list = work_dir / f'queries_with_intrinsics_{qtag}.txt'
        cleanup_dir = None
    else:
        work_dir = loc_outputs / f"tmp_localize_{qtag}_{uuid.uuid4().hex[:8]}"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        global_query = work_dir / 'global_query.h5'
        features_query = work_dir / 'query_features.h5'
        retrieval_pairs = work_dir / f'retrieval_top{k}.txt'
        matches_query_db = work_dir / 'query_db_matches.h5'
        queries_list = work_dir / 'queries_with_intrinsics.txt'
        cleanup_dir = work_dir

    # Intrinsics: reuse first camera (minimal; replace with your real intrinsics if needed)
    ref_cam = next(iter(ref.cameras.values()))
    model_str = getattr(ref_cam, 'model_name', None) or str(ref_cam.model)
    if '.' in model_str:
        model_str = model_str.split('.')[-1]
    params = ' '.join(map(str, ref_cam.params))
    try:
        queries_list.write_text(
            f"{query_name} {model_str} {ref_cam.width} {ref_cam.height} {params}\n"
        )

        # 1) Retrieval (query descriptor is computed only if missing)
        extract_features.main(
            retrieval_conf,
            query_dir,
            image_list=[query_name],
            feature_path=global_query,
            overwrite=OVERWRITE,
        )
        pairs_from_retrieval.main(
            descriptors=global_query,
            output=retrieval_pairs,
            num_matched=k,
            db_model=sfm_model,
            db_descriptors=global_db,
        )

        # 2) Query local features + matching to DB
        extract_features.main(
            feature_conf,
            query_dir,
            image_list=[query_name],
            feature_path=features_query,
            overwrite=OVERWRITE,
        )
        match_features.main(
            matcher_conf,
            retrieval_pairs,
            features=features_query,
            features_ref=features_db,
            matches=matches_query_db,
            overwrite=OVERWRITE,
        )

        # 3) PnP
        localize_sfm.main(
            reference_sfm=sfm_model,
            queries=queries_list,
            retrieval=retrieval_pairs,
            features=features_query,
            matches=matches_query_db,
            results=results,
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    print(f"Pose written to: {results}")


if __name__ == '__main__':
    main()