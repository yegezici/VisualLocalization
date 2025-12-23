from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)

# --- CONFIGURATION ---
images_path = Path('datasets/12-12-2025/')  # Reference images (mapping set)
outputs = Path('outputs/kulliye/')  # Where to write SfM artifacts

# SfM mapping settings
# - For accuracy, prefer retrieval-based pairing to avoid many weak/outlier matches.
# - Set FORCE_RECOMPUTE=True when you changed the dataset or configs.
PAIRING = 'retrieval'  # 'retrieval' | 'exhaustive'
NUM_MATCHED = 50  # top-K retrieval pairs per image (typical: 20-50)
FORCE_RECOMPUTE = False

# Create output directory if it doesn't exist
outputs.mkdir(parents=True, exist_ok=True)

# Choose models (pretrained)
# Local features + matcher (recommended for higher accuracy)
local_conf = extract_features.confs['aliked-n16']
matcher_conf = match_features.confs['aliked+lightglue']

# Global descriptors for retrieval pairing
global_conf = extract_features.confs['netvlad']

# Define file paths (include conf names to avoid accidental mixing)
sfm_dir = outputs / 'sfm'
local_features = outputs / f"{local_conf['output']}.h5"
global_descriptors = outputs / f"{global_conf['output']}.h5"
sfm_pairs = outputs / (
    f"pairs-sfm-{PAIRING}{'' if PAIRING != 'retrieval' else f'-netvlad{NUM_MATCHED}'}.txt"
)
matches = outputs / f"{matcher_conf['output']}.h5"

def run_mapping():
    print(f"1. Extracting local features ({local_conf['model']['name']})...")
    extract_features.main(
        local_conf,
        images_path,
        feature_path=local_features,
        overwrite=FORCE_RECOMPUTE,
    )

    print(f"2. Generating pairs ({PAIRING})...")
    if PAIRING == 'exhaustive':
        pairs_from_exhaustive.main(sfm_pairs, features=local_features)
    elif PAIRING == 'retrieval':
        print(f"2a. Extracting global descriptors ({global_conf['model']['name']})...")
        extract_features.main(
            global_conf,
            images_path,
            feature_path=global_descriptors,
            overwrite=FORCE_RECOMPUTE,
        )
        pairs_from_retrieval.main(
            descriptors=global_descriptors,
            output=sfm_pairs,
            num_matched=NUM_MATCHED,
        )
    else:
        raise ValueError(f"Unknown PAIRING={PAIRING!r}. Use 'retrieval' or 'exhaustive'.")

    print(f"3. Matching features ({matcher_conf['model']['name']})...")
    match_features.main(
        matcher_conf,
        sfm_pairs,
        features=local_features,
        matches=matches,
        overwrite=FORCE_RECOMPUTE,
    )

    print("4. Running SfM reconstruction (COLMAP)...")
    reconstruction.main(sfm_dir, images_path, sfm_pairs, local_features, matches)

    print(f"Mapping complete! Model saved at: {sfm_dir}")

if __name__ == '__main__':
    run_mapping()