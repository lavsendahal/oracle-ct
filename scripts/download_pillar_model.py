#!/usr/bin/env python3
"""
Download Pillar0-AbdomenCT from HuggingFace and save to local directory.

Usage:
    python download_pillar_model.py
    python download_pillar_model.py --dest /custom/path
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "YalaLab/Pillar0-AbdomenCT"
DEFAULT_DEST = "/scratch/railabs/ld258/output/ct_triage/oracle-ct/pretrained_models/Pillar0-AbdomenCT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Local directory to save the model (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HUGGINGFACE_HUB_TOKEN env var)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} → {dest}")
    print("This may take a while for a large 3D model...")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        token=args.token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print(f"\nDone. Model saved to: {dest}")
    print(f"\nUse in training with:")
    print(f"  model.model_repo_id={dest}")


if __name__ == "__main__":
    main()
