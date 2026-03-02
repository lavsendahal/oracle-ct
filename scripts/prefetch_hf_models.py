#!/usr/bin/env python3
"""
Pre-download HuggingFace models before DDP launch to avoid race conditions.
Reads experiment config from CLI args (same Hydra overrides as train.py),
finds any model_repo_id, and downloads it single-process.
"""

import sys
import re


def main():
    # Extract model_repo_id from CLI args (e.g. model.model_repo_id=YalaLab/Pillar0-AbdomenCT)
    repo_id = None
    for arg in sys.argv[1:]:
        m = re.match(r"model\.model_repo_id=(.+)", arg)
        if m:
            repo_id = m.group(1)
            break

    # Also check experiment config files for pillar experiments
    is_pillar = any("pillar" in arg for arg in sys.argv[1:])

    if repo_id is None and is_pillar:
        # Default Pillar-0 AbdomenCT repo
        repo_id = "YalaLab/Pillar0-AbdomenCT"

    if repo_id is None:
        # Not a HuggingFace model experiment — nothing to prefetch
        return

    print(f"[prefetch] Downloading {repo_id} to HF cache (single process)...")
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        print(f"[prefetch] Done — {repo_id} is cached.")
    except Exception as e:
        print(f"[prefetch] WARNING: prefetch failed: {e}")
        print("[prefetch] Continuing anyway — DDP will attempt to load.")


if __name__ == "__main__":
    main()
