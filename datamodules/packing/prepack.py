#Copyright 2026 LAVSEN DAHAL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#!/usr/bin/env python3
# janus/datamodules/packing/prepack.py
"""
Batch Pack Building CLI for Janus Pipeline

This script processes multiple CT scans in parallel, creating .pt pack files
for the deep learning training pipeline.

USAGE:
    # Basic usage (auto-discover cases from directory)
    python prepack.py \
        --images_root /path/to/images \
        --segs_root /path/to/segmentations \
        --output_dir /path/to/packs \
        --features_parquet /path/to/radioprior_features_v1.parquet

    # With body masks
    python prepack.py \
        --images_root /path/to/images \
        --segs_root /path/to/segmentations \
        --body_masks_root /path/to/body_masks \
        --output_dir /path/to/packs \
        --features_parquet /path/to/features.parquet

    # Process specific cases from CSV
    python prepack.py \
        --case_list /path/to/cases.csv \
        --images_root /path/to/images \
        --segs_root /path/to/segmentations \
        --output_dir /path/to/packs

    # With custom spacing and workers
    python prepack.py \
        --images_root /data/ct_images \
        --segs_root /data/totalseg \
        --output_dir /data/packs \
        --spacing 1.5 1.5 1.5 \
        --workers 16

DIRECTORY STRUCTURE:
    Expected input structure:
        images_root/
            {case_id}.nii.gz
            or {case_id}/ct.nii.gz
        
        segs_root/
            {case_id}.nii.gz
            or {case_id}/segmentation.nii.gz
        
        body_masks_root/ (optional)
            {case_id}.nii.gz
            or {case_id}/body.nii.gz

    Output structure:
        output_dir/
            {case_id}.pt
            index.json  # Metadata about all packs
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress nibabel warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch

try:
    import pandas as pd
except ImportError:
    pd = None


def find_nifti(root: Path, case_id: str, patterns: List[str]) -> Optional[Path]:
    """
    Find NIFTI file for a case using common patterns.
    
    Tries patterns in order:
    1. {root}/{case_id}.nii.gz
    2. {root}/{case_id}.nii
    3. {root}/{case_id}/{pattern}.nii.gz for each pattern
    """
    # Direct file
    for ext in [".nii.gz", ".nii"]:
        path = root / f"{case_id}{ext}"
        if path.exists():
            return path
    
    # Subdirectory with patterns
    case_dir = root / case_id
    if case_dir.is_dir():
        for pattern in patterns:
            for ext in [".nii.gz", ".nii"]:
                path = case_dir / f"{pattern}{ext}"
                if path.exists():
                    return path
    
    return None


def discover_cases(
    images_root: Path,
    segs_root: Path,
    body_masks_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Discover all valid cases from directory structure.
    
    Returns list of dicts with case_id, image_path, seg_path, body_mask_path
    """
    cases = []
    
    # Patterns to search for
    image_patterns = ["ct", "image", "CT", "volume"]
    seg_patterns = ["segmentation", "seg", "totalseg", "labels"]
    body_patterns = ["body", "body_mask", "mask"]
    
    # Find all potential case IDs from images directory
    potential_ids = set()
    
    for item in images_root.iterdir():
        if item.is_file() and item.suffix in [".gz", ".nii"]:
            # Remove .nii.gz or .nii extension
            name = item.name.replace(".nii.gz", "").replace(".nii", "")
            potential_ids.add(name)
        elif item.is_dir():
            potential_ids.add(item.name)
    
    # Validate each case
    for case_id in sorted(potential_ids):
        image_path = find_nifti(images_root, case_id, image_patterns)
        seg_path = find_nifti(segs_root, case_id, seg_patterns)
        
        if image_path is None:
            continue
        if seg_path is None:
            continue
        
        body_path = None
        if body_masks_root is not None:
            body_path = find_nifti(body_masks_root, case_id, body_patterns)
        
        cases.append({
            "case_id": case_id,
            "image_path": str(image_path),
            "seg_path": str(seg_path),
            "body_mask_path": str(body_path) if body_path else None,
        })
    
    return cases


def load_case_list(csv_path: Path) -> List[str]:
    """Load case IDs from a CSV file (expects 'case_id' column)."""
    if pd is None:
        raise ImportError("pandas required for case list: pip install pandas")
    
    df = pd.read_csv(csv_path)
    if "case_id" not in df.columns:
        # Try first column
        return df.iloc[:, 0].astype(str).tolist()
    return df["case_id"].astype(str).tolist()


def process_single_case(
    case_info: Dict[str, Any],
    output_dir: Path,
    features_df: Optional["pd.DataFrame"],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single case (designed to run in subprocess).
    
    Returns dict with status and timing information.
    """
    # Import here to avoid pickling issues with multiprocessing
    from .packer import PackBuilder, PackConfig
    
    case_id = case_info["case_id"]
    output_path = output_dir / f"{case_id}.pt"
    
    # Skip if already exists
    if output_path.exists():
        return {
            "case_id": case_id,
            "status": "skipped",
            "reason": "already exists",
        }
    
    start_time = time.time()
    
    try:
        # Rebuild config from dict
        config = PackConfig(**config_dict)
        
        # Build pack
        builder = PackBuilder(config=config, features_df=features_df)
        pack = builder.build(
            case_id=case_id,
            image_path=case_info["image_path"],
            seg_path=case_info["seg_path"],
            body_mask_path=case_info.get("body_mask_path"),
        )
        
        # Validate
        warnings_list = builder.validate_pack(pack)
        
        # Save
        torch.save(pack, str(output_path))
        
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        return {
            "case_id": case_id,
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "file_size_mb": round(file_size_mb, 2),
            "shape": list(pack["image"].shape),
            "warnings": warnings_list,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "case_id": case_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(elapsed, 2),
        }


def run_parallel(
    cases: List[Dict[str, Any]],
    output_dir: Path,
    features_df: Optional["pd.DataFrame"],
    config_dict: Dict[str, Any],
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Process cases in parallel using ProcessPoolExecutor.
    """
    results = []
    
    # For single worker, run sequentially (easier debugging)
    if num_workers == 1:
        for i, case_info in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] Processing {case_info['case_id']}...")
            result = process_single_case(
                case_info, output_dir, features_df, config_dict
            )
            results.append(result)
            
            if result["status"] == "error":
                print(f"  ERROR: {result['error']}")
            elif result["status"] == "success":
                print(f"  Done in {result['elapsed_sec']}s, {result['file_size_mb']} MB")
        
        return results
    
    # Parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_case,
                case_info,
                output_dir,
                features_df,
                config_dict,
            ): case_info["case_id"]
            for case_info in cases
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            case_id = futures[future]
            
            try:
                result = future.result()
                results.append(result)
                
                status = result["status"]
                if status == "success":
                    print(
                        f"[{completed}/{len(cases)}] {case_id}: "
                        f"{result['elapsed_sec']}s, {result['file_size_mb']} MB"
                    )
                elif status == "skipped":
                    print(f"[{completed}/{len(cases)}] {case_id}: skipped")
                else:
                    print(f"[{completed}/{len(cases)}] {case_id}: ERROR - {result['error']}")
                    
            except Exception as e:
                print(f"[{completed}/{len(cases)}] {case_id}: EXCEPTION - {e}")
                results.append({
                    "case_id": case_id,
                    "status": "error",
                    "error": str(e),
                })
    
    return results


def save_index(
    results: List[Dict[str, Any]],
    output_dir: Path,
    config_dict: Dict[str, Any],
) -> None:
    """Save index.json with metadata about all packs."""
    # Compute summary statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    index = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config_dict,
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
        },
        "cases": {r["case_id"]: r for r in results},
    }
    
    if successful:
        sizes = [r["file_size_mb"] for r in successful]
        times = [r["elapsed_sec"] for r in successful]
        index["summary"]["avg_size_mb"] = round(np.mean(sizes), 2)
        index["summary"]["total_size_mb"] = round(sum(sizes), 2)
        index["summary"]["avg_time_sec"] = round(np.mean(times), 2)
    
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\nIndex saved to {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch pack building for Janus pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input paths
    parser.add_argument(
        "--images_root", "-i",
        type=Path,
        required=True,
        help="Root directory containing CT images"
    )
    parser.add_argument(
        "--segs_root", "-s",
        type=Path,
        required=True,
        help="Root directory containing TotalSegmentator segmentations"
    )
    parser.add_argument(
        "--body_masks_root", "-b",
        type=Path,
        default=None,
        help="Optional: Root directory containing body masks"
    )
    parser.add_argument(
        "--features_parquet", "-f",
        type=Path,
        default=None,
        help="Optional: Parquet file with pre-extracted Janus features"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for .pt pack files"
    )
    
    # Case selection
    parser.add_argument(
        "--case_list",
        type=Path,
        default=None,
        help="Optional: CSV file with case_id column to process specific cases"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Limit number of cases to process (for testing)"
    )
    
    # Processing options
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.5, 1.5, 1.5],
        metavar=("X", "Y", "Z"),
        help="Target isotropic spacing in mm (default: 1.5 1.5 1.5)"
    )
    parser.add_argument(
        "--hu_min",
        type=float,
        default=-1000.0,
        help="Minimum HU value for clipping (default: -1000)"
    )
    parser.add_argument(
        "--hu_max",
        type=float,
        default=400.0,
        help="Maximum HU value for clipping (default: 400)"
    )
    parser.add_argument(
        "--merge_name",
        type=str,
        default="radioprior_v1",
        help="Organ merge profile name (default: radioprior_v1)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pack files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.images_root.exists():
        print(f"Error: images_root not found: {args.images_root}")
        sys.exit(1)
    if not args.segs_root.exists():
        print(f"Error: segs_root not found: {args.segs_root}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features if provided
    features_df = None
    if args.features_parquet is not None:
        if pd is None:
            print("Error: pandas required for features parquet")
            sys.exit(1)
        print(f"Loading features from {args.features_parquet}...")
        features_df = pd.read_parquet(args.features_parquet)
        print(f"  Loaded {len(features_df)} cases, {len(features_df.columns)} features")
    
    # Discover or load cases
    if args.case_list is not None:
        print(f"Loading case list from {args.case_list}...")
        case_ids = load_case_list(args.case_list)
        
        # Build case info for each ID
        cases = []
        for case_id in case_ids:
            image_path = find_nifti(
                args.images_root, case_id, ["ct", "image", "CT", "volume"]
            )
            seg_path = find_nifti(
                args.segs_root, case_id, ["segmentation", "seg", "totalseg"]
            )
            body_path = None
            if args.body_masks_root:
                body_path = find_nifti(
                    args.body_masks_root, case_id, ["body", "body_mask"]
                )
            
            if image_path and seg_path:
                cases.append({
                    "case_id": case_id,
                    "image_path": str(image_path),
                    "seg_path": str(seg_path),
                    "body_mask_path": str(body_path) if body_path else None,
                })
            else:
                print(f"  Warning: Missing files for {case_id}")
    else:
        print("Discovering cases from directory structure...")
        cases = discover_cases(
            args.images_root,
            args.segs_root,
            args.body_masks_root,
        )
    
    print(f"Found {len(cases)} valid cases")
    
    # Apply limit if specified
    if args.limit is not None:
        cases = cases[:args.limit]
        print(f"Limited to {len(cases)} cases")
    
    if len(cases) == 0:
        print("No cases to process!")
        sys.exit(0)
    
    # Filter already processed (unless overwrite)
    if not args.overwrite:
        original_count = len(cases)
        cases = [
            c for c in cases
            if not (args.output_dir / f"{c['case_id']}.pt").exists()
        ]
        skipped = original_count - len(cases)
        if skipped > 0:
            print(f"Skipping {skipped} already processed cases")
    
    if len(cases) == 0:
        print("All cases already processed!")
        sys.exit(0)
    
    # Build config dict (for serialization)
    config_dict = {
        "target_spacing": tuple(args.spacing),
        "hu_min": args.hu_min,
        "hu_max": args.hu_max,
        "merge_name": args.merge_name,
        "apply_body_mask": args.body_masks_root is not None,
    }
    
    print(f"\nConfiguration:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    
    print(f"\nProcessing {len(cases)} cases with {args.workers} workers...")
    start_time = time.time()
    
    # Run processing
    results = run_parallel(
        cases,
        args.output_dir,
        features_df,
        config_dict,
        num_workers=args.workers,
    )
    
    total_time = time.time() - start_time
    
    # Save index
    save_index(results, args.output_dir, config_dict)
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed cases:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['case_id']}: {r['error']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
