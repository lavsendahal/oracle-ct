#!/usr/bin/env python3
"""
RadioPrior Pre-Packing Script with Multiprocessing

Batch processes NIFTI CT scans into training-ready .pt packs.
"""

import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import traceback

# Global variables for worker processes
_worker_packer = None
_worker_output_root = None
_worker_overwrite = None


def find_cases(image_root: Path, seg_root: Path, body_mask_root: Path,
               kidney_cyst_root: Path) -> List[Dict]:
    """Find all cases with matching files in ALL four directories (flat structure).

    Only processes cases where image, segmentation, body mask, AND kidney cyst
    all exist. This ensures a complete common set.
    """
    # Get case IDs from each directory
    image_ids = {p.name.replace(".nii.gz", "") for p in image_root.glob("*.nii.gz")}
    seg_ids = {p.name.replace(".nii.gz", "") for p in seg_root.glob("*.nii.gz")}
    body_ids = {p.name.replace(".nii.gz", "") for p in body_mask_root.glob("*.nii.gz")}
    cyst_ids = {p.name.replace(".nii.gz", "") for p in kidney_cyst_root.glob("*.nii.gz")}

    # Find common set (intersection of all four)
    common_ids = image_ids & seg_ids & body_ids & cyst_ids

    print(f"  Images: {len(image_ids)}")
    print(f"  Segmentations: {len(seg_ids)}")
    print(f"  Body masks: {len(body_ids)}")
    print(f"  Kidney cysts: {len(cyst_ids)}")
    print(f"  Common set: {len(common_ids)}")

    pairs = []
    for case_id in sorted(common_ids):
        pairs.append({
            "case_id": case_id,
            "image_path": image_root / f"{case_id}.nii.gz",
            "seg_path": seg_root / f"{case_id}.nii.gz",
            "body_mask_path": body_mask_root / f"{case_id}.nii.gz",
            "kidney_cyst_path": kidney_cyst_root / f"{case_id}.nii.gz",
        })

    return pairs


def worker_init(args_dict):
    """Initialize worker process."""
    global _worker_packer, _worker_output_root, _worker_overwrite

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from janus.datamodules.packing import RadioPriorPacker

    _worker_packer = RadioPriorPacker(
        target_spacing=tuple(args_dict["target_spacing"]),
        target_shape=tuple(args_dict["target_shape"]),
        hu_range=(args_dict["hu_min"], args_dict["hu_max"]),
    )

    _worker_output_root = Path(args_dict["output_root"])
    _worker_overwrite = args_dict["overwrite"]


def process_single_case(case_info: Dict):
    """Process a single case (called in worker).

    Returns:
        Tuple of (case_id, success, message, shape_info)
        shape_info is dict with shape_after_resampling if successful, else None
    """
    global _worker_packer, _worker_output_root, _worker_overwrite

    case_id = case_info["case_id"]
    output_path = _worker_output_root / f"{case_id}.pt"

    if output_path.exists() and not _worker_overwrite:
        return (case_id, True, "skipped", None)

    try:
        pack = _worker_packer.build_pack(
            image_path=case_info["image_path"],
            seg_path=case_info["seg_path"],
            case_id=case_id,
            body_mask_path=case_info.get("body_mask_path"),
            kidney_cyst_path=case_info.get("kidney_cyst_path"),
        )
        _worker_packer.save_pack(pack, output_path)

        # Extract shape info from pack metadata
        shape_info = {
            "case_id": case_id,
            "shape_after_resampling": pack["meta"].get("shape_after_resampling"),
        }
        return (case_id, True, "ok", shape_info)
    except Exception as e:
        return (case_id, False, f"{str(e)}\n{traceback.format_exc()}", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--seg_root", type=str, required=True)
    parser.add_argument("--body_mask_root", type=str, required=True)
    parser.add_argument("--kidney_cyst_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--target_shape", type=int, nargs=3, default=[224, 224, 224])
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=1000)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Paths
    image_root = Path(args.image_root)
    seg_root = Path(args.seg_root)
    body_mask_root = Path(args.body_mask_root)
    kidney_cyst_root = Path(args.kidney_cyst_root)
    output_root = Path(args.output_root)

    print(f"Image root: {image_root}")
    print(f"Seg root: {seg_root}")
    print(f"Body mask root: {body_mask_root}")
    print(f"Kidney cyst root: {kidney_cyst_root}")
    print(f"Output: {output_root}")
    print(f"Num workers: {args.num_workers}")
    
    # Find cases
    pairs = find_cases(image_root, seg_root, body_mask_root, kidney_cyst_root)
    print(f"\nFound {len(pairs)} cases")
    
    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)}")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Worker arguments
    worker_args = {
        "target_spacing": args.target_spacing,
        "target_shape": args.target_shape,
        "hu_min": args.hu_min,
        "hu_max": args.hu_max,
        "output_root": str(output_root),
        "overwrite": args.overwrite,
    }
    
    # Process with multiprocessing
    success = 0
    skipped = 0
    errors = []
    shape_records = []  # Collect shape info for CSV

    print(f"\nProcessing {len(pairs)} cases with {args.num_workers} workers...")

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=worker_init,
        initargs=(worker_args,),
    ) as executor:
        futures = {executor.submit(process_single_case, p): p["case_id"] for p in pairs}

        with tqdm(total=len(futures), desc="Packing") as pbar:
            for future in as_completed(futures):
                case_id, ok, msg, shape_info = future.result()

                if ok:
                    if msg == "skipped":
                        skipped += 1
                    else:
                        success += 1
                        # Collect shape info for newly processed cases
                        if shape_info and shape_info.get("shape_after_resampling"):
                            shape = shape_info["shape_after_resampling"]
                            shape_records.append({
                                "case_id": case_id,
                                "size_x": shape[0],
                                "size_y": shape[1],
                                "size_z": shape[2],
                            })
                else:
                    errors.append((case_id, msg))

                pbar.update(1)
                pbar.set_postfix({"ok": success, "skip": skipped, "err": len(errors)})

    print(f"\nDone: {success} success, {skipped} skipped, {len(errors)} errors")

    # Save shape info to CSV
    if shape_records:
        import pandas as pd
        shape_csv_path = output_root / "shape_after_resampling.csv"
        df_shapes = pd.DataFrame(shape_records)
        df_shapes = df_shapes.sort_values("case_id").reset_index(drop=True)
        df_shapes.to_csv(shape_csv_path, index=False)
        print(f"Shape info saved to {shape_csv_path} ({len(shape_records)} cases)")

    if errors:
        with open(output_root / "errors.log", "w") as f:
            for case_id, msg in errors:
                f.write(f"=== {case_id} ===\n{msg}\n\n")
        print(f"Errors saved to {output_root / 'errors.log'}")


if __name__ == "__main__":
    main()
