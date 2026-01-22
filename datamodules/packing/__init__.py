# janus/datamodules/packing/__init__.py
"""
Pack Building Utilities

This subpackage provides tools for building standardized .pt pack files
from raw NIFTI CT data:

- packer: PackBuilder class and PackConfig
- prepack: CLI for batch pack building

Example usage:

    from datamodules.packing import PackBuilder, PackConfig

    config = PackConfig(
        target_spacing=(1.5, 1.5, 1.5),
        merge_name="janus_v1",
    )

    builder = PackBuilder(config=config)
    pack = builder.build(
        case_id="AC123",
        image_path="/path/to/ct.nii.gz",
        seg_path="/path/to/seg.nii.gz",
    )
"""

from .packer import (
    PackBuilder,
    PackConfig,
    JanusPacker,
    build_pack,
    load_pack,
    estimate_pack_size_mb,
)

__all__ = [
    "PackBuilder",
    "PackConfig",
    "JanusPacker",
    "build_pack",
    "load_pack",
    "estimate_pack_size_mb",
]
