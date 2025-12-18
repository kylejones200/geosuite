"""
Core image processing module.

This module provides tools for processing and analyzing core photographs,
including cropping, rescaling, and extracting slabbed core images from
service company templates.
"""

from geosuite.imaging.core_processing import (
    crop_core_image,
    process_core_directory,
    extract_depth_from_filename,
)

__all__ = [
    'crop_core_image',
    'process_core_directory',
    'extract_depth_from_filename',
]


