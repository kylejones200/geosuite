"""
Core image processing utilities for extracting and cleaning core photographs.

This module provides tools to extract slabbed core photographs from image
templates produced by service companies, crop and rescale images, and
extract depth information from filenames.

Based on workflows for processing Pharos-1 core images from the Poseidon
dataset (licensed CC-BY 3.0 by Geoscience Australia and ConocoPhillips).
"""

import os
import logging
import glob
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
from skimage import io as skio
from skimage.transform import rescale

logger = logging.getLogger(__name__)


def extract_depth_from_filename(filename: str, top_start: int = 21, top_end: int = 28,
                                 bottom_start: int = 33, bottom_end: int = 40) -> Tuple[str, str]:
    """
    Extract top and bottom depth from core image filename.
    
    Args:
        filename: Core image filename
        top_start: Start index for top depth in filename
        top_end: End index for top depth in filename
        bottom_start: Start index for bottom depth in filename
        bottom_end: End index for bottom depth in filename
        
    Returns:
        Tuple of (top_depth, bottom_depth) as strings
        
    Example:
        >>> extract_depth_from_filename('pharos-1_wl_1m_core_4963_00m_to_4964_00m.jpg')
        ('4963_00', '4964_00')
    """
    basename = os.path.basename(filename)
    top_depth = basename[top_start:top_end]
    bottom_depth = basename[bottom_start:bottom_end]
    return top_depth, bottom_depth


def crop_core_image(image: np.ndarray, top: int, left: int, bottom: int, 
                     right: int, scale_factor: float = 0.3) -> np.ndarray:
    """
    Crop and rescale a core image to extract the core slab region.
    
    Args:
        image: Input image as numpy array (H, W, C)
        top: Distance in pixels from top of image to top of core
        left: Distance in pixels from left edge to left of core
        bottom: Distance in pixels from top of image to bottom of core
        right: Distance in pixels from left of image to right of core
        scale_factor: Rescaling factor for reducing image resolution (default 0.3)
        
    Returns:
        Cropped and rescaled image as numpy array
        
    Example:
        >>> img = skio.imread('core_photo.jpg')
        >>> cropped = crop_core_image(img, top=642, left=630, bottom=12520, right=1570)
    """
    # Crop to core region
    img_cropped = image[top:bottom, left:right]
    
    # Rescale to reduce file size
    img_rescaled = rescale(
        img_cropped, 
        scale_factor, 
        multichannel=True,
        anti_aliasing=True,
        mode='constant',
        channel_axis=2
    )
    
    # Convert back to uint8 for saving
    img_rescaled = (img_rescaled * 255).astype(np.uint8)
    
    return img_rescaled


def process_core_directory(
    input_dir: str,
    output_dir: str,
    top: int = 642,
    left: int = 630,
    bottom: int = 12520,
    right: int = 1570,
    scale_factor: float = 0.3,
    pattern: str = '*.jpg',
    top_start: int = 21,
    top_end: int = 28,
    bottom_start: int = 33,
    bottom_end: int = 40
) -> List[str]:
    """
    Process all core images in a directory: crop, rescale, and save with depth labels.
    
    Args:
        input_dir: Directory containing raw core images
        output_dir: Directory to save processed images
        top: Distance in pixels from top of image to top of core
        left: Distance in pixels from left edge to left of core
        bottom: Distance in pixels from top of image to bottom of core
        right: Distance in pixels from left of image to right of core
        scale_factor: Rescaling factor for reducing resolution (default 0.3)
        pattern: Glob pattern for input files (default '*.jpg')
        top_start: Start index for top depth in filename
        top_end: End index for top depth in filename
        bottom_start: Start index for bottom depth in filename
        bottom_end: End index for bottom depth in filename
        
    Returns:
        List of processed output filenames
        
    Example:
        >>> processed = process_core_directory(
        ...     'data/raw/pharos-1_wl/',
        ...     'data/cleaned/WL/',
        ...     top=642, left=630, bottom=12520, right=1570
        ... )
        >>> print(f"Processed {len(processed)} images")
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    input_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not input_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
        return processed_files
    
    logger.info(f"Processing {len(input_files)} images from {input_dir}")
    
    for full_filename in input_files:
        try:
            # Extract depth from filename
            top_depth, bottom_depth = extract_depth_from_filename(
                full_filename, top_start, top_end, bottom_start, bottom_end
            )
            
            # Read image
            img = skio.imread(full_filename)
            
            # Crop and rescale
            img_processed = crop_core_image(img, top, left, bottom, right, scale_factor)
            
            # Create new filename with depth range
            new_filename = f"{top_depth}-{bottom_depth}.jpg"
            output_path = os.path.join(output_dir, new_filename)
            
            # Save processed image
            skio.imsave(output_path, img_processed)
            
            processed_files.append(new_filename)
            logger.info(f"Processed {new_filename}")
            
        except Exception as e:
            logger.error(f"Error processing {full_filename}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_files)} of {len(input_files)} images")
    return processed_files


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Core image processing module")
    logger.info("Example: Process directory of core photos")
    
    # Example parameters for Pharos-1 white light images
    # These values would need to be adjusted for different image templates
    example_params = {
        'top': 642,
        'left': 630,
        'bottom': 12520,
        'right': 1570,
        'scale_factor': 0.3
    }
    
    logger.info(f"Example crop parameters: {example_params}")
    logger.info("Adjust parameters based on your core image template")


