"""
LAS (Log ASCII Standard) file loader with support for LAS 2.0 and LAS 3.0.

Supports reading well log data from LAS files in both legacy (2.0) and
modern (3.0) formats.
"""
import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import lasio
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False
    logger.warning(
        "lasio not available. LAS support requires lasio. "
        "Install with: pip install lasio"
    )


def load_las_file(
    las_path: Union[str, Path],
    version: Optional[str] = None
) -> pd.DataFrame:
    """
    Load LAS file (supports LAS 2.0 and LAS 3.0).
    
    Parameters
    ----------
    las_path : str or Path
        Path to LAS file
    version : str, optional
        LAS version ('2.0' or '3.0'). Auto-detected if not specified.
        
    Returns
    -------
    pd.DataFrame
        Well log data with depth/index as first column
        
    Example:
        >>> df = load_las_file('well_log.las')
        >>> print(df.columns)
    """
    if not LASIO_AVAILABLE:
        raise ImportError(
            "lasio is required for LAS support. "
            "Install with: pip install lasio"
        )
    
    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")
    
    # Read LAS file
    try:
        las = lasio.read(str(las_path))
    except Exception as e:
        raise ValueError(f"Error reading LAS file: {e}")
    
    # Detect version
    detected_version = _detect_las_version(las)
    if version and version != detected_version:
        logger.warning(
            f"Specified version {version} differs from detected {detected_version}"
        )
    
    # Convert to DataFrame
    df = las.df()
    
    # Handle LAS 3.0 specific features
    if detected_version == '3.0':
        df = _process_las3_features(las, df)
    
    # Ensure depth column is properly named
    if df.index.name is None or df.index.name == '':
        df.index.name = 'DEPTH'
    
    # Reset index to make depth a column
    df = df.reset_index()
    
    logger.info(
        f"Loaded LAS {detected_version} file: {len(df)} rows, "
        f"{len(df.columns)} columns"
    )
    
    return df


def _detect_las_version(las: Any) -> str:
    """
    Detect LAS file version.
    
    Parameters
    ----------
    las : lasio.LASFile
        LAS file object
        
    Returns
    -------
    str
        Version string ('2.0' or '3.0')
    """
    # Check version in header
    if hasattr(las, 'version'):
        version_str = str(las.version)
        if '3.0' in version_str:
            return '3.0'
    
    # Check for LAS 3.0 specific features
    # LAS 3.0 has more structured metadata
    if hasattr(las, 'well') and hasattr(las.well, 'metadata'):
        # LAS 3.0 has enhanced metadata structure
        return '3.0'
    
    # Default to 2.0 for legacy files
    return '2.0'


def _process_las3_features(las: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process LAS 3.0 specific features.
    
    Parameters
    ----------
    las : lasio.LASFile
        LAS file object
    df : pd.DataFrame
        DataFrame from LAS file
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with LAS 3.0 enhancements
    """
    # LAS 3.0 has better metadata handling
    # Add metadata as DataFrame attributes if available
    if hasattr(las, 'well'):
        well_metadata = {}
        for item in las.well:
            if hasattr(item, 'mnemonic') and hasattr(item, 'value'):
                well_metadata[item.mnemonic] = item.value
        
        # Store metadata as DataFrame attribute
        df.attrs['well_metadata'] = well_metadata
    
    # LAS 3.0 has better unit handling
    if hasattr(las, 'curves'):
        curve_units = {}
        for curve in las.curves:
            if hasattr(curve, 'unit') and hasattr(curve, 'mnemonic'):
                curve_units[curve.mnemonic] = curve.unit
        
        df.attrs['curve_units'] = curve_units
    
    return df


def get_las_metadata(las_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from LAS file.
    
    Parameters
    ----------
    las_path : str or Path
        Path to LAS file
        
    Returns
    -------
    dict
        Metadata including well information, version, etc.
    """
    if not LASIO_AVAILABLE:
        raise ImportError("lasio is required for LAS support")
    
    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")
    
    las = lasio.read(str(las_path))
    
    metadata = {
        'version': _detect_las_version(las),
        'well': {},
        'curves': []
    }
    
    # Extract well information
    if hasattr(las, 'well'):
        for item in las.well:
            if hasattr(item, 'mnemonic') and hasattr(item, 'value'):
                metadata['well'][item.mnemonic] = item.value
    
    # Extract curve information
    if hasattr(las, 'curves'):
        for curve in las.curves:
            curve_info = {
                'mnemonic': curve.mnemonic if hasattr(curve, 'mnemonic') else None,
                'unit': curve.unit if hasattr(curve, 'unit') else None,
                'description': curve.descr if hasattr(curve, 'descr') else None
            }
            metadata['curves'].append(curve_info)
    
    return metadata


def read_las_summary(dash_upload_contents: str) -> Dict[str, Any]:
    """
    Parse a Dash dcc.Upload contents string (data URL) assumed to be LAS and
    return a lightweight summary dict useful for preview.
    
    Supports both LAS 2.0 and LAS 3.0 formats.
    """
    import base64
    import io
    
    if not dash_upload_contents:
        raise ValueError("No contents provided")
    if not LASIO_AVAILABLE:
        raise ImportError("lasio is required to read LAS files")

    header, b64data = dash_upload_contents.split(',', 1)
    raw = base64.b64decode(b64data)
    buff = io.BytesIO(raw)
    las = lasio.read(buff)

    curves = [c.mnemonic for c in las.curves]
    start = float(getattr(las, 'start', float('nan')))
    stop = float(getattr(las, 'stop', float('nan')))
    step = float(getattr(las, 'step', float('nan')))

    version = _detect_las_version(las)

    return {
        "well": getattr(las.well.WELL, 'value', None) if hasattr(las, 'well') else None,
        "version": version,
        "curves_count": len(curves),
        "curves": curves[:25],  # limit for preview
        "start": start,
        "stop": stop,
        "step": step,
        "null": getattr(las, 'null', None),
    }
