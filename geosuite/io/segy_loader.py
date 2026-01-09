"""
SEG-Y file loader with enhanced trace header parsing.

Supports reading seismic data and comprehensive trace header information
from SEG-Y files.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import segyio
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    logger.warning(
        "segyio not available. SEG-Y support requires segyio. "
        "Install with: pip install segyio"
    )


@dataclass
class SegySummary:
    """Summary information from SEG-Y file."""
    path: str
    n_traces: int
    n_samples: int
    sample_rate_us: float
    text_header: str
    has_inline_crossline: bool
    format: int  # Data format code
    sample_interval_us: float  # Sample interval in microseconds


@dataclass
class TraceHeader:
    """Trace header information."""
    trace_number: int
    inline: Optional[int]
    crossline: Optional[int]
    x: Optional[float]  # X coordinate
    y: Optional[float]  # Y coordinate
    elevation: Optional[float]
    source_depth: Optional[float]
    receiver_elevation: Optional[float]
    source_x: Optional[float]
    source_y: Optional[float]
    receiver_x: Optional[float]
    receiver_y: Optional[float]
    offset: Optional[float]
    cdp_x: Optional[float]
    cdp_y: Optional[float]
    cdp: Optional[int]
    raw_header: Dict[str, Any]  # Raw header dictionary


def read_segy_summary(path: str) -> SegySummary:
    """
    Read summary information from SEG-Y file.
    
    Parameters
    ----------
    path : str
        Path to SEG-Y file
        
    Returns
    -------
    SegySummary
        Summary information
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size
        dt = float(segyio.tools.dt(f))  # microseconds
        text = segyio.tools.wrap(f.text[0]) if hasattr(f, 'text') else ''
        has_ix = hasattr(f, 'attributes') and (
            segyio.TraceField.INLINE_3D in f.attributes or 
            segyio.TraceField.CROSSLINE_3D in f.attributes
        )
        
        # Get format from binary header
        format_code = f.bin[segyio.BinField.Format]
        sample_interval = f.bin[segyio.BinField.Interval]
        
        return SegySummary(
            path=path,
            n_traces=int(n_traces),
            n_samples=int(n_samples),
            sample_rate_us=dt,
            text_header=text,
            has_inline_crossline=bool(has_ix),
            format=int(format_code),
            sample_interval_us=float(sample_interval) if sample_interval else dt
        )


def read_trace_headers(
    path: str,
    trace_indices: Optional[List[int]] = None,
    max_traces: int = 1000
) -> List[TraceHeader]:
    """
    Read trace headers from SEG-Y file.
    
    Parameters
    ----------
    path : str
        Path to SEG-Y file
    trace_indices : list of int, optional
        Specific trace indices to read (reads all if not specified)
    max_traces : int, default 1000
        Maximum number of traces to read (for large files)
        
    Returns
    -------
    list of TraceHeader
        Trace header information
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    headers = []
    
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        if trace_indices is None:
            # Read headers for all traces (up to max_traces)
            trace_indices = list(range(min(f.tracecount, max_traces)))
        
        for idx in trace_indices:
            if idx >= f.tracecount:
                continue
            
            # Read trace header
            header = f.header[idx]
            
            # Extract common fields
            trace_header = TraceHeader(
                trace_number=header.get(segyio.TraceField.TRACE_SEQUENCE_LINE),
                inline=header.get(segyio.TraceField.INLINE_3D),
                crossline=header.get(segyio.TraceField.CROSSLINE_3D),
                x=header.get(segyio.TraceField.CDP_X) or header.get(segyio.TraceField.GroupX),
                y=header.get(segyio.TraceField.CDP_Y) or header.get(segyio.TraceField.GroupY),
                elevation=header.get(segyio.TraceField.ReceiverGroupElevation),
                source_depth=header.get(segyio.TraceField.SourceSurfaceElevation),
                receiver_elevation=header.get(segyio.TraceField.ReceiverGroupElevation),
                source_x=header.get(segyio.TraceField.SourceX),
                source_y=header.get(segyio.TraceField.SourceY),
                receiver_x=header.get(segyio.TraceField.GroupX),
                receiver_y=header.get(segyio.TraceField.GroupY),
                offset=header.get(segyio.TraceField.offset),
                cdp_x=header.get(segyio.TraceField.CDP_X),
                cdp_y=header.get(segyio.TraceField.CDP_Y),
                cdp=header.get(segyio.TraceField.CDP),
                raw_header={k: header.get(k) for k in header.keys()}
            )
            
            headers.append(trace_header)
    
    return headers


def read_trace_headers_dataframe(
    path: str,
    trace_indices: Optional[List[int]] = None,
    max_traces: int = 1000
) -> pd.DataFrame:
    """
    Read trace headers as DataFrame.
    
    Parameters
    ----------
    path : str
        Path to SEG-Y file
    trace_indices : list of int, optional
        Specific trace indices to read
    max_traces : int, default 1000
        Maximum number of traces to read
        
    Returns
    -------
    pd.DataFrame
        Trace headers as DataFrame
    """
    headers = read_trace_headers(path, trace_indices, max_traces)
    
    # Convert to DataFrame
    data = []
    for h in headers:
        data.append({
            'trace_number': h.trace_number,
            'inline': h.inline,
            'crossline': h.crossline,
            'x': h.x,
            'y': h.y,
            'elevation': h.elevation,
            'source_depth': h.source_depth,
            'receiver_elevation': h.receiver_elevation,
            'source_x': h.source_x,
            'source_y': h.source_y,
            'receiver_x': h.receiver_x,
            'receiver_y': h.receiver_y,
            'offset': h.offset,
            'cdp_x': h.cdp_x,
            'cdp_y': h.cdp_y,
            'cdp': h.cdp
        })
    
    return pd.DataFrame(data)


def read_trace(path: str, trace_index: int = 0) -> np.ndarray:
    """Read a single trace from SEG-Y file."""
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        idx = max(0, min(trace_index, f.tracecount - 1))
        return np.asarray(f.trace[idx], dtype=float)


def read_inline(path: str, iline: Optional[int] = None) -> Optional[np.ndarray]:
    """Return a 2D array [x, z] for a given inline if available (None otherwise)."""
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    try:
        with segyio.open(path, mode='r', strict=False) as f:
            if not hasattr(f, 'iline'):
                return None
            if iline is None:
                keys = list(f.iline.keys())
                if not keys:
                    return None
                iline = keys[0]
            data = f.iline[iline]
            return np.asarray(data, dtype=float)
    except Exception:
        return None


def read_trace_with_phase_correction(
    path: str,
    inline: int,
    crossline: int,
    apply_correction: bool = True,
    threshold_pct: float = 75.0
) -> tuple[np.ndarray, float, dict]:
    """
    Read a trace from SEGY file and optionally apply phase correction.
    
    This is a convenience function that combines trace loading with phase
    correction from the seismic_processing module.
    
    Parameters
    ----------
    path : str
        Path to SEGY file
    inline : int
        Inline number
    crossline : int
        Crossline number
    apply_correction : bool, default True
        If True, apply phase correction using Hilbert transform
    threshold_pct : float, default 75.0
        Percentile threshold for peak detection (0-100)
        
    Returns
    -------
    tuple
        (trace, dt, info_dict):
        - trace: Seismic trace array (corrected if apply_correction=True)
        - dt: Sample interval (seconds)
        - info_dict: Dictionary with correction info (envelope, phase, etc.)
        
    Example
    -------
    >>> trace, dt, info = read_trace_with_phase_correction(
    ...     'data.sgy', iline=1190, xline=1155, apply_correction=True
    ... )
    >>> print(f"Applied phase shift: {info.get('phase_shift', 0):.2f} radians")
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    # Load trace using the seismic_processing function
    try:
        from ..petro.seismic_processing import correct_trace_phase
        
        # Load trace using basic function first
        trace, dt = _read_trace_basic(path, inline, crossline)
        
        if apply_correction:
            corrected, phase_shift, info = correct_trace_phase(
                trace, dt, threshold_pct=threshold_pct
            )
            info['phase_shift'] = phase_shift
            return corrected, dt, info
        else:
            return trace, dt, {}
            
    except ImportError:
        logger.warning(
            "Phase correction requires scipy. Using basic trace loading. "
            "Install with: pip install scipy"
        )
        # Fallback to basic trace loading
        trace, dt = _read_trace_basic(path, inline, crossline)
        return trace, dt, {}


def _read_trace_basic(path: str, inline: int, crossline: int) -> tuple[np.ndarray, float]:
    """Internal helper to read trace without phase correction."""
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEG-Y support")
    
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        f.mmap()
        dt_us = f.bin[segyio.BinField.Interval]
        dt = dt_us / 1e6  # Convert to seconds
        
        ilines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        
        for i, (il, xl) in enumerate(zip(ilines, xlines)):
            if il == inline and xl == crossline:
                trace = f.trace[i]
                return trace.astype(np.float32), dt
        
        raise ValueError(
            f"Trace not found for inline {inline}, crossline {crossline}"
        )
