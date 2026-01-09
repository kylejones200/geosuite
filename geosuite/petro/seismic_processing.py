"""
Seismic processing utilities including phase correction and attribute analysis.

This module provides functions for seismic trace processing including:
- Phase correction using Hilbert transform
- Envelope and phase attribute computation
- Wavelet estimation for well ties
"""

from __future__ import annotations
from typing import Union, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import scipy.signal
try:
    from scipy.signal import hilbert, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy.signal not available. Phase correction requires scipy. "
        "Install with: pip install scipy"
    )

# Try to import segyio
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    logger.warning(
        "segyio not available. SEGY trace loading requires segyio. "
        "Install with: pip install segyio"
    )


def compute_hilbert_attributes(
    trace: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute envelope and unwrapped phase using the Hilbert transform.
    
    The Hilbert transform is used to compute the analytic signal from a
    real-valued seismic trace, from which the envelope (instantaneous amplitude)
    and phase (instantaneous phase) can be extracted.
    
    Args:
        trace: Seismic trace (real-valued array)
        dt: Sample interval in seconds
        
    Returns:
        Tuple of (time, envelope, phase) arrays:
            - time: Time array (seconds)
            - envelope: Envelope (instantaneous amplitude)
            - phase: Unwrapped phase (radians)
            
    Example:
        >>> trace = np.random.randn(1000)
        >>> dt = 0.004  # 4 ms
        >>> time, envelope, phase = compute_hilbert_attributes(trace, dt)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.signal is required for Hilbert transform")
    
    trace = np.asarray(trace, dtype=np.float64)
    
    if len(trace) == 0:
        raise ValueError("Trace cannot be empty")
    
    # Compute analytic signal using Hilbert transform
    analytic = hilbert(trace)
    
    # Envelope (instantaneous amplitude)
    envelope = np.abs(analytic)
    
    # Phase (instantaneous phase), unwrapped to remove discontinuities
    phase = np.unwrap(np.angle(analytic))
    
    # Time array
    time = np.arange(len(trace)) * dt
    
    return time, envelope, phase


def estimate_residual_phase(
    phase: np.ndarray,
    envelope: np.ndarray,
    time: Optional[np.ndarray] = None,
    threshold_pct: float = 75.0,
    distance: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate residual phase at envelope peaks.
    
    This function finds peaks in the envelope and estimates the residual phase
    by computing the circular mean of phase values at those peaks. This is used
    for phase correction of seismic data.
    
    Args:
        phase: Unwrapped phase array (radians)
        envelope: Envelope (instantaneous amplitude) array
        time: Time array (seconds). If None, uses sample indices.
        threshold_pct: Percentile threshold for peak detection (0-100)
        distance: Minimum distance between peaks (samples)
        
    Returns:
        Tuple of (residual_phase, peak_times, peak_phases):
            - residual_phase: Estimated residual phase (radians), circular mean
            - peak_times: Time values at envelope peaks
            - peak_phases: Phase values at envelope peaks
            
    Example:
        >>> time, envelope, phase = compute_hilbert_attributes(trace, dt)
        >>> residual, peak_t, peak_p = estimate_residual_phase(phase, envelope, time)
        >>> print(f"Residual phase: {residual:.2f} radians")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.signal is required for peak detection")
    
    phase = np.asarray(phase, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)
    
    if len(phase) != len(envelope):
        raise ValueError("phase and envelope must have the same length")
    
    if len(phase) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Find peaks in envelope
    threshold = np.percentile(envelope, threshold_pct)
    peaks, _ = find_peaks(
        envelope,
        distance=distance,
        height=threshold
    )
    
    if len(peaks) == 0:
        logger.warning("No peaks found in envelope, returning zero residual phase")
        return 0.0, np.array([]), np.array([])
    
    # Get phase values at peaks
    peak_phases = phase[peaks]
    
    # Circular mean of wrapped phase (handles phase wrapping correctly)
    # Convert to complex exponential, average, then get angle
    residual = np.angle(np.mean(np.exp(1j * peak_phases)))
    
    # Get peak times
    if time is not None:
        peak_times = time[peaks]
    else:
        peak_times = peaks.astype(float)
    
    return residual, peak_times, peak_phases


def apply_phase_shift(
    trace: np.ndarray,
    phase_shift: float
) -> np.ndarray:
    """
    Apply phase shift to seismic trace in frequency domain.
    
    Performs a constant phase shift using the Fourier transform. The phase shift
    is applied uniformly across all frequencies.
    
    Args:
        trace: Seismic trace (real-valued array)
        phase_shift: Phase shift to apply (radians)
        
    Returns:
        Phase-shifted trace (same length as input)
        
    Example:
        >>> corrected_trace = apply_phase_shift(trace, phase_shift=-0.5)
    """
    trace = np.asarray(trace, dtype=np.float64)
    
    if len(trace) == 0:
        raise ValueError("Trace cannot be empty")
    
    N = len(trace)
    
    # Compute FFT
    spectrum = np.fft.fft(trace)
    
    # Frequency array
    freqs = np.fft.fftfreq(N)
    
    # Phase shift in frequency domain
    # For each frequency f, multiply by exp(-1j * 2*pi * f * phase_shift)
    shift = np.exp(-1j * 2 * np.pi * freqs * phase_shift)
    shifted_spectrum = spectrum * shift
    
    # Inverse FFT to get corrected trace
    corrected = np.real(np.fft.ifft(shifted_spectrum))
    
    return corrected


def correct_trace_phase(
    trace: np.ndarray,
    dt: float,
    threshold_pct: float = 75.0,
    distance: int = 10
) -> Tuple[np.ndarray, float, dict]:
    """
    Correct residual phase in seismic trace.
    
    This is a convenience function that combines envelope/phase computation,
    residual phase estimation, and phase correction in one step.
    
    Args:
        trace: Seismic trace (real-valued array)
        dt: Sample interval (seconds)
        threshold_pct: Percentile threshold for peak detection (0-100)
        distance: Minimum distance between peaks (samples)
        
    Returns:
        Tuple of (corrected_trace, phase_shift, info_dict):
            - corrected_trace: Phase-corrected trace
            - phase_shift: Applied phase shift (radians)
            - info_dict: Dictionary with additional info:
                - 'envelope': Envelope of original trace
                - 'phase': Phase of original trace
                - 'peak_times': Times at envelope peaks
                - 'peak_phases': Phase values at peaks
                
    Example:
        >>> trace, dt = load_trace_from_segy('data.sgy', iline=100, xline=200)
        >>> corrected, shift, info = correct_trace_phase(trace, dt)
        >>> print(f"Applied phase shift: {shift:.2f} radians")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.signal is required for phase correction")
    
    # Compute envelope and phase
    time, envelope, phase = compute_hilbert_attributes(trace, dt)
    
    # Estimate residual phase
    phase_shift, peak_times, peak_phases = estimate_residual_phase(
        phase, envelope, time, threshold_pct=threshold_pct, distance=distance
    )
    
    # Apply phase correction
    corrected = apply_phase_shift(trace, phase_shift)
    
    # Return corrected trace, shift, and info
    info = {
        'envelope': envelope,
        'phase': phase,
        'peak_times': peak_times,
        'peak_phases': peak_phases,
        'time': time
    }
    
    return corrected, phase_shift, info


def load_trace_from_segy(
    segy_path: str,
    inline: int,
    crossline: int
) -> Tuple[np.ndarray, float]:
    """
    Load a single seismic trace from SEGY file by inline and crossline.
    
    Args:
        segy_path: Path to SEGY file
        inline: Inline number
        crossline: Crossline number
        
    Returns:
        Tuple of (trace, dt):
            - trace: Seismic trace array
            - dt: Sample interval (seconds)
            
    Raises:
        ValueError: If trace not found for specified inline/crossline
        ImportError: If segyio not available
        
    Example:
        >>> trace, dt = load_trace_from_segy('data.sgy', iline=1190, xline=1155)
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError("segyio is required for SEGY file loading")
    
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        f.mmap()
        
        # Get sample interval from binary header (in microseconds)
        dt_us = f.bin[segyio.BinField.Interval]
        dt = dt_us / 1e6  # Convert to seconds
        
        # Get inline and crossline attributes
        ilines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        
        # Find trace with matching inline and crossline
        for i, (il, xl) in enumerate(zip(ilines, xlines)):
            if il == inline and xl == crossline:
                trace = f.trace[i]
                return trace.astype(np.float32), dt
        
        raise ValueError(
            f"Trace not found for inline {inline}, crossline {crossline}"
        )

