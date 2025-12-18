from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SegySummary:
    path: str
    n_traces: int
    n_samples: int
    sample_rate_us: float
    text_header: str
    has_inline_crossline: bool


def read_segy_summary(path: str) -> SegySummary:
    import segyio
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size
        # sample interval in microseconds from binary header (dt in microseconds)
        dt = float(segyio.tools.dt(f))  # microseconds
        text = segyio.tools.wrap(f.text[0]) if hasattr(f, 'text') else ''
        has_ix = hasattr(f, 'attributes') and (segyio.TraceField.INLINE_3D in f.attributes or segyio.TraceField.CROSSLINE_3D in f.attributes)
        return SegySummary(
            path=path,
            n_traces=int(n_traces),
            n_samples=int(n_samples),
            sample_rate_us=dt,
            text_header=text,
            has_inline_crossline=bool(has_ix),
        )


def read_trace(path: str, trace_index: int = 0) -> np.ndarray:
    import segyio
    with segyio.open(path, mode='r', strict=False, ignore_geometry=True) as f:
        idx = max(0, min(trace_index, f.tracecount - 1))
        return np.asarray(f.trace[idx], dtype=float)


def read_inline(path: str, iline: Optional[int] = None) -> Optional[np.ndarray]:
    """Return a 2D array [x, z] for a given inline if available (None otherwise)."""
    import segyio
    try:
        with segyio.open(path, mode='r', strict=False) as f:
            if not hasattr(f, 'iline'):
                return None
            if iline is None:
                # choose first available inline
                keys = list(f.iline.keys())
                if not keys:
                    return None
                iline = keys[0]
            data = f.iline[iline]
            return np.asarray(data, dtype=float)
    except Exception:
        return None
