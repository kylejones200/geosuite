import base64
import io
from typing import Dict, Any

try:
    import lasio  # type: ignore
except Exception:  # keep import optional for environments without lasio
    lasio = None


def read_las_summary(dash_upload_contents: str) -> Dict[str, Any]:
    """
    Parse a Dash dcc.Upload contents string (data URL) assumed to be LAS and
    return a lightweight summary dict useful for preview.
    """
    if not dash_upload_contents:
        raise ValueError("No contents provided")
    if lasio is None:
        raise ImportError("lasio is required to read LAS files")

    header, b64data = dash_upload_contents.split(',', 1)
    raw = base64.b64decode(b64data)
    buff = io.BytesIO(raw)
    las = lasio.read(buff)

    curves = [c.mnemonic for c in las.curves]
    start = float(getattr(las, 'start', float('nan')))
    stop = float(getattr(las, 'stop', float('nan')))
    step = float(getattr(las, 'step', float('nan')))

    return {
        "well": getattr(las.well.WELL, 'value', None) if hasattr(las, 'well') else None,
        "curves_count": len(curves),
        "curves": curves[:25],  # limit for preview
        "start": start,
        "stop": stop,
        "step": step,
        "null": getattr(las, 'null', None),
    }
