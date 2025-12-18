import base64
import io
import pandas as pd
from typing import Optional


def load_csv(dash_upload_contents: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Parse a Dash dcc.Upload contents string (data URL) assumed to be CSV and return a DataFrame.
    """
    if not dash_upload_contents:
        raise ValueError("No contents provided")
    try:
        header, b64data = dash_upload_contents.split(',', 1)
    except ValueError:
        raise ValueError("Invalid contents format; expected data URL with base64 payload")

    raw = base64.b64decode(b64data)
    buffer = io.BytesIO(raw)
    if encoding:
        return pd.read_csv(buffer, encoding=encoding)
    # pandas will infer encoding
    return pd.read_csv(buffer)
