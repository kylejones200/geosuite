# Input/output utilities for well data
from .witsml_parser import WitsmlParser, WitsmlDataConverter, export_to_witsml
from .ppdm_parser import PpdmParser, PpdmDataManager, PpdmDataModel, create_ppdm_sample_data

# Conditional imports for format-specific loaders
try:
    from .las_loader import load_las_file, get_las_metadata, read_las_summary
    _has_las = True
except ImportError:
    _has_las = False
    load_las_file = None
    get_las_metadata = None
    read_las_summary = None

try:
    from .segy_loader import (
        read_segy_summary,
        read_trace_headers,
        read_trace_headers_dataframe,
        read_trace,
        read_inline
    )
    _has_segy = True
except ImportError:
    _has_segy = False
    read_segy_summary = None
    read_trace_headers = None
    read_trace_headers_dataframe = None
    read_trace = None
    read_inline = None

try:
    from .resqml_parser import ResqmlParser, load_resqml_grid, load_resqml_properties
    _has_resqml = True
except ImportError:
    _has_resqml = False
    ResqmlParser = None
    load_resqml_grid = None
    load_resqml_properties = None

try:
    from .dlis_parser import DlisParser, load_dlis_file
    _has_dlis = True
except ImportError:
    _has_dlis = False
    DlisParser = None
    load_dlis_file = None

try:
    from .crs_utils import CRSHandler, standardize_crs, get_common_crs
    _has_crs = True
except ImportError:
    _has_crs = False
    CRSHandler = None
    standardize_crs = None
    get_common_crs = None

try:
    from .remote_access import (
        PPDMApiClient,
        WitsmlStreamClient,
        create_ppdm_client,
        create_witsml_stream_client
    )
    _has_remote = True
except ImportError:
    _has_remote = False
    PPDMApiClient = None
    WitsmlStreamClient = None
    create_ppdm_client = None
    create_witsml_stream_client = None

__all__ = [
    "WitsmlParser", "WitsmlDataConverter", "export_to_witsml",
    "PpdmParser", "PpdmDataManager", "PpdmDataModel", "create_ppdm_sample_data"
]

if _has_las:
    __all__.extend(["load_las_file", "get_las_metadata", "read_las_summary"])

if _has_segy:
    __all__.extend([
        "read_segy_summary",
        "read_trace_headers",
        "read_trace_headers_dataframe",
        "read_trace",
        "read_inline"
    ])

if _has_resqml:
    __all__.extend(["ResqmlParser", "load_resqml_grid", "load_resqml_properties"])

if _has_dlis:
    __all__.extend(["DlisParser", "load_dlis_file"])

if _has_crs:
    __all__.extend(["CRSHandler", "standardize_crs", "get_common_crs"])

if _has_remote:
    __all__.extend([
        "PPDMApiClient",
        "WitsmlStreamClient",
        "create_ppdm_client",
        "create_witsml_stream_client"
    ])
