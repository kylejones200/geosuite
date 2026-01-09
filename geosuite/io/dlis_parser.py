"""
DLIS (Digital Log Interchange Standard) format support.

DLIS is an industry standard for well log data exchange.
This module provides parsing utilities for DLIS files.
"""
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import DLIS libraries
try:
    import dlisio
    DLISIO_AVAILABLE = True
except ImportError:
    DLISIO_AVAILABLE = False
    logger.warning(
        "dlisio not available. DLIS support requires dlisio. "
        "Install with: pip install dlisio"
    )


class DlisParser:
    """
    Parser for DLIS well log files.
    
    Supports reading channels, frames, and well information
    from DLIS files.
    
    Example:
        >>> parser = DlisParser()
        >>> df = parser.load_channels('log.dlis')
        >>> well_info = parser.get_well_info('log.dlis')
    """
    
    def __init__(self):
        """Initialize DLIS parser."""
        if not DLISIO_AVAILABLE:
            raise ImportError(
                "dlisio is required for DLIS support. "
                "Install with: pip install dlisio"
            )
    
    def load_channels(
        self,
        dlis_path: Union[str, Path],
        channel_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load channel data from DLIS file.
        
        Parameters
        ----------
        dlis_path : str or Path
            Path to DLIS file
        channel_names : list of str, optional
            Specific channels to load (loads all if not specified)
            
        Returns
        -------
        pd.DataFrame
            Channel data with channels as columns
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")
        
        with dlisio.open(str(dlis_path)) as f:
            # Get all channels
            channels = f.channels()
            
            if not channels:
                raise ValueError("No channels found in DLIS file")
            
            # Filter by name if specified
            if channel_names:
                channels = {name: ch for name, ch in channels.items() 
                          if name in channel_names}
            
            if not channels:
                raise ValueError(
                    f"None of the specified channels found: {channel_names}"
                )
            
            # Get frame to determine length
            frames = f.frames()
            if not frames:
                raise ValueError("No frames found in DLIS file")
            
            frame = list(frames.values())[0]
            n_samples = len(frame)
            
            # Build DataFrame
            data = {}
            
            for name, channel in channels.items():
                try:
                    # Read channel data
                    values = channel.curves()
                    if len(values) == n_samples:
                        data[name] = values
                    else:
                        logger.warning(
                            f"Channel {name} has length {len(values)}, "
                            f"expected {n_samples}. Skipping."
                        )
                except Exception as e:
                    logger.warning(f"Error reading channel {name}: {e}")
            
            if not data:
                raise ValueError("No valid channel data could be read")
            
            df = pd.DataFrame(data)
            
            # Add index as depth if available
            if 'TDEP' in df.columns:
                df.index = df['TDEP']
                df.index.name = 'DEPTH'
            elif 'DEPT' in df.columns:
                df.index = df['DEPT']
                df.index.name = 'DEPTH'
            
            return df
    
    def get_well_info(self, dlis_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract well information from DLIS file.
        
        Parameters
        ----------
        dlis_path : str or Path
            Path to DLIS file
            
        Returns
        -------
        dict
            Well information including:
            - 'well_name': Well name
            - 'field': Field name
            - 'company': Company name
            - 'service_company': Service company
            - 'run_number': Run number
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")
        
        with dlisio.open(str(dlis_path)) as f:
            # Get origin (well information)
            origins = f.origins()
            if not origins:
                return {}
            
            origin = list(origins.values())[0]
            
            info = {}
            
            # Extract well information
            if hasattr(origin, 'well_name'):
                info['well_name'] = str(origin.well_name)
            if hasattr(origin, 'field_name'):
                info['field'] = str(origin.field_name)
            if hasattr(origin, 'producer_name'):
                info['company'] = str(origin.producer_name)
            if hasattr(origin, 'service_company'):
                info['service_company'] = str(origin.service_company)
            if hasattr(origin, 'run_number'):
                info['run_number'] = str(origin.run_number)
            
            return info
    
    def list_channels(self, dlis_path: Union[str, Path]) -> List[str]:
        """
        List all available channels in DLIS file.
        
        Parameters
        ----------
        dlis_path : str or Path
            Path to DLIS file
            
        Returns
        -------
        list of str
            Channel names
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")
        
        with dlisio.open(str(dlis_path)) as f:
            channels = f.channels()
            return list(channels.keys())


def load_dlis_file(
    dlis_path: Union[str, Path],
    channel_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to load DLIS file.
    
    Parameters
    ----------
    dlis_path : str or Path
        Path to DLIS file
    channel_names : list of str, optional
        Specific channels to load
        
    Returns
    -------
    pd.DataFrame
        Channel data
    """
    parser = DlisParser()
    return parser.load_channels(dlis_path, channel_names=channel_names)


