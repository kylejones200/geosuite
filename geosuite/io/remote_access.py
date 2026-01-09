"""
Remote data access for PPDM and WITSML.

Provides API-based access to remote data sources including:
- PPDM database access via API
- WITSML subscription streaming
"""
import logging
from typing import Dict, List, Optional, Any, Union, Iterator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

# Try to import requests for API access
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning(
        "requests not available. Remote access requires requests. "
        "Install with: pip install requests"
    )


class PPDMApiClient:
    """
    Client for accessing PPDM data via API.
    
    Provides methods to query PPDM databases remotely.
    
    Example:
        >>> client = PPDMApiClient(api_url='https://api.example.com/ppdm')
        >>> wells = client.get_wells(limit=100)
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize PPDM API client.
        
        Parameters
        ----------
        api_url : str
            Base URL for PPDM API
        api_key : str, optional
            API key for authentication
        timeout : int, default 30
            Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for remote PPDM access. "
                "Install with: pip install requests"
            )
        
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def _request(
        self,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request.
        
        Parameters
        ----------
        endpoint : str
            API endpoint
        method : str, default 'GET'
            HTTP method
        params : dict, optional
            Query parameters
        json_data : dict, optional
            JSON body for POST requests
            
        Returns
        -------
        dict
            API response
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method == 'POST':
                response = self.session.post(url, json=json_data, params=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_wells(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Get wells from PPDM database.
        
        Parameters
        ----------
        limit : int, default 100
            Maximum number of wells to return
        offset : int, default 0
            Offset for pagination
        filters : dict, optional
            Filter criteria (e.g., {'field': 'Permian'})
            
        Returns
        -------
        pd.DataFrame
            Well data
        """
        params = {'limit': limit, 'offset': offset}
        if filters:
            params.update(filters)
        
        response = self._request('wells', params=params)
        
        if 'data' in response:
            return pd.DataFrame(response['data'])
        elif isinstance(response, list):
            return pd.DataFrame(response)
        else:
            raise ValueError("Unexpected API response format")
    
    def get_production(
        self,
        well_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get production data from PPDM database.
        
        Parameters
        ----------
        well_id : str, optional
            Specific well ID
        start_date : datetime, optional
            Start date for production data
        end_date : datetime, optional
            End date for production data
        limit : int, default 1000
            Maximum number of records
            
        Returns
        -------
        pd.DataFrame
            Production data
        """
        params = {'limit': limit}
        if well_id:
            params['well_id'] = well_id
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        
        response = self._request('production', params=params)
        
        if 'data' in response:
            return pd.DataFrame(response['data'])
        elif isinstance(response, list):
            return pd.DataFrame(response)
        else:
            raise ValueError("Unexpected API response format")


class WitsmlStreamClient:
    """
    Client for WITSML subscription streaming.
    
    Provides real-time access to WITSML data streams.
    
    Example:
        >>> client = WitsmlStreamClient(stream_url='wss://stream.example.com/witsml')
        >>> for data in client.stream_logs(well_uid='well-123'):
        ...     process(data)
    """
    
    def __init__(
        self,
        stream_url: str,
        api_key: Optional[str] = None,
        reconnect_interval: int = 5
    ):
        """
        Initialize WITSML stream client.
        
        Parameters
        ----------
        stream_url : str
            WebSocket URL for WITSML stream
        api_key : str, optional
            API key for authentication
        reconnect_interval : int, default 5
            Reconnection interval in seconds
        """
        try:
            import websocket
            WEBSOCKET_AVAILABLE = True
        except ImportError:
            WEBSOCKET_AVAILABLE = False
            logger.warning(
                "websocket-client not available. WITSML streaming requires websocket-client. "
                "Install with: pip install websocket-client"
            )
        
        if not WEBSOCKET_AVAILABLE:
            raise ImportError(
                "websocket-client is required for WITSML streaming. "
                "Install with: pip install websocket-client"
            )
        
        self.stream_url = stream_url
        self.api_key = api_key
        self.reconnect_interval = reconnect_interval
        self.ws = None
    
    def stream_logs(
        self,
        well_uid: str,
        log_uid: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream log data from WITSML subscription.
        
        Parameters
        ----------
        well_uid : str
            Well unique identifier
        log_uid : str, optional
            Specific log unique identifier
            
        Yields
        ------
        dict
            Log data chunks
        """
        import websocket
        import json
        
        url = f"{self.stream_url}?well_uid={well_uid}"
        if log_uid:
            url += f"&log_uid={log_uid}"
        
        if self.api_key:
            url += f"&api_key={self.api_key}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                yield data
            except Exception as e:
                logger.error(f"Error parsing stream message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
        
        while True:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                ws.run_forever()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(self.reconnect_interval)
    
    def subscribe(
        self,
        well_uid: str,
        log_uid: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> None:
        """
        Subscribe to WITSML data stream.
        
        Parameters
        ----------
        well_uid : str
            Well unique identifier
        log_uid : str, optional
            Specific log unique identifier
        callback : callable, optional
            Callback function for received data
        """
        for data in self.stream_logs(well_uid, log_uid):
            if callback:
                callback(data)
            else:
                logger.info(f"Received data: {data}")


def create_ppdm_client(
    api_url: str,
    api_key: Optional[str] = None
) -> PPDMApiClient:
    """
    Create PPDM API client.
    
    Parameters
    ----------
    api_url : str
        Base URL for PPDM API
    api_key : str, optional
        API key for authentication
        
    Returns
    -------
    PPDMApiClient
        Configured API client
    """
    return PPDMApiClient(api_url, api_key)


def create_witsml_stream_client(
    stream_url: str,
    api_key: Optional[str] = None
) -> WitsmlStreamClient:
    """
    Create WITSML stream client.
    
    Parameters
    ----------
    stream_url : str
        WebSocket URL for WITSML stream
    api_key : str, optional
        API key for authentication
        
    Returns
    -------
    WitsmlStreamClient
        Configured stream client
    """
    return WitsmlStreamClient(stream_url, api_key)


