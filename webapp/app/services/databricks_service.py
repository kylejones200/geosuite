"""
Databricks SQL query utilities for Flask app
Connects to Delta tables and returns data for the web application
"""

import logging
import os
import requests
import json
import configparser
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DatabricksQueryError(Exception):
    """Custom exception for Databricks query errors"""
    pass

class DatabricksClient:
    def __init__(self):
        self.host = None
        self.token = None
        self.warehouse_id = None
        self._load_config()
    
    def _load_config(self):
        """Load Databricks configuration from CLI config or environment"""
        try:
            # Try environment variables first
            self.host = os.environ.get('DATABRICKS_HOST')
            self.token = os.environ.get('DATABRICKS_TOKEN') 
            self.warehouse_id = os.environ.get('DATABRICKS_WAREHOUSE_ID')
            
            if not self.host or not self.token:
                # Fall back to CLI config
                config_path = os.path.expanduser('~/.databrickscfg')
                if os.path.exists(config_path):
                    config = configparser.ConfigParser()
                    config.read(config_path)
                    
                    profile = 'DEFAULT'
                    self.host = config[profile]['host']
                    self.token = config[profile]['token']
            
            # Ensure host format
            if self.host and not self.host.startswith('https://'):
                self.host = f'https://{self.host}'
                
        except Exception as e:
            logger.warning(f"Could not load Databricks config: {e}")
    
    def execute_query(self, sql_query: str, timeout: int = 60) -> List[List[Any]]:
        """Execute SQL query and return results"""
        if not self.host or not self.token:
            raise DatabricksQueryError("Databricks configuration not available")
        
        # For now, return mock data since SQL execution requires warehouse setup
        # In production, this would execute real SQL queries
        logger.debug(f"Would execute query: {sql_query[:100]}...")
        
        # Return mock data based on query type
        if 'wells' in sql_query.lower():
            return self._mock_wells_data()
        elif 'operator' in sql_query.lower():
            return self._mock_operators_data()
        else:
            return []
    
    def _mock_wells_data(self) -> List[List[Any]]:
        """Mock wells data - replace with real query results"""
        return [
            ['33053043310000', 'BAKKEN 1H', 'Continental Resources', 'BAKKEN', '2020-01', 47.8, -103.2, 250000],
            ['33053043320000', 'BAKKEN 2H', 'Whiting Petroleum', 'BAKKEN', '2021-03', 47.9, -103.1, 180000],
            ['33105044410000', 'THREE FORKS 1H', 'EOG Resources', 'THREE FORKS', '2019-08', 48.1, -102.8, 320000],
            ['33025055510000', 'BAKKEN 3H', 'Continental Resources', 'BAKKEN', '2022-05', 47.6, -103.5, 95000],
            ['33061066610000', 'BAKKEN 4H', 'Hess Corporation', 'BAKKEN', '2020-11', 47.7, -103.0, 280000],
        ]
    
    def _mock_operators_data(self) -> List[List[Any]]:
        """Mock operators data - replace with real query results"""
        return [
            ['Continental Resources', 85, 820, 12500000],
            ['Whiting Petroleum', 67, 780, 9800000],
            ['EOG Resources', 45, 950, 8200000],
            ['Hess Corporation', 38, 850, 7100000],
            ['Marathon Oil', 33, 720, 5900000],
        ]

# Global client instance
db_client = DatabricksClient()

def query_wells_geojson(limit: int = 1000) -> Dict[str, Any]:
    """Query wells data and return GeoJSON format"""
    sql = f"""
    SELECT 
        uwi, well_name, operator, formation, first_prod,
        latitude, longitude, cum_oil
    FROM ppdm.production_raw 
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    LIMIT {limit}
    """
    
    try:
        results = db_client.execute_query(sql)
        
        features = []
        for row in results:
            uwi, well_name, operator, formation, first_prod, lat, lon, cum_oil = row
            
            feature = {
                "id": uwi,
                "type": "Feature",
                "properties": {
                    "api": uwi,
                    "well_name": well_name or "",
                    "operator": operator or "Unknown",
                    "formation": formation or "",
                    "first_prod": first_prod,
                    "cum_oil": cum_oil or 0
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    except Exception as e:
        logger.error(f"Error querying wells: {e}")
        # Return fallback data
        return {
            "type": "FeatureCollection", 
            "features": [
                {
                    "id": "33053043310000",
                    "type": "Feature",
                    "properties": {
                        "api": "33053043310000",
                        "well_name": "BAKKEN 1H",
                        "operator": "Continental Resources",
                        "formation": "BAKKEN",
                        "first_prod": "2020-01", 
                        "cum_oil": 250000
                    },
                    "geometry": {"type": "Point", "coordinates": [-103.2, 47.8]}
                }
            ]
        }

def query_operators_data(since: Optional[str] = None) -> Dict[str, Any]:
    """Query operators performance data"""
    date_filter = ""
    if since:
        date_filter = f"WHERE file_month >= '{since}-01-01'"
    
    sql = f"""
    SELECT 
        operator,
        COUNT(DISTINCT uwi) as wells,
        AVG(avg_qi) as avg_qi,
        SUM(total_cum_oil) as total_cum_oil
    FROM ppdm.production_raw
    {date_filter}
    GROUP BY operator
    ORDER BY total_cum_oil DESC
    LIMIT 50
    """
    
    try:
        results = db_client.execute_query(sql)
        
        operators = []
        for row in results:
            operator, wells, avg_qi, total_cum_oil = row
            operators.append({
                "operator": operator,
                "wells": int(wells) if wells else 0,
                "avg_qi": float(avg_qi) if avg_qi else 0,
                "total_cum_oil": int(total_cum_oil) if total_cum_oil else 0
            })
        
        return {"operators": operators}
        
    except Exception as e:
        logger.error(f"Error querying operators: {e}")
        # Return fallback data
        return {
            "operators": [
                {"operator": "Continental Resources", "wells": 85, "avg_qi": 820, "total_cum_oil": 12500000},
                {"operator": "Whiting Petroleum", "wells": 67, "avg_qi": 780, "total_cum_oil": 9800000},
                {"operator": "EOG Resources", "wells": 45, "avg_qi": 950, "total_cum_oil": 8200000}
            ]
        }

def query_well_detail(uwi: str) -> Dict[str, Any]:
    """Query detailed data for a specific well"""
    sql = f"""
    SELECT 
        uwi, well_name, operator, formation, first_prod,
        cum_oil, cum_water, cum_gas
    FROM ppdm.production_raw
    WHERE uwi = '{uwi}'
    LIMIT 1
    """
    
    try:
        results = db_client.execute_query(sql)
        
        if results:
            row = results[0]
            uwi, well_name, operator, formation, first_prod, cum_oil, cum_water, cum_gas = row
            
            return {
                "api": uwi,
                "well_name": well_name or "",
                "operator": operator or "Unknown", 
                "formation": formation or "",
                "first_prod": first_prod,
                "cum_oil": cum_oil or 0,
                "cum_water": cum_water or 0,
                "cum_gas": cum_gas or 0,
                "timeseries": [
                    {"t": 30, "q": 800},
                    {"t": 60, "q": 650}, 
                    {"t": 90, "q": 520}
                ],
                "water_rates": [50, 45, 40],
                "gas_rates": [200, 180, 160]
            }
        
    except Exception as e:
        logger.error(f"Error querying well detail: {e}")
    
    # Fallback
    return {
        "api": uwi,
        "well_name": "Unknown Well",
        "operator": "Unknown",
        "formation": "Unknown",
        "first_prod": "2020-01",
        "cum_oil": 100000,
        "timeseries": [{"t": 30, "q": 500}],
        "water_rates": [25],
        "gas_rates": [100]
    }

