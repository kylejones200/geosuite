"""
PPDM (Professional Petroleum Data Management) Integration for GeoSuite
Industry-standard data model for petroleum data management and storage
"""

import pandas as pd
import numpy as np
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

class PpdmDataModel:
    """PPDM data model definitions and utilities"""
    
    # Core PPDM table structures
    CORE_TABLES = {
        'well': {
            'table_name': 'ppdm_well',
            'primary_key': 'uwi',
            'required_fields': ['uwi', 'well_name', 'operator', 'surface_longitude', 'surface_latitude'],
            'field_mappings': {
                'UWI': 'uwi',
                'ALT_WELL_NAME': 'well_name', 
                'OPERATOR': 'operator',
                'SURFACE_LATITUDE': 'surface_latitude',
                'SURFACE_LONGITUDE': 'surface_longitude',
                'ASSIGNED_FIELD': 'field_name',
                'CURRENT_STATUS': 'well_status',
                'CURRENT_CLASS': 'well_class',
                'COUNTRY': 'country',
                'PROVINCE_STATE': 'province_state',
                'PPDM_GUID': 'ppdm_guid'
            }
        },
        'business_associate': {
            'table_name': 'ppdm_business_associate',
            'primary_key': 'business_associate_id',
            'required_fields': ['business_associate_id', 'ba_name'],
            'field_mappings': {
                'BUSINESS_ASSOCIATE': 'business_associate_id',
                'BA_NAME': 'ba_name',
                'BA_TYPE': 'ba_type',
                'BA_CATEGORY': 'ba_category',
                'CURRENT_STATUS': 'current_status',
                'PPDM_GUID': 'ppdm_guid'
            }
        },
        'production': {
            'table_name': 'ppdm_production',
            'primary_key': ['uwi', 'production_date', 'product_type'],
            'required_fields': ['uwi', 'production_date', 'product_type'],
            'field_mappings': {
                'UWI': 'uwi',
                'PRODUCTION_DATE': 'production_date',
                'PRODUCT_TYPE': 'product_type',
                'DAILY_PROD_VOL': 'daily_volume',
                'MONTHLY_PROD_VOL': 'monthly_volume',
                'CUMULATIVE_PROD_VOL': 'cumulative_volume',
                'PROD_METHOD': 'production_method',
                'PPDM_GUID': 'ppdm_guid'
            }
        }
    }

    @classmethod
    def get_table_definition(cls, table_type: str) -> Dict[str, Any]:
        """Get PPDM table definition"""
        return cls.CORE_TABLES.get(table_type, {})
    
    @classmethod
    def validate_uwi(cls, uwi: str) -> bool:
        """Validate UWI format (basic validation)"""
        if not uwi or len(uwi) < 10:
            return False
        return uwi.replace('-', '').replace('.', '').isdigit()
    
    @classmethod
    def standardize_uwi(cls, uwi: str) -> str:
        """Standardize UWI format"""
        if not uwi:
            return ''
        # Remove common separators and pad if needed
        clean_uwi = uwi.replace('-', '').replace('.', '').replace(' ', '')
        return clean_uwi.zfill(14)  # Standard 14-digit UWI


class PpdmParser:
    """Parser for PPDM data files and database structures"""
    
    def __init__(self, ppdm_directory: str = None):
        self.ppdm_directory = Path(ppdm_directory) if ppdm_directory else None
        self.data_model = PpdmDataModel()
        
    def parse_sql_schema(self, sql_file: str) -> Dict[str, Any]:
        """Parse PPDM SQL schema file to extract table definitions"""
        try:
            with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract CREATE TABLE statements
            table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
            tables = {}
            
            matches = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for table_name, definition in matches:
                # Parse column definitions
                columns = []
                column_pattern = r'(\w+)\s+([A-Z0-9_(),\s]+)(?:,|\s*$)'
                
                for line in definition.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('--'):
                        col_match = re.match(r'(\w+)\s+([^,\n]+)', line)
                        if col_match:
                            col_name, col_type = col_match.groups()
                            col_type = col_type.strip().rstrip(',')
                            columns.append({
                                'name': col_name,
                                'type': col_type,
                                'nullable': 'NOT NULL' not in col_type
                            })
                
                tables[table_name.lower()] = {
                    'name': table_name,
                    'columns': columns
                }
            
            return tables
            
        except Exception as e:
            raise Exception(f"Error parsing SQL schema {sql_file}: {str(e)}")
    
    def load_ppdm_csv(self, csv_file: str, data_type: str = 'well') -> pd.DataFrame:
        """Load PPDM CSV data with proper formatting"""
        try:
            # Load CSV with error handling for large files
            if data_type == 'production':
                # For large production files, use chunking
                chunk_size = 10000
                chunks = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(csv_file, low_memory=False)
            
            # Get table definition
            table_def = self.data_model.get_table_definition(data_type)
            if not table_def:
                return df  # Return as-is if no definition
            
            # Apply field mappings
            field_mappings = table_def.get('field_mappings', {})
            for old_col, new_col in field_mappings.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Data type conversions
            if data_type == 'well':
                df = self._process_well_data(df)
            elif data_type == 'production':
                df = self._process_production_data(df)
            elif data_type == 'business_associate':
                df = self._process_business_associate_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading PPDM CSV {csv_file}: {str(e)}")
    
    def _process_well_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process well data with PPDM standards"""
        processed_df = df.copy()
        
        # Standardize UWI
        if 'uwi' in processed_df.columns:
            processed_df['uwi'] = processed_df['uwi'].apply(
                lambda x: self.data_model.standardize_uwi(str(x)) if pd.notna(x) else ''
            )
        
        # Convert coordinates to numeric
        coord_columns = ['surface_latitude', 'surface_longitude']
        for col in coord_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Clean well names
        if 'well_name' in processed_df.columns:
            processed_df['well_name'] = processed_df['well_name'].astype(str).str.strip()
        
        # Add derived fields
        if 'surface_latitude' in processed_df.columns and 'surface_longitude' in processed_df.columns:
            # Add coordinate quality flag
            processed_df['coord_quality'] = np.where(
                (processed_df['surface_latitude'].notna()) & 
                (processed_df['surface_longitude'].notna()) &
                (processed_df['surface_latitude'] != 0) &
                (processed_df['surface_longitude'] != 0),
                'GOOD', 'POOR'
            )
        
        return processed_df
    
    def _process_production_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process production data with PPDM standards"""
        processed_df = df.copy()
        
        # Convert production date
        if 'production_date' in processed_df.columns:
            processed_df['production_date'] = pd.to_datetime(
                processed_df['production_date'], errors='coerce'
            )
        
        # Convert volume columns to numeric
        volume_columns = ['daily_volume', 'monthly_volume', 'cumulative_volume']
        for col in volume_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Standardize product types
        if 'product_type' in processed_df.columns:
            product_mapping = {
                'OIL': 'OIL',
                'GAS': 'GAS', 
                'WATER': 'WATER',
                'COND': 'CONDENSATE',
                'CONDENSATE': 'CONDENSATE'
            }
            processed_df['product_type'] = processed_df['product_type'].map(
                product_mapping
            ).fillna(processed_df['product_type'])
        
        return processed_df
    
    def _process_business_associate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process business associate data"""
        processed_df = df.copy()
        
        # Clean business associate names
        if 'ba_name' in processed_df.columns:
            processed_df['ba_name'] = processed_df['ba_name'].astype(str).str.strip()
        
        # Standardize BA types
        if 'ba_type' in processed_df.columns:
            ba_type_mapping = {
                'OPERATOR': 'OPERATOR',
                'SERVICE_COMPANY': 'SERVICE_COMPANY',
                'GOVERNMENT': 'GOVERNMENT',
                'INDIVIDUAL': 'INDIVIDUAL'
            }
            processed_df['ba_type'] = processed_df['ba_type'].map(
                ba_type_mapping
            ).fillna(processed_df['ba_type'])
        
        return processed_df


class PpdmDataManager:
    """Manage PPDM data integration with GeoSuite"""
    
    def __init__(self, database_path: str = None):
        self.database_path = database_path or ':memory:'
        self.parser = PpdmParser()
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for PPDM data"""
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Create core tables
        self._create_core_tables()
    
    def _create_core_tables(self):
        """Create core PPDM tables in SQLite"""
        cursor = self.connection.cursor()
        
        # Wells table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppdm_well (
                uwi TEXT PRIMARY KEY,
                well_name TEXT,
                operator TEXT,
                surface_latitude REAL,
                surface_longitude REAL,
                field_name TEXT,
                well_status TEXT,
                well_class TEXT,
                country TEXT,
                province_state TEXT,
                coord_quality TEXT,
                ppdm_guid TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT
            )
        ''')
        
        # Business associates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppdm_business_associate (
                business_associate_id TEXT PRIMARY KEY,
                ba_name TEXT,
                ba_type TEXT,
                ba_category TEXT,
                current_status TEXT,
                ppdm_guid TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Production table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppdm_production (
                uwi TEXT,
                production_date DATE,
                product_type TEXT,
                daily_volume REAL,
                monthly_volume REAL,
                cumulative_volume REAL,
                production_method TEXT,
                ppdm_guid TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (uwi, production_date, product_type)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_well_operator ON ppdm_well(operator)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_well_field ON ppdm_well(field_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prod_uwi ON ppdm_production(uwi)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prod_date ON ppdm_production(production_date)')
        
        self.connection.commit()
    
    def load_data_from_csv(self, csv_file: str, data_type: str, source: str = None) -> int:
        """Load PPDM data from CSV into database"""
        try:
            # Parse CSV data
            df = self.parser.load_ppdm_csv(csv_file, data_type)
            
            if df.empty:
                return 0
            
            # Add metadata
            df['data_source'] = source or Path(csv_file).name
            df['created_date'] = datetime.now().isoformat()
            
            # Get table definition
            table_def = self.parser.data_model.get_table_definition(data_type)
            table_name = table_def.get('table_name', f'ppdm_{data_type}')
            
            # Insert data
            df.to_sql(table_name, self.connection, if_exists='append', index=False)
            
            return len(df)
            
        except Exception as e:
            raise Exception(f"Error loading PPDM data from {csv_file}: {str(e)}")
    
    def get_wells_for_geosuite(self, field_name: str = None, operator: str = None) -> pd.DataFrame:
        """Get well data formatted for GeoSuite analysis"""
        query = '''
            SELECT 
                uwi,
                well_name,
                operator,
                surface_latitude as y_coord,
                surface_longitude as x_coord,
                field_name as formation,
                well_status as drilling_status,
                'PPDM' as data_source
            FROM ppdm_well 
            WHERE coord_quality = 'GOOD'
        '''
        
        params = []
        if field_name:
            query += ' AND field_name = ?'
            params.append(field_name)
        if operator:
            query += ' AND operator = ?'
            params.append(operator)
        
        query += ' ORDER BY well_name'
        
        df = pd.read_sql_query(query, self.connection, params=params)
        
        # Add required GeoSuite columns with defaults
        if not df.empty:
            df['depth_m'] = 1500.0  # Default depth
            df['GR'] = np.random.uniform(40, 120, len(df))  # Simulated log data
            df['RHOB'] = np.random.uniform(2.2, 2.7, len(df))
            df['NPHI'] = np.random.uniform(0.1, 0.3, len(df))
            df['RT'] = np.random.uniform(5, 50, len(df))
            df['mud_weight_used'] = np.random.uniform(1.1, 1.4, len(df))
            df['cost_per_meter'] = np.random.uniform(800, 1200, len(df))
            df['days_to_drill'] = np.random.uniform(10, 25, len(df))
        
        return df
    
    def get_production_summary(self, uwi_list: List[str] = None, 
                             start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get production summary data"""
        query = '''
            SELECT 
                p.uwi,
                w.well_name,
                w.operator,
                p.product_type,
                SUM(p.monthly_volume) as total_volume,
                COUNT(p.production_date) as months_produced,
                MIN(p.production_date) as first_production,
                MAX(p.production_date) as last_production,
                AVG(p.daily_volume) as avg_daily_rate
            FROM ppdm_production p
            JOIN ppdm_well w ON p.uwi = w.uwi
            WHERE 1=1
        '''
        
        params = []
        if uwi_list:
            placeholders = ','.join(['?'] * len(uwi_list))
            query += f' AND p.uwi IN ({placeholders})'
            params.extend(uwi_list)
        
        if start_date:
            query += ' AND p.production_date >= ?'
            params.append(start_date)
            
        if end_date:
            query += ' AND p.production_date <= ?'
            params.append(end_date)
        
        query += '''
            GROUP BY p.uwi, w.well_name, w.operator, p.product_type
            ORDER BY total_volume DESC
        '''
        
        return pd.read_sql_query(query, self.connection, params=params)
    
    def get_field_statistics(self, field_name: str = None) -> Dict[str, Any]:
        """Get field-level statistics"""
        cursor = self.connection.cursor()
        
        # Well counts by status
        well_stats_query = '''
            SELECT 
                well_status,
                COUNT(*) as count,
                COUNT(DISTINCT operator) as operators
            FROM ppdm_well
        '''
        
        params = []
        if field_name:
            well_stats_query += ' WHERE field_name = ?'
            params.append(field_name)
        
        well_stats_query += ' GROUP BY well_status'
        
        well_stats = cursor.execute(well_stats_query, params).fetchall()
        
        # Production statistics (if available)
        prod_stats_query = '''
            SELECT 
                product_type,
                SUM(monthly_volume) as total_volume,
                COUNT(DISTINCT uwi) as producing_wells
            FROM ppdm_production p
        '''
        
        if field_name:
            prod_stats_query += '''
                JOIN ppdm_well w ON p.uwi = w.uwi 
                WHERE w.field_name = ?
            '''
        
        prod_stats_query += ' GROUP BY product_type'
        
        prod_stats = cursor.execute(prod_stats_query, params).fetchall()
        
        return {
            'well_statistics': [dict(row) for row in well_stats],
            'production_statistics': [dict(row) for row in prod_stats],
            'field_name': field_name
        }
    
    def export_to_geosuite_format(self, output_file: str, data_type: str = 'field_data'):
        """Export PPDM data in GeoSuite-compatible format"""
        if data_type == 'field_data':
            df = self.get_wells_for_geosuite()
        else:
            raise ValueError(f"Unknown export type: {data_type}")
        
        df.to_csv(output_file, index=False)
        return len(df)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


def create_ppdm_sample_data():
    """Create sample PPDM data for demonstration"""
    
    # Sample wells data
    wells_data = {
        'uwi': ['33053043310000', '33053038990000', '33053048330000'],
        'well_name': ['DEMO WELL A', 'DEMO WELL B', 'DEMO WELL C'],
        'operator': ['BA000127', 'BA000201', 'BA000127'],
        'surface_latitude': [47.8491, 47.9074, 47.9046],
        'surface_longitude': [-103.5533, -103.5799, -103.5692],
        'field_name': ['DEMO_FIELD', 'DEMO_FIELD', 'DEMO_FIELD'],
        'well_status': ['ACTIVE', 'ACTIVE', 'ACTIVE'],
        'well_class': ['OIL_GAS', 'OIL_GAS', 'OIL_GAS'],
        'country': ['US', 'US', 'US'],
        'province_state': ['ND', 'ND', 'ND'],
        'coord_quality': ['GOOD', 'GOOD', 'GOOD'],
        'ppdm_guid': ['75CB2E37-3319-41BE-B8DB-44B3E0E68CC4', 
                      'D4C2CE72-F3A0-459C-B5D9-C01A4DA19617',
                      '73EE3B91-CFDA-4970-A675-801A75F5C6C0']
    }
    
    # Sample business associates
    ba_data = {
        'business_associate_id': ['BA000127', 'BA000201'],
        'ba_name': ['DEMO OPERATOR A', 'DEMO OPERATOR B'],
        'ba_type': ['OPERATOR', 'OPERATOR'],
        'ba_category': ['COMPANY', 'COMPANY'],
        'current_status': ['ACTIVE', 'ACTIVE']
    }
    
    return pd.DataFrame(wells_data), pd.DataFrame(ba_data)
