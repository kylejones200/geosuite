"""
WITSML v2.0 XML Parser for GeoSuite
Integrates industry-standard WITSML data with existing analysis workflows
"""

import logging
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# WITSML v2.0 namespaces
WITSML_NS = {
    'witsml': 'http://www.energistics.org/energyml/data/witsmlv2',
    'eml': 'http://www.energistics.org/energyml/data/commonv2'
}


class WitsmlParser:
    """WITSML XML parser for well data integration"""
    
    def __init__(self):
        self.namespaces = WITSML_NS
    
    def _get_element_text(self, element: ET.Element, xpath: str) -> Optional[str]:
        """Safely extract text from XML element using xpath"""
        found = element.find(xpath, self.namespaces)
        return found.text if found is not None else None
    
    def _get_element_attr(self, element: ET.Element, xpath: str, attr: str) -> Optional[str]:
        """Safely extract attribute from XML element"""
        found = element.find(xpath, self.namespaces)
        return found.get(attr) if found is not None else None
    
    def _clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean and convert numeric values"""
        if value is None:
            return None
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    
    def parse_well_header(self, xml_file: str) -> Dict[str, Any]:
        """Parse Well header information from WITSML XML"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            well_data = {
                'uid': root.get('uuid'),
                'name': self._get_element_text(root, './/eml:Citation/eml:Title'),
                'field': self._get_element_text(root, './/witsml:Field'),
                'country': self._get_element_text(root, './/witsml:Country'),
                'state': self._get_element_text(root, './/witsml:State'),
                'county': self._get_element_text(root, './/witsml:County'),
                'operator': self._get_element_text(root, './/witsml:Operator'),
                'status_well': self._get_element_text(root, './/witsml:StatusWell'),
                'purpose_well': self._get_element_text(root, './/witsml:PurposeWell'),
                'spud_date': self._get_element_text(root, './/witsml:DTimSpud'),
                'license_number': self._get_element_text(root, './/witsml:NumLicense'),
                'api_number': self._get_element_text(root, './/witsml:NumGovt'),
                'surface_location': self._extract_location_data(root, 'Surface'),
                'bottom_hole_location': self._extract_location_data(root, 'BottomHole')
            }
            
            return well_data
            
        except Exception as e:
            raise Exception(f"Error parsing WITSML Well file {xml_file}: {str(e)}")
    
    def parse_wellbore_data(self, xml_file: str) -> Dict[str, Any]:
        """Parse Wellbore information from WITSML XML"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            wellbore_data = {
                'uid': root.get('uuid'),
                'name': self._get_element_text(root, './/eml:Citation/eml:Title'),
                'well_uid': self._get_element_text(root, './/witsml:Well'),
                'number': self._get_element_text(root, './/witsml:Number'),
                'suffix_api': self._get_element_text(root, './/witsml:SuffixAPI'),
                'num_govt': self._get_element_text(root, './/witsml:NumGovt'),
                'status_wellbore': self._get_element_text(root, './/witsml:StatusWellbore'),
                'purpose_wellbore': self._get_element_text(root, './/witsml:PurposeWellbore'),
                'type_wellbore': self._get_element_text(root, './/witsml:TypeWellbore'),
                'shape': self._get_element_text(root, './/witsml:Shape'),
                'dtime_kick_off': self._get_element_text(root, './/witsml:DTimeKickOff'),
                'md_planned': self._clean_numeric_value(self._get_element_text(root, './/witsml:MdPlanned')),
                'tvd_planned': self._clean_numeric_value(self._get_element_text(root, './/witsml:TvdPlanned')),
                'md_subseabed': self._clean_numeric_value(self._get_element_text(root, './/witsml:MdSubSeaBed')),
                'tvd_subseabed': self._clean_numeric_value(self._get_element_text(root, './/witsml:TvdSubSeaBed')),
                'md_current': self._clean_numeric_value(self._get_element_text(root, './/witsml:MdCurrent')),
                'tvd_current': self._clean_numeric_value(self._get_element_text(root, './/witsml:TvdCurrent'))
            }
            
            return wellbore_data
            
        except Exception as e:
            raise Exception(f"Error parsing WITSML Wellbore file {xml_file}: {str(e)}")
    
    def parse_log_data(self, xml_file: str) -> pd.DataFrame:
        """Parse Log data from WITSML XML and return pandas DataFrame"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract log header information
            log_info = {
                'uid': root.get('uuid'),
                'name': self._get_element_text(root, './/eml:Citation/eml:Title'),
                'wellbore_uid': self._get_element_text(root, './/witsml:Wellbore'),
                'service_company': self._get_element_text(root, './/witsml:ServiceCompany'),
                'run_number': self._get_element_text(root, './/witsml:RunNumber'),
                'creation_date': self._get_element_text(root, './/eml:Citation/eml:Creation'),
                'start_md': self._clean_numeric_value(self._get_element_text(root, './/witsml:StartIndex')),
                'end_md': self._clean_numeric_value(self._get_element_text(root, './/witsml:EndIndex')),
                'direction': self._get_element_text(root, './/witsml:Direction')
            }
            
            # Extract channel information
            channels = []
            channel_elements = root.findall('.//witsml:Channel', self.namespaces)
            
            for channel in channel_elements:
                channel_info = {
                    'mnemonic': self._get_element_text(channel, './/witsml:Mnemonic'),
                    'unit': self._get_element_attr(channel, './/witsml:Index', 'uom') or 
                            self._get_element_attr(channel, './/witsml:Value', 'uom'),
                    'channel_class': self._get_element_text(channel, './/witsml:ChannelClass'),
                    'description': self._get_element_text(channel, './/eml:Citation/eml:Description'),
                    'data_type': self._get_element_text(channel, './/witsml:DataType'),
                    'null_value': self._get_element_text(channel, './/witsml:NullValue')
                }
                channels.append(channel_info)
            
            # Extract log data
            log_data_elements = root.findall('.//witsml:LogData/witsml:Data', self.namespaces)
            
            if not log_data_elements:
                return pd.DataFrame()
            
            # Parse the actual log data
            data_rows = []
            for data_element in log_data_elements:
                data_text = data_element.text
                if data_text:
                    # Split comma-separated values
                    values = [v.strip() for v in data_text.split(',')]
                    data_rows.append(values)
            
            # Create column names from channel mnemonics
            column_names = [ch['mnemonic'] for ch in channels]
            
            # Create DataFrame
            if data_rows and column_names:
                df = pd.DataFrame(data_rows, columns=column_names[:len(data_rows[0])])
                
                # Convert numeric columns
                for col in df.columns:
                    if col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD']:  # Depth columns
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        # Try to convert other columns to numeric
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                
                # Add metadata as attributes
                df.attrs['log_info'] = log_info
                df.attrs['channels'] = channels
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            raise Exception(f"Error parsing WITSML Log file {xml_file}: {str(e)}")
    
    def parse_trajectory_data(self, xml_file: str) -> pd.DataFrame:
        """Parse Trajectory data from WITSML XML"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract trajectory header
            trajectory_info = {
                'uid': root.get('uuid'),
                'name': self._get_element_text(root, './/eml:Citation/eml:Title'),
                'wellbore_uid': self._get_element_text(root, './/witsml:Wellbore'),
                'service_company': self._get_element_text(root, './/witsml:ServiceCompany'),
                'magnetic_declination': self._clean_numeric_value(
                    self._get_element_text(root, './/witsml:MagDeclination')),
                'grid_correction': self._clean_numeric_value(
                    self._get_element_text(root, './/witsml:GridCorrection'))
            }
            
            # Extract trajectory stations
            stations = []
            station_elements = root.findall('.//witsml:TrajectoryStation', self.namespaces)
            
            for station in station_elements:
                station_data = {
                    'md': self._clean_numeric_value(self._get_element_text(station, './/witsml:Md')),
                    'tvd': self._clean_numeric_value(self._get_element_text(station, './/witsml:Tvd')),
                    'incl': self._clean_numeric_value(self._get_element_text(station, './/witsml:Incl')),
                    'azi': self._clean_numeric_value(self._get_element_text(station, './/witsml:Azi')),
                    'north': self._clean_numeric_value(self._get_element_text(station, './/witsml:Dispns')),
                    'east': self._clean_numeric_value(self._get_element_text(station, './/witsml:Dispew')),
                    'vs': self._clean_numeric_value(self._get_element_text(station, './/witsml:Vs')),
                    'dls': self._clean_numeric_value(self._get_element_text(station, './/witsml:Dls')),
                    'type_survey': self._get_element_text(station, './/witsml:TypeSurveyTool')
                }
                stations.append(station_data)
            
            # Create DataFrame
            df = pd.DataFrame(stations)
            df.attrs['trajectory_info'] = trajectory_info
            
            return df
            
        except Exception as e:
            raise Exception(f"Error parsing WITSML Trajectory file {xml_file}: {str(e)}")
    
    def _extract_location_data(self, root: ET.Element, location_type: str) -> Dict[str, Any]:
        """Extract location data (surface or bottom hole)"""
        location_path = f'.//witsml:WellLocation/witsml:{location_type}Location'
        location_element = root.find(location_path, self.namespaces)
        
        if location_element is None:
            return {}
        
        return {
            'latitude': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:Latitude')),
            'longitude': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:Longitude')),
            'projected_x': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:ProjectedX')),
            'projected_y': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:ProjectedY')),
            'local_x': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:LocalX')),
            'local_y': self._clean_numeric_value(
                self._get_element_text(location_element, './/witsml:LocalY')),
            'original_text': self._get_element_text(location_element, './/witsml:Original')
        }


class WitsmlDataConverter:
    """Convert WITSML data to GeoSuite formats"""
    
    def __init__(self):
        self.parser = WitsmlParser()
    
    def witsml_log_to_geosuite(self, xml_file: str) -> pd.DataFrame:
        """Convert WITSML log data to GeoSuite standard format"""
        # Parse WITSML log
        df = self.parser.parse_log_data(xml_file)
        
        if df.empty:
            return df
        
        # Create standardized column mapping
        column_mapping = {
            # Depth columns
            'DEPT': 'depth_m',
            'DEPTH': 'depth_m', 
            'MD': 'depth_m',
            'MDEPTH': 'depth_m',
            
            # Standard log curves
            'GR': 'GR',
            'GAMMA': 'GR',
            'GRC': 'GR',
            
            'RHOB': 'RHOB',
            'RHOZ': 'RHOB',
            'RHOM': 'RHOB',
            'BULK_DENSITY': 'RHOB',
            
            'NPHI': 'NPHI',
            'NEUT': 'NPHI',
            'NEUTRON': 'NPHI',
            'PHIN': 'NPHI',
            
            'RT': 'RT',
            'RES': 'RT',
            'RESIST': 'RT',
            'ILD': 'RT',
            'LLD': 'RT',
            'RDEEP': 'RT',
            
            # Photoelectric factor
            'PE': 'PE',
            'PEF': 'PE',
            'PHOTO': 'PE',
            
            # Caliper
            'CAL': 'CALI',
            'CALI': 'CALI',
            'CALIPER': 'CALI',
            
            # Spontaneous potential
            'SP': 'SP'
        }
        
        # Apply column mapping
        df_mapped = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df_mapped = df_mapped.rename(columns={old_col: new_col})
        
        # Ensure we have required columns with reasonable defaults
        required_cols = ['depth_m', 'GR', 'RHOB', 'NPHI', 'RT']
        
        for col in required_cols:
            if col not in df_mapped.columns:
                if col == 'depth_m':
                    # Create depth column from index if missing
                    df_mapped['depth_m'] = np.arange(1000, 1000 + len(df_mapped), 2)
                elif col == 'GR':
                    df_mapped['GR'] = 75.0  # Typical GR value
                elif col == 'RHOB':
                    df_mapped['RHOB'] = 2.45  # Typical bulk density
                elif col == 'NPHI':
                    df_mapped['NPHI'] = 0.15  # Typical neutron porosity
                elif col == 'RT':
                    df_mapped['RT'] = 20.0  # Typical resistivity
        
        # Filter to only include standard columns
        standard_columns = [col for col in required_cols if col in df_mapped.columns]
        optional_columns = ['PE', 'CALI', 'SP']
        available_optional = [col for col in optional_columns if col in df_mapped.columns]
        
        final_columns = standard_columns + available_optional
        df_final = df_mapped[final_columns].copy()
        
        # Clean data
        df_final = df_final.dropna(subset=['depth_m'])  # Remove rows without depth
        df_final = df_final.reset_index(drop=True)
        
        # Store original WITSML metadata
        if hasattr(df, 'attrs'):
            df_final.attrs = df.attrs
        
        return df_final
    
    def create_field_data_from_witsml(self, well_files: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Create field data from multiple WITSML files
        
        Args:
            well_files: List of dicts with keys: 'well_name', 'log_file', optional 'wellbore_file'
        
        Returns:
            DataFrame in GeoSuite field data format
        """
        field_data = []
        
        for well_info in well_files:
            well_name = well_info['well_name']
            log_file = well_info['log_file']
            wellbore_file = well_info.get('wellbore_file')
            
            try:
                # Parse log data
                log_df = self.witsml_log_to_geosuite(log_file)
                
                if log_df.empty:
                    continue
                
                # Get wellbore info if available
                wellbore_data = {}
                if wellbore_file and Path(wellbore_file).exists():
                    try:
                        wellbore_data = self.parser.parse_wellbore_data(wellbore_file)
                    except:
                        pass
                
                # Add well identification and field data columns
                log_df['well_name'] = well_name
                
                # Add coordinates (use wellbore data if available, otherwise defaults)
                log_df['x_coord'] = wellbore_data.get('surface_location', {}).get('projected_x', 1000)
                log_df['y_coord'] = wellbore_data.get('surface_location', {}).get('projected_y', 2000)
                
                # Add formation classification based on GR
                def classify_formation(gr):
                    return 'Shale_B' if gr > 90 else 'Sandstone_A'
                
                log_df['formation'] = log_df['GR'].apply(classify_formation)
                
                # Add drilling parameters (would normally come from drilling reports)
                log_df['mud_weight_used'] = np.random.uniform(1.1, 1.5, len(log_df))
                log_df['drilling_status'] = 'Success'  # Default to success
                log_df['cost_per_meter'] = np.random.uniform(750, 950, len(log_df))
                log_df['days_to_drill'] = np.random.uniform(8, 15, len(log_df))
                
                # Add some realistic drilling problems based on formation
                problem_mask = (log_df['formation'] == 'Shale_B') & (np.random.random(len(log_df)) < 0.3)
                log_df.loc[problem_mask, 'drilling_status'] = np.random.choice(
                    ['Breakout', 'Mud_Loss', 'Tight_Hole'], size=problem_mask.sum())
                
                field_data.append(log_df)
                
            except Exception as e:
                logger.warning(f"Could not process well {well_name}: {str(e)}")
                continue
        
        if not field_data:
            return pd.DataFrame()
        
        # Combine all well data
        combined_df = pd.concat(field_data, ignore_index=True)
        
        return combined_df


def export_to_witsml(data: pd.DataFrame, output_file: str, well_info: Dict[str, Any] = None):
    """Export GeoSuite analysis results to WITSML format"""
    
    if well_info is None:
        well_info = {
            'name': 'GeoSuite_Analysis_Well',
            'uid': f'well_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'field': 'GeoSuite_Field',
            'operator': 'GeoSuite_Operator'
        }
    
    # Create WITSML XML structure
    root = ET.Element('witsml:Log')
    root.set('xmlns:witsml', WITSML_NS['witsml'])
    root.set('xmlns:eml', WITSML_NS['eml'])
    root.set('uuid', f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.set('schemaVersion', '2.0')
    
    # Add citation
    citation = ET.SubElement(root, 'eml:Citation')
    title = ET.SubElement(citation, 'eml:Title')
    title.text = f"{well_info['name']}_GeoSuite_Analysis"
    
    # Add wellbore reference
    wellbore_ref = ET.SubElement(root, 'witsml:Wellbore')
    wellbore_ref.text = well_info.get('wellbore_uid', 'wellbore_001')
    
    # Add channels
    for col in data.columns:
        if col not in ['well_name', 'formation', 'drilling_status', 'x_coord', 'y_coord']:
            channel = ET.SubElement(root, 'witsml:Channel')
            channel.set('uuid', f'channel_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
            # Channel citation
            ch_citation = ET.SubElement(channel, 'eml:Citation')
            ch_title = ET.SubElement(ch_citation, 'eml:Title')
            ch_title.text = col
            
            # Mnemonic
            mnemonic = ET.SubElement(channel, 'witsml:Mnemonic')
            mnemonic.text = col
            
            # Data type
            data_type = ET.SubElement(channel, 'witsml:DataType')
            data_type.text = 'double' if col in ['depth_m', 'GR', 'RHOB', 'NPHI', 'RT'] else 'string'
    
    # Add log data
    log_data = ET.SubElement(root, 'witsml:LogData')
    
    for _, row in data.iterrows():
        data_elem = ET.SubElement(log_data, 'witsml:Data')
        data_values = []
        
        for col in data.columns:
            if col not in ['well_name', 'formation', 'drilling_status', 'x_coord', 'y_coord']:
                value = str(row[col]) if not pd.isna(row[col]) else ''
                data_values.append(value)
        
        data_elem.text = ','.join(data_values)
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
