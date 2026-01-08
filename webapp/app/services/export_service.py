"""
Export service for generating downloadable results.

Provides functionality for exporting analysis results in various formats
(CSV, JSON, Excel, PDF reports).
"""
import logging
import json
import io
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import report generation libraries
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF export will be limited.")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available. Excel export will be limited.")


class ExportService:
    """Service for exporting analysis results."""
    
    def __init__(self):
        """Initialize export service."""
        pass
    
    def export_to_csv(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        filename: Optional[str] = None
    ) -> bytes:
        """
        Export data to CSV format.
        
        Parameters
        ----------
        data : pd.DataFrame or dict
            Data to export
        filename : str, optional
            Output filename
            
        Returns
        -------
        bytes
            CSV file content
        """
        if isinstance(data, dict):
            # Convert dict to DataFrame
            if 'data' in data and isinstance(data['data'], list):
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            df = data
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')
    
    def export_to_json(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        filename: Optional[str] = None
    ) -> bytes:
        """
        Export data to JSON format.
        
        Parameters
        ----------
        data : pd.DataFrame or dict
            Data to export
        filename : str, optional
            Output filename
            
        Returns
        -------
        bytes
            JSON file content
        """
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict('records')
        else:
            json_data = data
        
        return json.dumps(json_data, indent=2, default=str).encode('utf-8')
    
    def export_to_excel(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: Optional[str] = None
    ) -> bytes:
        """
        Export data to Excel format.
        
        Parameters
        ----------
        data : pd.DataFrame or dict of DataFrames
            Data to export (dict creates multiple sheets)
        filename : str, optional
            Output filename
            
        Returns
        -------
        bytes
            Excel file content
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export")
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if isinstance(data, dict):
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                data.to_excel(writer, sheet_name='Sheet1', index=False)
        
        output.seek(0)
        return output.read()
    
    def export_to_pdf_report(
        self,
        title: str,
        sections: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> bytes:
        """
        Export results as PDF report.
        
        Parameters
        ----------
        title : str
            Report title
        sections : list of dict
            Report sections, each with 'title' and 'content'
        filename : str, optional
            Output filename
            
        Returns
        -------
        bytes
            PDF file content
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF export")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        
        # Sections
        for section in sections:
            section_title = section.get('title', 'Section')
            section_content = section.get('content', '')
            
            story.append(Paragraph(section_title, styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Handle different content types
            if isinstance(section_content, str):
                story.append(Paragraph(section_content, styles['Normal']))
            elif isinstance(section_content, list):
                # Table
                table = Table(section_content)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            
            story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def create_analysis_report(
        self,
        analysis_type: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        format: str = 'pdf'
    ) -> bytes:
        """
        Create a formatted analysis report.
        
        Parameters
        ----------
        analysis_type : str
            Type of analysis (e.g., 'petrophysics', 'geomechanics')
        inputs : dict
            Input parameters
        results : dict
            Analysis results
        format : str, default 'pdf'
            Output format ('pdf', 'json', 'csv')
            
        Returns
        -------
        bytes
            Report file content
        """
        if format == 'pdf':
            sections = [
                {
                    'title': 'Analysis Type',
                    'content': analysis_type
                },
                {
                    'title': 'Input Parameters',
                    'content': json.dumps(inputs, indent=2)
                },
                {
                    'title': 'Results',
                    'content': json.dumps(results, indent=2, default=str)
                }
            ]
            return self.export_to_pdf_report(
                title=f"{analysis_type.title()} Analysis Report",
                sections=sections
            )
        elif format == 'json':
            report_data = {
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'inputs': inputs,
                'results': results
            }
            return self.export_to_json(report_data)
        else:
            # CSV format - flatten results
            if isinstance(results, dict) and 'data' in results:
                return self.export_to_csv(results['data'])
            else:
                return self.export_to_csv(results)

