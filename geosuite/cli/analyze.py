"""
Analysis and reporting utilities for well log data.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Import LAS loader
try:
    import lasio
    _has_lasio = True
    
    def load_las_file(file_path):
        """Load LAS file using lasio."""
        las = lasio.read(str(file_path))
        return las.df()
except ImportError:
    _has_lasio = False
    def load_las_file(*args, **kwargs):
        raise ImportError("lasio is required for LAS file loading. Install with: pip install lasio")
from ..petro import pickett_plot, buckles_plot
from ..geomech import calculate_overburden_stress, calculate_hydrostatic_pressure

logger = logging.getLogger(__name__)


def analyze_well(
    well_file: Path,
    output_dir: Path,
    create_plots: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a single well.
    
    Parameters
    ----------
    well_file : Path
        Path to well log file (LAS or CSV)
    output_dir : Path
        Directory for output files
    create_plots : bool, default True
        Whether to create visualization plots
        
    Returns
    -------
    Dict[str, Any]
        Analysis results and statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing well: {well_file.name}")
    
    # Load data
    if well_file.suffix.lower() == '.las':
        if not _has_lasio:
            raise ImportError("lasio is required for LAS files")
        df = load_las_file(well_file)
    else:
        raise ValueError(f"Unsupported file format: {well_file.suffix}")
    
    results = {
        "well_name": well_file.stem,
        "total_depth": float(df["DEPTH"].max()) if "DEPTH" in df.columns else None,
        "num_samples": len(df),
        "available_logs": list(df.columns),
        "statistics": {}
    }
    
    # Calculate basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        results["statistics"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median())
        }
    
    # Create plots if requested
    if create_plots:
        plot_dir = output_path / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Pickett plot if resistivity and porosity available
        if "RT" in df.columns and "NPHI" in df.columns:
            try:
                fig = pickett_plot(df, porosity_col="NPHI", resistivity_col="RT")
                fig.savefig(plot_dir / "pickett_plot.png", dpi=300, bbox_inches="tight")
                results["plots"] = ["pickett_plot.png"]
            except Exception as e:
                logger.warning(f"Failed to create Pickett plot: {e}")
        
        # Buckles plot if porosity and saturation available
        if "PHIND" in df.columns and "SW" in df.columns:
            try:
                fig = buckles_plot(df, porosity_col="PHIND", sw_col="SW")
                fig.savefig(plot_dir / "buckles_plot.png", dpi=300, bbox_inches="tight")
                if "plots" not in results:
                    results["plots"] = []
                results["plots"].append("buckles_plot.png")
            except Exception as e:
                logger.warning(f"Failed to create Buckles plot: {e}")
    
    # Save results
    results_file = output_path / f"{well_file.stem}_analysis.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {results_file}")
    
    return results


def create_analysis_report(
    well_files: list[Path],
    output_dir: Path
) -> pd.DataFrame:
    """
    Create a summary report for multiple wells.
    
    Parameters
    ----------
    well_files : list[Path]
        List of well log file paths
    output_dir : Path
        Output directory
        
    Returns
    -------
    pd.DataFrame
        Summary report DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for well_file in well_files:
        try:
            results = analyze_well(well_file, output_path, create_plots=False)
            all_results.append({
                "Well": results["well_name"],
                "Total_Depth": results["total_depth"],
                "Num_Samples": results["num_samples"],
                "Available_Logs": ", ".join(results["available_logs"])
            })
        except Exception as e:
            logger.error(f"Failed to analyze {well_file.name}: {e}")
    
    report_df = pd.DataFrame(all_results)
    
    # Save report
    report_file = output_path / "well_analysis_report.csv"
    report_df.to_csv(report_file, index=False)
    
    logger.info(f"Report saved to {report_file}")
    
    return report_df

