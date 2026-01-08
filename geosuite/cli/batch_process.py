"""
Batch processing utilities for well log data.
"""
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Import loaders - handle optional dependencies
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

try:
    import pandas as pd
    _has_csv = True
    
    def load_csv_data(file_path):
        """Load CSV file."""
        return pd.read_csv(file_path)
except ImportError:
    _has_csv = False
    def load_csv_data(*args, **kwargs):
        raise ImportError("pandas is required for CSV loading")
from ..petro import calculate_porosity_from_density, calculate_water_saturation
from ..geomech import calculate_overburden_stress

logger = logging.getLogger(__name__)


def batch_process_wells(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.las",
    output_format: str = "csv",
    calculate_properties: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process multiple well log files in batch.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing input files
    output_dir : Path
        Directory for output files
    file_pattern : str, default "*.las"
        File pattern to match (e.g., "*.las", "*.csv")
    output_format : str, default "csv"
        Output format: "csv", "parquet", or "json"
    calculate_properties : list, optional
        List of properties to calculate: ["porosity", "saturation", "stress"]
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics and processing results
    """
    if calculate_properties is None:
        calculate_properties = []
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = list(input_path.glob(file_pattern))
    
    if not files:
        logger.warning(f"No files found matching pattern '{file_pattern}' in {input_dir}")
        return {"processed": 0, "failed": 0, "files": []}
    
    logger.info(f"Found {len(files)} files to process")
    
    results = {
        "processed": 0,
        "failed": 0,
        "files": []
    }
    
    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}...")
            
            # Load data based on file extension
            loaders = {
                '.las': (load_las_file, _has_lasio, "lasio"),
                '.csv': (load_csv_data, _has_csv, "pandas"),
            }
            
            loader_func, has_lib, lib_name = loaders.get(
                file_path.suffix.lower(), (None, False, None)
            )
            
            if loader_func is None:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                results["failed"] += 1
                continue
            
            if not has_lib:
                raise ImportError(f"{lib_name} is required for {file_path.suffix} files")
            
            df = loader_func(file_path)
            
            # Calculate properties
            if "porosity" in calculate_properties and "RHOB" in df.columns:
                df["POROSITY"] = calculate_porosity_from_density(df["RHOB"])
            
            if "saturation" in calculate_properties and all(c in df.columns for c in ["RT", "POROSITY"]):
                df["SW"] = calculate_water_saturation(df["POROSITY"], df["RT"])
            
            if "stress" in calculate_properties and all(c in df.columns for c in ["DEPTH", "RHOB"]):
                df["SV"] = calculate_overburden_stress(df["DEPTH"].values, df["RHOB"].values)
            
            # Save output
            output_file = output_path / f"{file_path.stem}.{output_format}"
            
            savers = {
                "csv": lambda f: df.to_csv(f, index=False),
                "parquet": lambda f: df.to_parquet(f, index=False),
                "json": lambda f: df.to_json(f, orient="records", indent=2),
            }
            
            if output_format not in savers:
                raise ValueError(f"Unsupported output format: {output_format}. Choose: {', '.join(savers.keys())}")
            
            savers[output_format](output_file)
            
            results["processed"] += 1
            results["files"].append({
                "input": str(file_path),
                "output": str(output_file),
                "rows": len(df),
                "columns": len(df.columns)
            })
            
            logger.info(f"✓ Processed {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            results["failed"] += 1
    
    logger.info(
        f"Batch processing complete: {results['processed']} processed, "
        f"{results['failed']} failed"
    )
    
    return results


def process_las_files(
    input_files: List[Path],
    output_dir: Path,
    output_format: str = "csv"
) -> pd.DataFrame:
    """
    Process multiple LAS files and combine into a single DataFrame.
    
    Parameters
    ----------
    input_files : List[Path]
        List of LAS file paths
    output_dir : Path
        Output directory
    output_format : str, default "csv"
        Output format
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all well data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for file_path in input_files:
        try:
            logger.info(f"Loading {file_path.name}...")
            data = load_las_file(file_path)
            df = data.df() if hasattr(data, 'df') else data
            
            # Add well identifier
            df["WELL"] = file_path.stem
            
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded from input files")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined output
    output_file = output_path / f"combined_wells.{output_format}"
    
    savers = {
        "csv": lambda: combined_df.to_csv(output_file, index=False),
        "parquet": lambda: combined_df.to_parquet(output_file, index=False),
    }
    
    if output_format not in savers:
        raise ValueError(f"Unsupported output format: {output_format}. Choose: {', '.join(savers.keys())}")
    
    savers[output_format]()
    
    logger.info(
        f"Combined {len(input_files)} files into {len(combined_df)} rows. "
        f"Saved to {output_file}"
    )
    
    return combined_df


def main():
    """Command-line entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process well log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all LAS files in a directory
  geosuite-batch --input ./wells --output ./processed --pattern "*.las"
  
  # Process with property calculations
  geosuite-batch --input ./wells --output ./processed --calculate porosity saturation
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing well log files"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed files"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.las",
        help="File pattern to match (default: *.las)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--calculate",
        nargs="*",
        choices=["porosity", "saturation", "stress"],
        help="Properties to calculate"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run batch processing
    results = batch_process_wells(
        input_dir=args.input,
        output_dir=args.output,
        file_pattern=args.pattern,
        output_format=args.format,
        calculate_properties=args.calculate or []
    )
    
    # Print summary
    print(f"\nBatch Processing Summary:")
    print(f"  Processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total files: {len(results['files'])}")


if __name__ == "__main__":
    main()

