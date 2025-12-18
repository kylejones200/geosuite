from __future__ import annotations
from typing import Tuple
import pandas as pd
import importlib.resources as pkg_resources


def _read_parquet_from_package(filename: str) -> pd.DataFrame:
    with pkg_resources.files(__package__).joinpath(f"files/{filename}").open("rb") as f:
        return pd.read_parquet(f)


def load_demo_well_logs() -> pd.DataFrame:
    """Return a small demo well-log DataFrame with columns: depth_m, GR, RHOB, NPHI, RT."""
    return _read_parquet_from_package("demo_well_logs.parquet")


def load_demo_facies() -> pd.DataFrame:
    """Return a small demo facies-labeled log sample with columns incl. facies."""
    return _read_parquet_from_package("demo_facies.parquet")


def load_field_data() -> pd.DataFrame:
    """Return multi-well field data for field-wide analysis."""
    return _read_parquet_from_package("field_data.parquet")


def load_facies_training_data() -> pd.DataFrame:
    """
    Return the facies classification training dataset.
    
    This dataset contains well log data with expert-labeled facies from 9 wells
    from the University of Kansas exercise. Includes features like GR, NPHI, RHOB,
    PE, and depth, with corresponding facies labels.
    
    Source: https://github.com/brendonhall/facies_classification
    """
    return _read_parquet_from_package("training_data.parquet")


def load_facies_validation_data() -> pd.DataFrame:
    """
    Return the facies classification validation dataset (unlabeled).
    
    This dataset contains well log data without facies labels for testing
    trained models on unseen data.
    """
    return _read_parquet_from_package("validation_data_nofacies.parquet")


def load_facies_vectors() -> pd.DataFrame:
    """
    Return the complete facies vectors dataset.
    
    This is the full dataset including all wells with facies labels.
    """
    return _read_parquet_from_package("facies_vectors.parquet")


def load_facies_well_data() -> pd.DataFrame:
    """
    Return the well data with facies labels.
    
    Complete well log data with facies classifications from multiple wells.
    """
    return _read_parquet_from_package("well_data_with_facies.parquet")


def load_kansas_training_wells() -> pd.DataFrame:
    """
    Return the Kansas dataset training wells.
    
    Part of the Kansas University dataset for facies classification.
    """
    return _read_parquet_from_package("KSdata/training_wells.parquet")


def load_kansas_test_wells() -> pd.DataFrame:
    """
    Return the Kansas dataset test wells.
    
    Part of the Kansas University dataset for facies classification.
    """
    return _read_parquet_from_package("KSdata/test_wells.parquet")
