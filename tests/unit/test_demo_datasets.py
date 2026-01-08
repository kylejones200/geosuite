"""
Unit tests for demo dataset loading functions.
"""
import pytest
import pandas as pd
from geosuite.data import demo_datasets


class TestDemoDatasets:
    """Tests for demo dataset loading functions."""
    
    def test_load_facies_training_data(self):
        """Test loading facies training data with validation."""
        df = demo_datasets.load_facies_training_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[0] == 3232
        assert df.shape[1] == 11
        
        # Check required columns
        expected_cols = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'Facies']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check data types
        numeric_cols = ['GR', 'DeltaPHI', 'PHIND', 'PE']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
        
        # Check Facies column
        assert df['Facies'].notna().all()
        n_unique = df['Facies'].nunique()
        assert 2 <= n_unique <= 20
    
    def test_load_facies_validation_data(self):
        """Test loading facies validation data."""
        df = demo_datasets.load_facies_validation_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Facies' not in df.columns or df['Facies'].isna().all()
    
    def test_load_other_datasets(self):
        """Test loading other demo datasets."""
        datasets = {
            'load_demo_well_logs': demo_datasets.load_demo_well_logs,
            'load_demo_facies': demo_datasets.load_demo_facies,
            'load_field_data': demo_datasets.load_field_data,
            'load_facies_vectors': demo_datasets.load_facies_vectors,
            'load_facies_well_data': demo_datasets.load_facies_well_data,
            'load_kansas_training_wells': demo_datasets.load_kansas_training_wells,
            'load_kansas_test_wells': demo_datasets.load_kansas_test_wells,
        }
        
        for name, loader in datasets.items():
            df = loader()
            assert isinstance(df, pd.DataFrame), f"{name} failed"
            assert not df.empty, f"{name} returned empty DataFrame"
    
    def test_dataset_consistency(self):
        """Test consistency between training and validation datasets."""
        train_df = demo_datasets.load_facies_training_data()
        valid_df = demo_datasets.load_facies_validation_data()
        
        train_cols = set(train_df.columns) - {'Facies'}
        valid_cols = set(valid_df.columns) - {'Facies'}
        common_cols = train_cols & valid_cols
        
        assert len(common_cols) > 0, "Training and validation should share columns"
    
    def test_data_quality(self):
        """Test data quality checks."""
        df = demo_datasets.load_facies_training_data()
        
        # Check for duplicates
        n_duplicates = len(df) - len(df.drop_duplicates())
        assert n_duplicates <= 1, f"Too many duplicates: {n_duplicates}"
        
        # Check value ranges
        if 'GR' in df.columns:
            assert (df['GR'] >= 0).all()
            assert (df['GR'] <= 500).all()
        
        if 'PHIND' in df.columns:
            assert (df['PHIND'] >= 0).all()
    
    def test_reproducibility(self):
        """Test that data loading is reproducible."""
        df1 = demo_datasets.load_facies_training_data()
        df2 = demo_datasets.load_facies_training_data()
        
        pd.testing.assert_frame_equal(df1, df2)
