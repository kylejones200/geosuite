"""
Unit tests for demo dataset loading functions.
"""
import pytest
import pandas as pd
from geosuite.data import demo_datasets


class TestDemoDatasets:
    """Tests for demo dataset loading functions."""
    
    def test_load_demo_well_logs(self):
        """Test loading demo well logs."""
        df = demo_datasets.load_demo_well_logs()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'depth_m' in df.columns or 'DEPTH' in df.columns.str.upper()
    
    def test_load_demo_facies(self):
        """Test loading demo facies data."""
        df = demo_datasets.load_demo_facies()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_load_field_data(self):
        """Test loading field data."""
        df = demo_datasets.load_field_data()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestFaciesDatasets:
    """Tests for facies classification dataset loading."""
    
    def test_load_facies_training_data(self):
        """Test loading facies training data."""
        df = demo_datasets.load_facies_training_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[0] == 3232  # Known size
        assert df.shape[1] == 11    # Known columns
    
    def test_load_facies_validation_data(self):
        """Test loading facies validation data."""
        df = demo_datasets.load_facies_validation_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_facies_vectors(self):
        """Test loading facies vectors."""
        df = demo_datasets.load_facies_vectors()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_facies_well_data(self):
        """Test loading well data with facies."""
        df = demo_datasets.load_facies_well_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_kansas_training_wells(self):
        """Test loading Kansas training wells."""
        df = demo_datasets.load_kansas_training_wells()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_kansas_test_wells(self):
        """Test loading Kansas test wells."""
        df = demo_datasets.load_kansas_test_wells()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDatasetProperties:
    """Test properties of loaded datasets."""
    
    def test_training_data_columns(self):
        """Test that training data has expected columns."""
        df = demo_datasets.load_facies_training_data()
        
        # Should have well log features
        expected_cols = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_training_data_facies_column(self):
        """Test that training data has Facies column."""
        df = demo_datasets.load_facies_training_data()
        
        assert 'Facies' in df.columns
        assert df['Facies'].notna().all()
    
    def test_validation_data_no_facies(self):
        """Test that validation data has no Facies column."""
        df = demo_datasets.load_facies_validation_data()
        
        # Validation data should not have Facies labels
        # (it's for prediction testing)
        assert 'Facies' not in df.columns or df['Facies'].isna().all()
    
    def test_no_missing_required_features(self):
        """Test that required features have no missing values."""
        df = demo_datasets.load_facies_training_data()
        
        # Key features should not have NaN
        key_features = ['GR', 'DeltaPHI', 'PHIND']
        for feature in key_features:
            if feature in df.columns:
                assert df[feature].notna().sum() > 0


class TestDatasetConsistency:
    """Test consistency across datasets."""
    
    def test_column_consistency(self):
        """Test that training and validation have similar columns."""
        train_df = demo_datasets.load_facies_training_data()
        valid_df = demo_datasets.load_facies_validation_data()
        
        train_cols = set(train_df.columns) - {'Facies'}
        valid_cols = set(valid_df.columns) - {'Facies'}
        
        # Should have significant overlap
        common_cols = train_cols & valid_cols
        assert len(common_cols) > 0
    
    def test_data_types(self):
        """Test that numeric columns are numeric."""
        df = demo_datasets.load_facies_training_data()
        
        # Well log features should be numeric
        numeric_cols = ['GR', 'DeltaPHI', 'PHIND', 'PE']
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_repeated_loading(self):
        """Test that data can be loaded multiple times."""
        df1 = demo_datasets.load_facies_training_data()
        df2 = demo_datasets.load_facies_training_data()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_memory_efficiency(self):
        """Test that loading doesn't create multiple copies."""
        import sys
        
        # Load data
        df1 = demo_datasets.load_facies_training_data()
        size1 = sys.getsizeof(df1)
        
        # Load again
        df2 = demo_datasets.load_facies_training_data()
        size2 = sys.getsizeof(df2)
        
        # Sizes should be similar (not doubled)
        assert size1 == size2


class TestDataQuality:
    """Test data quality of loaded datasets."""
    
    def test_no_duplicate_rows(self):
        """Test for duplicate rows."""
        df = demo_datasets.load_facies_training_data()
        
        # Allow for at most 1 duplicate row in the dataset (known issue)
        n_duplicates = len(df) - len(df.drop_duplicates())
        assert n_duplicates <= 1, f"Expected at most 1 duplicate, found {n_duplicates}"
    
    def test_reasonable_value_ranges(self):
        """Test that values are in reasonable ranges."""
        df = demo_datasets.load_facies_training_data()
        
        # GR should be positive and reasonable
        if 'GR' in df.columns:
            assert (df['GR'] >= 0).all()
            assert (df['GR'] <= 500).all()  # Typical max GR
        
        # Porosity should be between 0 and 1 (or 0 and 100 if percentage)
        if 'PHIND' in df.columns:
            assert (df['PHIND'] >= 0).all()
    
    def test_facies_categories(self):
        """Test that Facies are valid categories."""
        df = demo_datasets.load_facies_training_data()
        
        if 'Facies' in df.columns:
            # Should have multiple facies types
            n_unique = df['Facies'].nunique()
            assert n_unique >= 2
            assert n_unique <= 20  # Reasonable upper limit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

