"""
Unit tests for crime data ingestion
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.ingest_crime import validate_crime_data


class TestCrimeDataValidation:
    """Test suite for crime data validation"""
    
    def test_validate_with_valid_data(self, tmp_path):
        """Test validation with valid crime data"""
        # Create temporary valid CSV
        test_data = {
            'INCIDENT_NUMBER': ['I12345', 'I12346', 'I12347'],
            'OFFENSE_CODE': [619, 724, 3410],
            'DISTRICT': ['D14', 'C11', 'A1'],
            'OCCURRED_ON_DATE': ['2023-01-01 10:00:00', '2023-01-02 14:30:00', '2023-01-03 20:15:00'],
            'Lat': [42.3601, 42.3602, 42.3603],
            'Long': [-71.0589, -71.0590, -71.0591]
        }
        df = pd.DataFrame(test_data)
        
        # Save to temporary file
        test_file = tmp_path / "test_crime.csv"
        df.to_csv(test_file, index=False)
        
        # Run validation
        result = validate_crime_data(str(test_file))
        
        # Assertions
        assert result['total_rows'] == 3
        assert result['duplicate_rows'] == 0
        assert len(result['critical_issues']) == 0
    
    def test_validate_detects_duplicates(self, tmp_path):
        """Test that validation detects duplicate rows"""
        # Create CSV with duplicates
        test_data = {
            'INCIDENT_NUMBER': ['I12345', 'I12345', 'I12346'],  # Duplicate
            'OFFENSE_CODE': [619, 619, 724],
            'DISTRICT': ['D14', 'D14', 'C11'],
            'OCCURRED_ON_DATE': ['2023-01-01', '2023-01-01', '2023-01-02'],
        }
        df = pd.DataFrame(test_data)
        
        test_file = tmp_path / "test_crime_dupes.csv"
        df.to_csv(test_file, index=False)
        
        # Run validation
        result = validate_crime_data(str(test_file))
        
        # Should detect duplicates
        assert result['duplicate_rows'] > 0
    
    def test_validate_detects_empty_dataset(self, tmp_path):
        """Test that validation detects empty dataset"""
        # Create empty CSV
        df = pd.DataFrame()
        
        test_file = tmp_path / "test_crime_empty.csv"
        df.to_csv(test_file, index=False)
        
        # Run validation
        result = validate_crime_data(str(test_file))
        
        # Should flag as critical issue
        assert 'Dataset is empty' in result['critical_issues']
    
    def test_validate_detects_invalid_coordinates(self, tmp_path):
        """Test that validation detects invalid coordinates"""
        # Create data with invalid coordinates
        test_data = {
            'INCIDENT_NUMBER': ['I12345', 'I12346', 'I12347'],
            'OFFENSE_CODE': [619, 724, 3410],
            'DISTRICT': ['D14', 'C11', 'A1'],
            'Lat': [99.9999, 99.9999, 99.9999],  # Invalid - outside Boston
            'Long': [-200.0, -200.0, -200.0]     # Invalid
        }
        df = pd.DataFrame(test_data)
        
        test_file = tmp_path / "test_crime_bad_coords.csv"
        df.to_csv(test_file, index=False)
        
        # Run validation
        result = validate_crime_data(str(test_file))
        
        # Should detect invalid coordinates
        assert any('invalid coordinates' in issue.lower() for issue in result['critical_issues'])


class TestDataQuality:
    """Test suite for data quality checks"""
    
    def test_expected_columns_present(self):
        """Test that expected columns are present in real data"""
        if os.path.exists('data/raw/crime_data.csv'):
            df = pd.read_csv('data/raw/crime_data.csv', nrows=100)
            
            expected_cols = ['INCIDENT_NUMBER', 'OFFENSE_CODE', 'DISTRICT']
            for col in expected_cols:
                assert col in df.columns, f"Missing expected column: {col}"
    
    def test_no_all_null_columns(self):
        """Test that no columns are completely null"""
        if os.path.exists('data/raw/crime_data.csv'):
            df = pd.read_csv('data/raw/crime_data.csv', nrows=1000)
        
        # Allow some columns to be all null (e.g., optional fields)
        # but critical columns should have data
            critical_columns = ['INCIDENT_NUMBER', 'OFFENSE_CODE', 'DISTRICT', 'OCCURRED_ON_DATE']
        
            for col in critical_columns:
                if col in df.columns:
                    null_pct = df[col].isnull().sum() / len(df)
                    assert null_pct < 1.0, f"Critical column {col} is completely null"
    
    def test_incident_numbers_unique(self):
        """Test that incident numbers are reasonably unique"""
        if os.path.exists('data/raw/crime_data.csv'):
            df = pd.read_csv('data/raw/crime_data.csv', nrows=1000)
            
            if 'INCIDENT_NUMBER' in df.columns:
                unique_pct = df['INCIDENT_NUMBER'].nunique() / len(df)
                assert unique_pct > 0.8, "Too many duplicate incident numbers"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])