"""
Test suite for generated bank statement parsers
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add custom_parsers to path
sys.path.insert(0, str(Path(__file__).parent.parent / "custom_parsers"))

class TestGeneratedParsers:
    """Test all generated parsers"""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def parsers_dir(self):
        return Path(__file__).parent.parent / "custom_parsers"
    
    def get_available_banks(self, data_dir):
        """Get list of banks with sample data"""
        banks = []
        if data_dir.exists():
            for bank_dir in data_dir.iterdir():
                if bank_dir.is_dir():
                    pdf_file = bank_dir / f"{bank_dir.name}_sample.pdf"
                    csv_file = bank_dir / f"{bank_dir.name}_sample.csv"
                    if pdf_file.exists() and csv_file.exists():
                        banks.append(bank_dir.name)
        return banks
    
    def get_available_parsers(self, parsers_dir):
        """Get list of generated parsers"""
        parsers = []
        if parsers_dir.exists():
            for parser_file in parsers_dir.glob("*_parser.py"):
                bank_name = parser_file.stem.replace("_parser", "")
                parsers.append(bank_name)
        return parsers
    
    def test_parser_files_exist(self, data_dir, parsers_dir):
        """Test that parser files are generated for available banks"""
        banks = self.get_available_banks(data_dir)
        parsers = self.get_available_parsers(parsers_dir)
        
        # At least one parser should exist after running agent
        if len(parsers) == 0:
            pytest.skip("No parsers generated yet - run 'python agent.py --target icici' first")
        
        # Test specific banks if they exist
        for bank in banks:
            if bank in ["icici"]:  # Add more banks as needed
                assert bank in parsers, f"Parser not found for {bank}"
    
    def test_icici_parser(self, data_dir, parsers_dir):
        """Test ICICI bank parser specifically"""
        bank = "icici"
        
        # Skip if no ICICI data
        pdf_path = data_dir / bank / f"{bank}_sample.pdf"
        csv_path = data_dir / bank / f"{bank}_sample.csv"
        parser_path = parsers_dir / f"{bank}_parser.py"
        
        if not all(p.exists() for p in [pdf_path, csv_path, parser_path]):
            pytest.skip(f"ICICI test files not found - run 'python agent.py --target icici' first")
        
        # Import parser
        try:
            if f"{bank}_parser" in sys.modules:
                del sys.modules[f"{bank}_parser"]
            parser_module = __import__(f"{bank}_parser")
        except ImportError as e:
            pytest.fail(f"Failed to import {bank}_parser: {e}")
        
        # Test parse function exists
        assert hasattr(parser_module, 'parse'), "parse() function not found"
        
        # Load expected data
        expected_df = pd.read_csv(csv_path)
        
        # Test parser
        try:
            result_df = parser_module.parse(str(pdf_path))
        except Exception as e:
            pytest.fail(f"Parser execution failed: {e}")
        
        # Validate result
        assert isinstance(result_df, pd.DataFrame), "parse() must return DataFrame"
        assert len(result_df) > 0, "Parser returned empty DataFrame"
        assert set(result_df.columns) == set(expected_df.columns), \
            f"Column mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])