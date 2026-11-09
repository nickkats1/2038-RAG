import pytest

from src.data_utils import load_documents

class TestLoadPDF:
    """test that the pdf file is successfully loaded"""
    def test_loads_documents(self, pdf_file):
        """test that file is successfully loaded"""
        
        docs = load_documents(pdf_file)
        
        assert docs is not None
        
        
        




