import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass


ROOT_DIR = Path(__file__).parent.resolve(strict=False)
DATA_DIR = ROOT_DIR / "data" / "2038.pdf"


class Config:
    
    
    chunk_size: int = 350
    chunk_overlap: int = 35
    
    
    