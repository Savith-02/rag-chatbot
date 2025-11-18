# app/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent

# Directory where raw PDF files are stored
RAW_FILES_PATH = os.getenv("RAW_FILES_PATH", str(BASE_DIR / "raw_files"))

# File to track which files have been processed
PROCESSED_FILES_TRACKER = os.getenv(
    "PROCESSED_FILES_TRACKER", 
    str(BASE_DIR / "processed_files.txt")
)

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Ensure raw_files directory exists
Path(RAW_FILES_PATH).mkdir(parents=True, exist_ok=True)
