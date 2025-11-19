# app/spreadsheet_logger.py

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Spreadsheet configuration
SPREADSHEET_DIR = Path("logs")
SPREADSHEET_FILE = SPREADSHEET_DIR / "rag_performance_log.xlsx"

# Column names for the spreadsheet
COLUMNS = [
    "Timestamp",
    "Question", 
    "Generated_Answer",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
    "Total_Time_Seconds",
    "Num_Documents_Retrieved",
    "Retrieved_Documents",
    "Status"
]


def initialize_spreadsheet() -> None:
    """
    Initialize the spreadsheet file if it doesn't exist.
    Creates the logs directory and an empty Excel file with headers.
    """
    try:
        # Create logs directory if it doesn't exist
        SPREADSHEET_DIR.mkdir(exist_ok=True)
        
        # Check if spreadsheet already exists
        if SPREADSHEET_FILE.exists():
            logger.info(f"Spreadsheet already exists: {SPREADSHEET_FILE}")
            return
        
        # Create new spreadsheet with headers
        df = pd.DataFrame(columns=COLUMNS)
        df.to_excel(SPREADSHEET_FILE, index=False, engine='openpyxl')
        
        logger.info(f"Created new spreadsheet: {SPREADSHEET_FILE}")
        
    except Exception as e:
        logger.error(f"Failed to initialize spreadsheet: {e}")


def log_rag_performance(
    question: str,
    generated_answer: str,
    retrieval_time: float,
    generation_time: float,
    num_documents_retrieved: int,
    status: str = "success",
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Log RAG performance data to the spreadsheet.
    
    Args:
        question: The user's question
        generated_answer: The generated answer from the LLM
        retrieval_time: Time taken for document retrieval (seconds)
        generation_time: Time taken for answer generation (seconds)
        num_documents_retrieved: Number of documents retrieved
        status: Status of the operation (success, error, no_documents_found)
        retrieved_docs: List of retrieved document dictionaries for logging
    """
    try:
        timestamp = datetime.now().isoformat()
        total_time = retrieval_time + generation_time
        
        # Format retrieved documents for logging (file names only)
        if retrieved_docs:
            file_names = [doc.get("file_name", "Unknown") for doc in retrieved_docs]
            retrieved_docs_text = "\n".join(file_names)
        else:
            retrieved_docs_text = "No documents retrieved"
        
        # Create new record
        new_record = {
            "Timestamp": timestamp,
            "Question": question[:500],  # Limit question length
            "Generated_Answer": generated_answer[:1000],  # Limit answer length
            "Retrieval_Time_Seconds": round(retrieval_time, 3),
            "Generation_Time_Seconds": round(generation_time, 3),
            "Total_Time_Seconds": round(total_time, 3),
            "Num_Documents_Retrieved": num_documents_retrieved,
            "Retrieved_Documents": retrieved_docs_text[:1000],  # Limit retrieved docs length
            "Status": status
        }
        
        # Read existing data
        if SPREADSHEET_FILE.exists():
            df = pd.read_excel(SPREADSHEET_FILE, engine='openpyxl')
        else:
            df = pd.DataFrame(columns=COLUMNS)
        
        # Add new record
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Write back to spreadsheet
        df.to_excel(SPREADSHEET_FILE, index=False, engine='openpyxl')
        
        logger.debug(f"Logged performance data to spreadsheet: {retrieval_time:.3f}s retrieval, {generation_time:.3f}s generation")
        
    except Exception as e:
        logger.error(f"Failed to log performance data to spreadsheet: {e}")


def get_performance_stats() -> Optional[dict]:
    """
    Get basic performance statistics from the spreadsheet.
    
    Returns:
        Dictionary with performance statistics or None if error
    """
    try:
        if not SPREADSHEET_FILE.exists():
            return None
            
        df = pd.read_excel(SPREADSHEET_FILE, engine='openpyxl')
        
        if df.empty:
            return {"total_requests": 0}
        
        stats = {
            "total_requests": len(df),
            "avg_retrieval_time": df["Retrieval_Time_Seconds"].mean(),
            "avg_generation_time": df["Generation_Time_Seconds"].mean(),
            "avg_total_time": df["Total_Time_Seconds"].mean(),
            "success_rate": (df["Status"] == "success").mean() * 100
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return None
