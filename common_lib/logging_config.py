"""
Structured logging configuration using structlog.

Provides JSON-structured logs that are queryable and include:
- Timestamp
- Log level
- Component name
- Data snapshots (hashed for large arrays)
- Memory usage
- Execution time

Created: 2025-08-06
"""

import structlog
import logging
import sys
import json
import hashlib
import psutil
import numpy as np
from typing import Any, Dict, Optional
from datetime import datetime


def hash_array(arr: np.ndarray) -> str:
    """Hash a numpy array for logging without dumping all data."""
    if arr.size == 0:
        return "empty"
    
    # Create a hash of the array data
    if arr.dtype == np.object_:
        # Handle object arrays carefully
        return f"object_array_size_{arr.size}"
    
    try:
        # Use tobytes for numeric arrays
        data_hash = hashlib.md5(arr.tobytes()).hexdigest()[:8]
        return f"{data_hash}_shape_{arr.shape}_dtype_{arr.dtype}"
    except Exception:
        # Fallback for problematic arrays
        return f"unhashable_size_{arr.size}"


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class DataSnapshotProcessor:
    """Processor to capture data snapshots for logging."""
    
    @staticmethod
    def process_data(data: Any, max_size: int = 100) -> Dict[str, Any]:
        """
        Process data for logging, hashing large arrays.
        
        Args:
            data: Data to process
            max_size: Maximum size before hashing
            
        Returns:
            Processed data suitable for JSON logging
        """
        if data is None:
            return {"type": "none"}
        
        # Handle numpy arrays
        if isinstance(data, np.ndarray):
            if data.size > max_size:
                return {
                    "type": "ndarray",
                    "hash": hash_array(data),
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "min": float(np.nanmin(data)) if data.size > 0 else None,
                    "max": float(np.nanmax(data)) if data.size > 0 else None,
                    "has_nan": bool(np.isnan(data).any()),
                    "has_complex": bool(np.iscomplexobj(data))
                }
            else:
                return {
                    "type": "ndarray",
                    "data": data.tolist(),
                    "shape": data.shape,
                    "dtype": str(data.dtype)
                }
        
        # Handle pandas objects
        if hasattr(data, 'shape'):  # DataFrame or Series
            mem_usage = None
            if hasattr(data, 'memory_usage') and callable(data.memory_usage):
                mem = data.memory_usage(deep=True)
                # memory_usage returns int for Series, Series for DataFrame
                mem_usage = mem.sum() if hasattr(mem, 'sum') else mem
            
            return {
                "type": type(data).__name__,
                "shape": data.shape,
                "columns": list(data.columns) if hasattr(data, 'columns') else None,
                "index_type": type(data.index).__name__,
                "memory_usage": mem_usage
            }
        
        # Handle simple types
        if isinstance(data, (int, float, str, bool)):
            return {"type": type(data).__name__, "value": data}
        
        # Handle dicts
        if isinstance(data, dict):
            if len(data) > 10:
                return {
                    "type": "dict",
                    "size": len(data),
                    "keys_sample": list(data.keys())[:5]
                }
            else:
                return {"type": "dict", "data": data}
        
        # Handle lists
        if isinstance(data, (list, tuple)):
            if len(data) > max_size:
                return {
                    "type": type(data).__name__,
                    "size": len(data),
                    "sample": data[:5]
                }
            else:
                return {"type": type(data).__name__, "data": data}
        
        # Default
        return {"type": type(data).__name__, "repr": str(data)[:100]}


def configure_structlog(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure structlog for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Set up standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if not log_file else open(log_file, 'a'),
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured structlog logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class InstrumentedLogger:
    """
    Enhanced logger with data snapshot and performance tracking capabilities.
    """
    
    def __init__(self, name: str):
        """Initialize instrumented logger."""
        self.logger = get_logger(name)
        self.snapshot_processor = DataSnapshotProcessor()
    
    def log_data_transform(
        self, 
        stage: str, 
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a data transformation with snapshots.
        
        Args:
            stage: Stage name (e.g., "resampling", "volatility_calc")
            input_data: Input data to transformation
            output_data: Output data from transformation
            metadata: Additional metadata to log
        """
        log_data = {
            "stage": stage,
            "memory_mb": get_memory_usage()
        }
        
        if input_data is not None:
            log_data["input"] = self.snapshot_processor.process_data(input_data)
        
        if output_data is not None:
            log_data["output"] = self.snapshot_processor.process_data(output_data)
        
        if metadata:
            log_data.update(metadata)
        
        self.logger.info("data_transform", **log_data)
    
    def log_validation_error(
        self,
        stage: str,
        error: str,
        data: Any = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a validation error with context.
        
        Args:
            stage: Stage where validation failed
            error: Error description
            data: Data that failed validation
            metadata: Additional context
        """
        log_data = {
            "stage": stage,
            "error": error,
            "memory_mb": get_memory_usage()
        }
        
        if data is not None:
            log_data["failed_data"] = self.snapshot_processor.process_data(data)
        
        if metadata:
            log_data.update(metadata)
        
        self.logger.error("validation_error", **log_data)
    
    def log_complex_number_detection(
        self,
        location: str,
        variable: str,
        value: Any,
        context: Optional[Dict] = None
    ):
        """
        Log detection of complex numbers.
        
        Args:
            location: Where complex number was detected
            variable: Variable name
            value: The complex value
            context: Additional context
        """
        log_data = {
            "location": location,
            "variable": variable,
            "value": str(value),
            "is_complex": np.iscomplexobj(value),
            "memory_mb": get_memory_usage()
        }
        
        if context:
            log_data.update(context)
        
        self.logger.critical("complex_number_detected", **log_data)