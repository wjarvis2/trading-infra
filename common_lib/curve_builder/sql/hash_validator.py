"""
SHA-256 hash validation for SQL files.
Ensures SQL queries haven't been modified without updating Python implementations.
"""

import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SQLHashValidator:
    """Validates SQL file integrity using SHA-256 hashes."""
    
    def __init__(self, project_root: Path):
        """
        Initialize hash validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.hash_cache: Dict[str, str] = {}
    
    def calculate_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hex digest of the hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def validate_file(
        self,
        relative_path: str,
        expected_hash: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a single SQL file.
        
        Args:
            relative_path: Path relative to project root
            expected_hash: Expected hash value (None for first time)
            
        Returns:
            Tuple of (is_valid, current_hash, error_message)
        """
        full_path = self.project_root / relative_path
        
        if not full_path.exists():
            return False, "", f"File not found: {relative_path}"
        
        try:
            current_hash = self.calculate_hash(full_path)
            
            if expected_hash is None:
                # First time - just return the hash
                logger.info(f"Initial hash for {relative_path}: {current_hash[:16]}...")
                return True, current_hash, None
            
            if current_hash != expected_hash:
                error_msg = (
                    f"Hash mismatch for {relative_path}! "
                    f"Expected: {expected_hash[:16]}..., "
                    f"Got: {current_hash[:16]}..."
                )
                logger.error(error_msg)
                return False, current_hash, error_msg
            
            return True, current_hash, None
            
        except Exception as e:
            error_msg = f"Error calculating hash for {relative_path}: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def validate_all(
        self,
        file_registry: Dict[str, Dict]
    ) -> Tuple[bool, List[str], Dict[str, str]]:
        """
        Validate all SQL files in registry.
        
        Args:
            file_registry: Registry of files with expected hashes
            
        Returns:
            Tuple of (all_valid, changed_files, updated_hashes)
        """
        changed_files = []
        updated_hashes = {}
        all_valid = True
        
        for query_name, query_info in file_registry.items():
            if query_info.get('path') is None:
                continue
            
            relative_path = query_info['path']
            expected_hash = query_info.get('sha256')
            
            is_valid, current_hash, error_msg = self.validate_file(
                relative_path,
                expected_hash
            )
            
            if not is_valid and expected_hash is not None:
                changed_files.append(query_name)
                all_valid = False
            
            if current_hash:
                updated_hashes[query_name] = {
                    'sha256': current_hash,
                    'last_validated': datetime.now().isoformat()
                }
        
        return all_valid, changed_files, updated_hashes
    
    def generate_hash_report(
        self,
        file_registry: Dict[str, Dict]
    ) -> str:
        """
        Generate a report of all SQL file hashes.
        
        Args:
            file_registry: Registry of files
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "SQL File Hash Report",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Project Root: {self.project_root}",
            "",
            "File Hashes:",
            "-" * 80
        ]
        
        for query_name, query_info in file_registry.items():
            if query_info.get('path') is None:
                continue
            
            path = query_info['path']
            full_path = self.project_root / path
            
            if full_path.exists():
                current_hash = self.calculate_hash(full_path)
                stored_hash = query_info.get('sha256', 'Not stored')
                status = "✓ Valid" if current_hash == stored_hash else "✗ Changed"
                
                report_lines.extend([
                    f"\nQuery: {query_name}",
                    f"  Path: {path}",
                    f"  Current Hash: {current_hash[:32]}...",
                    f"  Stored Hash:  {stored_hash[:32] if stored_hash != 'Not stored' else 'Not stored'}...",
                    f"  Status: {status}"
                ])
            else:
                report_lines.extend([
                    f"\nQuery: {query_name}",
                    f"  Path: {path}",
                    f"  Status: ✗ File not found"
                ])
        
        report_lines.extend(["", "-" * 80])
        return "\n".join(report_lines)