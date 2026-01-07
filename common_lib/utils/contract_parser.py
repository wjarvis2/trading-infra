"""
Contract parsing utilities - single source of truth for contract nomenclature.

This module provides centralized contract parsing following design principles:
- Single implementation for all contract parsing logic
- Handles continuous (CL1) and specific (CLF25) contracts
- Extensive edge case handling with clear error messages
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractSpec:
    """
    Parsed contract specification.
    
    Immutable to ensure thread safety and prevent accidental modifications.
    """
    root: str           # e.g., 'CL'
    position: int       # e.g., 1 for front month (continuous contracts)
    contract_code: str  # e.g., 'CL1', 'CLF25'
    month_code: Optional[str] = None  # e.g., 'F', 'Z' (specific contracts)
    year: Optional[int] = None        # e.g., 2025 (specific contracts)
    is_continuous: bool = True        # True for CL1, False for CLF25
    
    def __str__(self) -> str:
        return self.contract_code


class ContractParser:
    """
    Centralized contract parsing logic.
    
    Handles:
    - Continuous contracts: CL1, CL2, ..., CL99
    - Specific contracts: CLF25, CLZ26, etc.
    - Edge cases: double-digit positions, all month codes
    
    Examples
    --------
    >>> parser = ContractParser()
    >>> spec = parser.parse('CL1')
    >>> print(spec.root, spec.position)  # 'CL', 1
    >>> 
    >>> spec = parser.parse('CLZ25')
    >>> print(spec.month_code, spec.year)  # 'Z', 2025
    """
    
    # NYMEX month codes (F=Jan through Z=Dec, skipping I)
    MONTH_CODES = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    # Reverse mapping for month number to code
    MONTH_TO_CODE = {v: k for k, v in MONTH_CODES.items()}
    
    # Valid commodity roots we support
    VALID_ROOTS = {'CL', 'RB', 'HO', 'NG', 'BZ'}  # Expand as needed
    
    # Regex patterns for parsing
    CONTINUOUS_PATTERN = re.compile(r'^([A-Z]{2})(\d{1,2})$')  # CL1, CL12, CL99
    SPECIFIC_PATTERN = re.compile(r'^([A-Z]{2})([FGHJKMNQUVXZ])(\d{2})$')  # CLF25, CLZ26
    
    @classmethod
    def parse(cls, contract_code: str) -> ContractSpec:
        """
        Parse a contract code into its components.
        
        Parameters
        ----------
        contract_code : str
            Contract code to parse (e.g., 'CL1', 'CLF25')
            
        Returns
        -------
        ContractSpec
            Parsed contract specification
            
        Raises
        ------
        ValueError
            If contract code cannot be parsed or is invalid
        """
        if not contract_code:
            raise ValueError("Contract code cannot be empty")
        
        contract_code = contract_code.strip().upper()
        
        # Check again after stripping
        if not contract_code:
            raise ValueError("Contract code cannot be empty")
        
        # Try continuous contract first (most common)
        continuous_match = cls.CONTINUOUS_PATTERN.match(contract_code)
        if continuous_match:
            root = continuous_match.group(1)
            position = int(continuous_match.group(2))
            
            # Validate
            if root not in cls.VALID_ROOTS:
                raise ValueError(f"Unknown commodity root: {root}. Valid roots: {cls.VALID_ROOTS}")
            
            if position < 1 or position > 99:
                raise ValueError(f"Invalid position: {position}. Must be between 1 and 99")
            
            return ContractSpec(
                root=root,
                position=position,
                contract_code=contract_code,
                is_continuous=True
            )
        
        # Try specific contract
        specific_match = cls.SPECIFIC_PATTERN.match(contract_code)
        if specific_match:
            root = specific_match.group(1)
            month_code = specific_match.group(2)
            year_suffix = specific_match.group(3)
            
            # Validate
            if root not in cls.VALID_ROOTS:
                raise ValueError(f"Unknown commodity root: {root}. Valid roots: {cls.VALID_ROOTS}")
            
            if month_code not in cls.MONTH_CODES:
                raise ValueError(f"Invalid month code: {month_code}. Valid codes: {list(cls.MONTH_CODES.keys())}")
            
            # Convert 2-digit year to 4-digit
            year = 2000 + int(year_suffix)
            current_year = datetime.now().year
            
            # Handle century rollover (e.g., 99 could be 2099 not 1999)
            if year < current_year - 50:
                year += 100
            
            return ContractSpec(
                root=root,
                position=0,  # Not applicable for specific contracts
                contract_code=contract_code,
                month_code=month_code,
                year=year,
                is_continuous=False
            )
        
        # Unable to parse
        raise ValueError(
            f"Cannot parse contract code: '{contract_code}'. "
            f"Expected formats: continuous (CL1-CL99) or specific (CLF25, CLZ26)"
        )
    
    @classmethod
    def parse_multiple(cls, contract_codes: List[str]) -> List[ContractSpec]:
        """
        Parse multiple contract codes.
        
        Parameters
        ----------
        contract_codes : List[str]
            List of contract codes to parse
            
        Returns
        -------
        List[ContractSpec]
            List of parsed specifications
            
        Raises
        ------
        ValueError
            If any contract code cannot be parsed
        """
        specs = []
        errors = []
        
        for code in contract_codes:
            try:
                specs.append(cls.parse(code))
            except ValueError as e:
                errors.append(f"{code}: {str(e)}")
        
        if errors:
            raise ValueError(f"Failed to parse contracts:\n" + "\n".join(errors))
        
        return specs
    
    @classmethod
    def validate_spread(cls, contract1: str, contract2: str) -> Tuple[ContractSpec, ContractSpec]:
        """
        Validate and parse a spread pair.
        
        Parameters
        ----------
        contract1 : str
            First contract in spread
        contract2 : str
            Second contract in spread
            
        Returns
        -------
        Tuple[ContractSpec, ContractSpec]
            Parsed specifications for both contracts
            
        Raises
        ------
        ValueError
            If contracts are invalid for a spread
        """
        spec1 = cls.parse(contract1)
        spec2 = cls.parse(contract2)
        
        # Validate same root
        if spec1.root != spec2.root:
            raise ValueError(
                f"Spread contracts must have same root. "
                f"Got {spec1.root} and {spec2.root}"
            )
        
        # Validate both continuous or both specific
        if spec1.is_continuous != spec2.is_continuous:
            raise ValueError(
                f"Cannot mix continuous and specific contracts in spread. "
                f"Got {contract1} (continuous={spec1.is_continuous}) "
                f"and {contract2} (continuous={spec2.is_continuous})"
            )
        
        return spec1, spec2
    
    @classmethod
    def generate_continuous_list(cls, root: str, positions: Union[int, List[int]]) -> List[str]:
        """
        Generate list of continuous contract codes.
        
        Parameters
        ----------
        root : str
            Commodity root (e.g., 'CL')
        positions : int or List[int]
            Either max position (1-N) or specific positions
            
        Returns
        -------
        List[str]
            List of contract codes
            
        Examples
        --------
        >>> ContractParser.generate_continuous_list('CL', 8)
        ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8']
        >>> ContractParser.generate_continuous_list('CL', [1, 3, 6])
        ['CL1', 'CL3', 'CL6']
        """
        if root not in cls.VALID_ROOTS:
            raise ValueError(f"Unknown commodity root: {root}")
        
        if isinstance(positions, int):
            positions = list(range(1, positions + 1))
        
        return [f"{root}{pos}" for pos in positions]
    
    @classmethod
    def month_code_to_number(cls, month_code: str) -> int:
        """Convert month code to month number (1-12)."""
        if month_code not in cls.MONTH_CODES:
            raise ValueError(f"Invalid month code: {month_code}")
        return cls.MONTH_CODES[month_code]
    
    @classmethod
    def month_number_to_code(cls, month: int) -> str:
        """Convert month number (1-12) to month code."""
        if month not in cls.MONTH_TO_CODE:
            raise ValueError(f"Invalid month number: {month}. Must be 1-12")
        return cls.MONTH_TO_CODE[month]