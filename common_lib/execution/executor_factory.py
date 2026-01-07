"""
Executor Factory - Resolves instrument definitions to executors.

Entry point for creating spread executors based on InstrumentDefinition.
Routes to SyntheticSpreadExecutor or NativeSpreadExecutor as appropriate.

See research/decisions/003_execution_architecture.md for design rationale.

"""

from typing import Union

from common_lib.interfaces.instrument import (
    InstrumentDefinition,
    ExecutionMode,
)
from common_lib.execution.product_spec import ProductSpec, ProductCatalog
from common_lib.execution.executor_config import ExecutorConfig
from common_lib.execution.cost_models import (
    SlippageModel,
    CommissionModel,
    create_slippage_model,
    create_commission_model,
)
from common_lib.execution.spread_executor import (
    SyntheticSpreadExecutor,
    NativeSpreadExecutor,
    SpreadExecutorProtocol,
)


def resolve_executor(
    instrument_def: InstrumentDefinition,
    config: ExecutorConfig,
    product_catalog: ProductCatalog,
    slippage_model: SlippageModel = None,
    commission_model: CommissionModel = None,
) -> SpreadExecutorProtocol:
    """
    Resolve an InstrumentDefinition to the appropriate executor.

    This is the main entry point for creating spread executors.
    Routes based on execution_mode:
    - NATIVE → NativeSpreadExecutor
    - SYNTHETIC → SyntheticSpreadExecutor

    Parameters
    ----------
    instrument_def : InstrumentDefinition
        Definition of the instrument to trade
    config : ExecutorConfig
        Behavioral configuration
    product_catalog : ProductCatalog
        Registry of product specifications
    slippage_model : SlippageModel, optional
        Custom slippage model (if None, created from config)
    commission_model : CommissionModel, optional
        Custom commission model (if None, uses ProductSpec-based model)

    Returns
    -------
    SpreadExecutorProtocol
        Configured executor (Synthetic or Native)

    Raises
    ------
    KeyError
        If product not found in catalog

    Examples
    --------
    >>> from common_lib.interfaces.instrument import create_calendar_spread, ExecutionMode
    >>> from common_lib.execution.product_spec import create_default_catalog
    >>> from common_lib.execution.executor_config import DEFAULT_CONFIG
    >>>
    >>> instrument = create_calendar_spread("CL_FEB25", "CL_MAR25")
    >>> catalog = create_default_catalog()
    >>> executor = resolve_executor(instrument, DEFAULT_CONFIG, catalog)
    """
    # Extract product root from instrument_id (e.g., "CL_FEB25_MAR25" -> "CL")
    root = instrument_def.instrument_id.split("_")[0]

    # Get product spec from catalog
    product = product_catalog.get(root)

    # Create cost models if not provided
    if slippage_model is None:
        slippage_model = create_slippage_model(config)
    if commission_model is None:
        commission_model = create_commission_model(use_product_spec=True)

    # Route to appropriate executor
    if instrument_def.execution_mode == ExecutionMode.NATIVE:
        return NativeSpreadExecutor(
            product=product,
            config=config,
            slippage_model=slippage_model,
            commission_model=commission_model,
        )
    else:  # SYNTHETIC
        return SyntheticSpreadExecutor(
            product=product,
            config=config,
            slippage_model=slippage_model,
            commission_model=commission_model,
        )


def create_spread_executor(
    root: str,
    config: ExecutorConfig = None,
    product_catalog: ProductCatalog = None,
    execution_mode: ExecutionMode = ExecutionMode.SYNTHETIC,
) -> SpreadExecutorProtocol:
    """
    Simplified factory for spread executors.

    Use this when you just need an executor without a full InstrumentDefinition.

    Parameters
    ----------
    root : str
        Product root symbol (e.g., "CL")
    config : ExecutorConfig, optional
        Behavioral config (defaults to DEFAULT_CONFIG)
    product_catalog : ProductCatalog, optional
        Product catalog (defaults to standard catalog)
    execution_mode : ExecutionMode
        NATIVE or SYNTHETIC (default SYNTHETIC)

    Returns
    -------
    SpreadExecutorProtocol
        Configured executor

    Examples
    --------
    >>> executor = create_spread_executor("CL")
    >>> fill = executor.execute_entry(-1, bar, "MR")
    """
    from common_lib.execution.executor_config import DEFAULT_CONFIG
    from common_lib.execution.product_spec import create_default_catalog

    if config is None:
        config = DEFAULT_CONFIG
    if product_catalog is None:
        product_catalog = create_default_catalog()

    product = product_catalog.get(root)
    slippage_model = create_slippage_model(config)
    commission_model = create_commission_model(use_product_spec=True)

    if execution_mode == ExecutionMode.NATIVE:
        return NativeSpreadExecutor(
            product=product,
            config=config,
            slippage_model=slippage_model,
            commission_model=commission_model,
        )
    else:
        return SyntheticSpreadExecutor(
            product=product,
            config=config,
            slippage_model=slippage_model,
            commission_model=commission_model,
        )
