"""
Data feeds - composition layer for multi-source data.

DataFeeds compose multiple DataSources into a unified view
for strategy consumption. Implements:
- Multi-source joins (asof, range)
- Feature enrichment
- DataFeedProtocol for GenericRunner compatibility

"""

from .composable_feed import (
    ComposableDataFeed,
    JoinSpec,
    AsofJoin,
    EnricherSpec,
)
from .spread_data_feed import (
    SpreadDataFeed,
    SpreadDataFeedConfig,
    create_spread_feed,
)

__all__ = [
    "ComposableDataFeed",
    "JoinSpec",
    "AsofJoin",
    "EnricherSpec",
    "SpreadDataFeed",
    "SpreadDataFeedConfig",
    "create_spread_feed",
]
