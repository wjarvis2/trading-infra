"""
Subscription Manager for IBKR market data feeds.

Enforces hard cap of 200 simultaneous subscriptions and provides
atomic swap operations for delta-targeted options rebalancing.

Key principles:
- Fail closed: refuse to exceed cap
- Subscribe once, hold: minimize churn to avoid pacing violations
- Atomic swaps: 1-for-1 replacement keeps count unchanged
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Callable

from ib_insync import IB, Contract, Ticker

logger = logging.getLogger(__name__)


class ContractType(Enum):
    FUTURE = "FUTURE"
    OPTION = "OPTION"


@dataclass
class Subscription:
    """Tracks an active market data subscription."""
    contract: Contract
    contract_type: ContractType
    ticker: Ticker
    subscribed_at: datetime = field(default_factory=datetime.now)
    reason: str = ""  # Why this subscription exists (e.g., "delta_ladder:CL:25D:C")


class SubscriptionManager:
    """
    Manages IBKR market data subscriptions with hard cap enforcement.

    Usage:
        manager = SubscriptionManager(ib)

        # Subscribe to a contract
        if manager.subscribe(cl_future, ContractType.FUTURE, reason="front_month"):
            print("Subscribed!")
        else:
            print("At cap, cannot subscribe")

        # Atomic swap for rebalancing
        if manager.swap(old_option, new_option, ContractType.OPTION, reason="drift_rebalance"):
            print("Swapped!")
    """

    MAX_SUBSCRIPTIONS = 200

    def __init__(
        self,
        ib: IB,
        audit_callback: Optional[Callable[[str, ContractType, Contract, str, int], None]] = None
    ):
        """
        Initialize subscription manager.

        Args:
            ib: Connected IB instance
            audit_callback: Optional callback for audit logging
                           (action, contract_type, contract, reason, count_after)
        """
        self.ib = ib
        self.active: Dict[int, Subscription] = {}  # conId -> Subscription
        self.audit_callback = audit_callback

    def count(self) -> int:
        """Current number of active subscriptions."""
        return len(self.active)

    def headroom(self) -> int:
        """Number of subscriptions available before hitting cap."""
        return self.MAX_SUBSCRIPTIONS - self.count()

    def is_subscribed(self, contract: Contract) -> bool:
        """Check if a contract is currently subscribed."""
        return contract.conId in self.active

    def get_ticker(self, contract: Contract) -> Optional[Ticker]:
        """Get the ticker for a subscribed contract."""
        sub = self.active.get(contract.conId)
        return sub.ticker if sub else None

    def subscribe(
        self,
        contract: Contract,
        contract_type: ContractType,
        reason: str = ""
    ) -> bool:
        """
        Subscribe to market data for a contract.

        Args:
            contract: IBKR contract (must have conId populated)
            contract_type: FUTURE or OPTION
            reason: Why subscribing (for audit trail)

        Returns:
            True if subscribed successfully, False if at cap or already subscribed
        """
        if not contract.conId:
            logger.error(f"Cannot subscribe: contract has no conId: {contract}")
            return False

        # Already subscribed?
        if contract.conId in self.active:
            logger.debug(f"Already subscribed to {contract.localSymbol} (conId={contract.conId})")
            return True

        # At cap?
        if self.count() >= self.MAX_SUBSCRIPTIONS:
            logger.warning(
                f"SUBSCRIPTION CAP REACHED ({self.MAX_SUBSCRIPTIONS}). "
                f"Cannot subscribe to {contract.localSymbol}"
            )
            return False

        # Subscribe
        ticker = self.ib.reqMktData(contract, genericTickList="", snapshot=False)

        self.active[contract.conId] = Subscription(
            contract=contract,
            contract_type=contract_type,
            ticker=ticker,
            reason=reason
        )

        logger.info(
            f"SUBSCRIBED: {contract.localSymbol} (conId={contract.conId}) "
            f"[{contract_type.value}] reason={reason} "
            f"[{self.count()}/{self.MAX_SUBSCRIPTIONS}]"
        )

        self._audit("SUBSCRIBE", contract_type, contract, reason)
        return True

    def unsubscribe(self, contract: Contract) -> bool:
        """
        Cancel market data subscription for a contract.

        Args:
            contract: Contract to unsubscribe

        Returns:
            True if unsubscribed, False if wasn't subscribed
        """
        if contract.conId not in self.active:
            logger.debug(f"Not subscribed to {contract.localSymbol}, nothing to cancel")
            return False

        sub = self.active[contract.conId]

        # Cancel the subscription
        self.ib.cancelMktData(sub.ticker.contract)
        del self.active[contract.conId]

        logger.info(
            f"UNSUBSCRIBED: {contract.localSymbol} (conId={contract.conId}) "
            f"[{sub.contract_type.value}] "
            f"[{self.count()}/{self.MAX_SUBSCRIPTIONS}]"
        )

        self._audit("UNSUBSCRIBE", sub.contract_type, contract, sub.reason)
        return True

    def swap(
        self,
        old_contract: Contract,
        new_contract: Contract,
        contract_type: ContractType,
        reason: str = ""
    ) -> bool:
        """
        Atomic 1-for-1 swap of subscriptions.

        Unsubscribes from old_contract and subscribes to new_contract.
        Total subscription count remains unchanged.

        Args:
            old_contract: Contract to unsubscribe
            new_contract: Contract to subscribe
            contract_type: Type of both contracts
            reason: Why swapping (for audit trail)

        Returns:
            True if swap succeeded, False otherwise
        """
        if old_contract.conId == new_contract.conId:
            logger.debug("Swap called with same contract, no-op")
            return True

        if old_contract.conId not in self.active:
            logger.warning(
                f"Cannot swap: old contract {old_contract.localSymbol} not subscribed"
            )
            return False

        if new_contract.conId in self.active:
            # New contract already subscribed, just unsubscribe old
            logger.info(
                f"New contract {new_contract.localSymbol} already subscribed, "
                f"just unsubscribing {old_contract.localSymbol}"
            )
            return self.unsubscribe(old_contract)

        # Get old subscription info
        old_sub = self.active[old_contract.conId]

        # Subscribe to new first (while we have the slot)
        new_ticker = self.ib.reqMktData(new_contract, genericTickList="", snapshot=False)

        # Cancel old
        self.ib.cancelMktData(old_sub.ticker.contract)
        del self.active[old_contract.conId]

        # Add new
        self.active[new_contract.conId] = Subscription(
            contract=new_contract,
            contract_type=contract_type,
            ticker=new_ticker,
            reason=reason
        )

        logger.info(
            f"SWAPPED: {old_contract.localSymbol} -> {new_contract.localSymbol} "
            f"[{contract_type.value}] reason={reason} "
            f"[{self.count()}/{self.MAX_SUBSCRIPTIONS}]"
        )

        self._audit("SWAP", contract_type, new_contract, reason)
        return True

    def bulk_subscribe(
        self,
        contracts: list[Contract],
        contract_type: ContractType,
        reason: str = ""
    ) -> tuple[int, int]:
        """
        Subscribe to multiple contracts, stopping at cap.

        Args:
            contracts: List of contracts to subscribe
            contract_type: Type of all contracts
            reason: Why subscribing

        Returns:
            (succeeded, failed) count tuple
        """
        succeeded = 0
        failed = 0

        for contract in contracts:
            if self.subscribe(contract, contract_type, reason):
                succeeded += 1
            else:
                failed += 1

        return succeeded, failed

    def unsubscribe_all(self) -> int:
        """
        Cancel all subscriptions. Used for cleanup.

        Returns:
            Number of subscriptions cancelled
        """
        count = self.count()

        for sub in list(self.active.values()):
            self.ib.cancelMktData(sub.ticker.contract)

        self.active.clear()
        logger.info(f"Unsubscribed all ({count} subscriptions)")

        return count

    def get_subscriptions_by_type(self, contract_type: ContractType) -> list[Subscription]:
        """Get all subscriptions of a specific type."""
        return [s for s in self.active.values() if s.contract_type == contract_type]

    def get_subscriptions_by_reason(self, reason_prefix: str) -> list[Subscription]:
        """Get all subscriptions with reasons starting with prefix."""
        return [s for s in self.active.values() if s.reason.startswith(reason_prefix)]

    def status(self) -> dict:
        """Get subscription status summary."""
        futures = len(self.get_subscriptions_by_type(ContractType.FUTURE))
        options = len(self.get_subscriptions_by_type(ContractType.OPTION))

        return {
            "total": self.count(),
            "max": self.MAX_SUBSCRIPTIONS,
            "headroom": self.headroom(),
            "futures": futures,
            "options": options,
            "utilization_pct": round(100 * self.count() / self.MAX_SUBSCRIPTIONS, 1)
        }

    def _audit(
        self,
        action: str,
        contract_type: ContractType,
        contract: Contract,
        reason: str
    ):
        """Record action to audit trail."""
        if self.audit_callback:
            try:
                self.audit_callback(
                    action,
                    contract_type,
                    contract,
                    reason,
                    self.count()
                )
            except Exception as e:
                logger.error(f"Audit callback failed: {e}")
