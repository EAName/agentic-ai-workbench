"""
Behavior Tree Engine for Autonomous Agents.

Based on Ch 6: Building autonomous assistants.

Behavior trees provide structured decision-making for agents. They define
HOW an agent should approach complex, multi-step tasks with fallbacks,
conditions, and parallel execution.

Node Types:
  - Sequence: Run children in order; fail if any child fails
  - Selector (Fallback): Try children in order; succeed on first success
  - Condition: Check if a condition is met
  - Action: Execute an action (tool call, LLM query, etc.)
  - Parallel: Run children concurrently

Status: SUCCESS, FAILURE, RUNNING
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Abstract base class for all behavior tree nodes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        """Execute this node. Blackboard is shared state across the tree."""
        ...


class SequenceNode(BehaviorNode):
    """
    Runs children in order. Returns FAILURE on first child failure.
    Returns SUCCESS only if ALL children succeed.

    Use for: ordered steps that must all complete.
    Example: [fetch_data, analyze_data, write_report]
    """

    def __init__(self, name: str, children: list[BehaviorNode]):
        super().__init__(name)
        self.children = children

    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        for child in self.children:
            status = await child.tick(blackboard)
            if status == NodeStatus.FAILURE:
                logger.info("sequence_failed", node=self.name, failed_at=child.name)
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.SUCCESS


class SelectorNode(BehaviorNode):
    """
    Tries children in order. Returns SUCCESS on first child success.
    Returns FAILURE only if ALL children fail.

    Use for: fallback strategies.
    Example: [try_cache, try_database, try_api, return_default]
    """

    def __init__(self, name: str, children: list[BehaviorNode]):
        super().__init__(name)
        self.children = children

    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        for child in self.children:
            status = await child.tick(blackboard)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        logger.info("selector_exhausted", node=self.name)
        return NodeStatus.FAILURE


class ConditionNode(BehaviorNode):
    """
    Checks a condition against the blackboard.

    Use for: guards and preconditions.
    Example: check if data_available == True before processing.
    """

    def __init__(self, name: str, condition: Callable[[dict[str, Any]], bool]):
        super().__init__(name)
        self.condition = condition

    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        result = self.condition(blackboard)
        return NodeStatus.SUCCESS if result else NodeStatus.FAILURE


class ActionNode(BehaviorNode):
    """
    Executes an async action function.

    The action receives the blackboard and can read/write shared state.

    Use for: tool calls, LLM queries, data processing steps.
    """

    def __init__(
        self,
        name: str,
        action: Callable[[dict[str, Any]], Awaitable[NodeStatus]],
    ):
        super().__init__(name)
        self.action = action

    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        try:
            result = await self.action(blackboard)
            logger.info("action_executed", node=self.name, status=result.value)
            return result
        except Exception as e:
            logger.error("action_failed", node=self.name, error=str(e))
            return NodeStatus.FAILURE


class ParallelNode(BehaviorNode):
    """
    Runs all children concurrently. Configurable success/failure thresholds.

    Args:
        success_threshold: Number of children that must succeed.
        fail_threshold: Number of children that must fail to return FAILURE.
    """

    def __init__(
        self,
        name: str,
        children: list[BehaviorNode],
        success_threshold: int = 1,
    ):
        super().__init__(name)
        self.children = children
        self.success_threshold = success_threshold

    async def tick(self, blackboard: dict[str, Any]) -> NodeStatus:
        import asyncio

        results = await asyncio.gather(
            *[child.tick(blackboard) for child in self.children],
            return_exceptions=True,
        )

        successes = sum(
            1 for r in results
            if isinstance(r, NodeStatus) and r == NodeStatus.SUCCESS
        )

        if successes >= self.success_threshold:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class BehaviorTree:
    """
    The behavior tree runner.

    Usage:
        tree = BehaviorTree(root_node)
        blackboard = {"task": "analyze dataset", "data": [...]}
        status = await tree.run(blackboard)
    """

    def __init__(self, root: BehaviorNode):
        self.root = root

    async def run(self, blackboard: dict[str, Any] | None = None) -> NodeStatus:
        blackboard = blackboard or {}
        logger.info("behavior_tree_started", root=self.root.name)
        status = await self.root.tick(blackboard)
        logger.info("behavior_tree_completed", root=self.root.name, status=status.value)
        return status
