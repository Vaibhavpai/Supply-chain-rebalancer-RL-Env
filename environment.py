"""
environment.py — Static data + SupplyChainEnv skeleton.
OpenEnv-compliant Supply Chain Inventory Rebalancer.

Dev 2 fills:  step(), _compute_reward()
Dev 3 fills:  grade_task_1(), grade_task_2(), grade_task_3()
"""

from __future__ import annotations

import copy
import random
from typing import Any

from models import (
    Action,
    DemandForecast,
    IncomingShipment,
    Observation,
    StepResult,
    Transfer,
    WarehouseState,
    ProcessSupervisionReward,
)

# ---------------------------------------------------------------------------
# Static topology constants
# ---------------------------------------------------------------------------

NODES: list[str] = ["east", "central", "west", "supplier_1", "supplier_2"]
WAREHOUSES: list[str] = ["east", "central", "west"]
SKUS: list[str] = ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]

SHIPPING_COSTS: dict[str, dict[str, float]] = {
    "east":       {"central": 2.0, "west": 3.5},
    "central":    {"east": 2.0,    "west": 1.5},
    "west":       {"east": 3.5,    "central": 1.5},
    "supplier_1": {"east": 5.0,    "central": 5.0, "west": 5.0},
    "supplier_2": {"east": 6.0,    "central": 6.0, "west": 6.0},
}

TRANSIT_TIMES: dict[str, dict[str, int]] = {
    "east":       {"central": 1, "west": 2},
    "central":    {"east": 1,    "west": 1},
    "west":       {"east": 2,    "central": 1},
    "supplier_1": {"east": 2,    "central": 2, "west": 2},
    "supplier_2": {"east": 2,    "central": 2, "west": 2},
}

HOLDING_COSTS: dict[str, float] = {
    "east":    0.10,
    "central": 0.08,
    "west":    0.10,
}

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_TASK_CATALOGUE: dict[int, dict[str, Any]] = {
    1: {
        # Task 1: Simple Restock — West has 0 SKU_A, Central has 500, demand=50 at West
        "inventory": {
            "east":    {"SKU_A": 100, "SKU_B": 100, "SKU_C": 100, "SKU_D": 100, "SKU_E": 100},
            "central": {"SKU_A": 500, "SKU_B": 100, "SKU_C": 100, "SKU_D": 100, "SKU_E": 100},
            "west":    {"SKU_A": 0,   "SKU_B": 100, "SKU_C": 100, "SKU_D": 100, "SKU_E": 100},
        },
        "demand_pattern": {
            "east":    {"SKU_A": 10, "SKU_B": 10, "SKU_C": 10, "SKU_D": 10, "SKU_E": 10},
            "central": {"SKU_A": 10, "SKU_B": 10, "SKU_C": 10, "SKU_D": 10, "SKU_E": 10},
            "west":    {"SKU_A": 50, "SKU_B": 10, "SKU_C": 10, "SKU_D": 10, "SKU_E": 10},
        },
        "horizon": 5,
        "alerts": [],
    },
    2: {
        # Task 2: Cost-Optimized Rebalancing — mismatched inventory, 5-day episode
        "inventory": {
            "east":    {"SKU_A": 40,  "SKU_B": 30,  "SKU_C": 20,  "SKU_D": 15,  "SKU_E": 10},
            "central": {"SKU_A": 300, "SKU_B": 250, "SKU_C": 200, "SKU_D": 150, "SKU_E": 120},
            "west":    {"SKU_A": 180, "SKU_B": 160, "SKU_C": 140, "SKU_D": 130, "SKU_E": 100},
        },
        "demand_pattern": {
            "east":    {"SKU_A": 40, "SKU_B": 30, "SKU_C": 15, "SKU_D": 8,  "SKU_E": 12},
            "central": {"SKU_A": 10, "SKU_B": 15, "SKU_C": 10, "SKU_D": 8,  "SKU_E": 6},
            "west":    {"SKU_A": 12, "SKU_B": 10, "SKU_C": 20, "SKU_D": 30, "SKU_E": 18},
        },
        "horizon": 5,
        "alerts": [],
    },
    3: {
        # Task 3: Supply Shock — steady 7-day sim, shock injected on Day 3
        "inventory": {
            "east":    {"SKU_A": 150, "SKU_B": 150, "SKU_C": 150, "SKU_D": 150, "SKU_E": 150},
            "central": {"SKU_A": 150, "SKU_B": 150, "SKU_C": 150, "SKU_D": 150, "SKU_E": 150},
            "west":    {"SKU_A": 150, "SKU_B": 150, "SKU_C": 150, "SKU_D": 150, "SKU_E": 150},
        },
        "demand_pattern": {
            "east":    {"SKU_A": 25, "SKU_B": 25, "SKU_C": 25, "SKU_D": 25, "SKU_E": 25},
            "central": {"SKU_A": 25, "SKU_B": 25, "SKU_C": 25, "SKU_D": 25, "SKU_E": 25},
            "west":    {"SKU_A": 25, "SKU_B": 25, "SKU_C": 25, "SKU_D": 25, "SKU_E": 25},
        },
        "horizon": 7,
        "alerts": [],
    },
}

_FORECAST_HORIZON: int = 3


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnv:
    """
    Skeleton environment.  Dev 2 implements step(); Dev 3 implements graders.

    _internal_state keys — see docs/state_contract.md for full contract.
    """

    def __init__(self, task_id: int = 1, seed: int = 42) -> None:
        if task_id not in _TASK_CATALOGUE:
            raise ValueError(
                f"Unknown task_id={task_id}. Available: {sorted(_TASK_CATALOGUE)}"
            )
        self._task_id = task_id
        self._seed = seed

        # Declare ALL keys up-front so the contract is visible before reset()
        self._internal_state: dict[str, Any] = {
            "day":              None,   # int
            "inventory":        None,   # dict[warehouse_id, dict[sku, int]]
            "in_transit":       None,   # list[dict]
            "demand_log":       None,   # list[dict]
            "cost_accumulator": None,   # dict[str, float]
            "alerts":           None,   # list[str]
            "horizon":          None,   # int
            "rng":              None,   # random.Random
            "demand_pattern":   None,   # dict[warehouse_id, dict[sku, int]]
        }

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Wipe all state, load task-specific initial conditions, return fresh Observation."""
        task = _TASK_CATALOGUE[self._task_id]
        self._internal_state = {
            "day":              0,
            "inventory":        copy.deepcopy(task["inventory"]),
            "in_transit":       [],
            "demand_log":       [],
            "cost_accumulator": {"holding": 0.0, "shipping": 0.0, "unfulfilled": 0.0},
            "alerts":           list(task["alerts"]),
            "horizon":          task["horizon"],
            "rng":              random.Random(self._seed),
            "demand_pattern":   copy.deepcopy(task["demand_pattern"]),
        }
        return self._get_observation()

    def state(self) -> Observation:
        """Return current Observation without mutating any state."""
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one action, process transfers and optionally advance the simulation
        by one day.

        Processing order (mirrors docs/state_contract.md):
          1. Validate action → on violations apply penalty, return early.
          2. Dispatch transfers → deduct inventory, push to in_transit.
          3. If advance_time=False → compute reward without advancing day.
          4. Arrive shipments (arrives_on_day == current_day).
          5. Realise demand via rng, fulfil, log shortfalls.
          6. Charge holding costs.
          7. _compute_reward()
          8. day += 1
          9. done = day >= horizon
         10. return StepResult
        """
        s = self._internal_state
        prev_inventory = copy.deepcopy(s["inventory"])

        # ------------------------------------------------------------------ #
        # 1. Validate action
        # ------------------------------------------------------------------ #
        violations = self._validate_action(action)
        if violations:
            reward = self._compute_reward(
                action=action,
                shipping_cost=0.0,
                fulfilled_units=0,
                stockout_units=0,
                holding_cost=0.0,
                is_invalid=True,
            )
            return StepResult(
                observation=self._get_observation(),
                reward=reward,
                done=s["day"] >= s["horizon"],
                info={"violations": violations},
            )

        # ------------------------------------------------------------------ #
        # 2. Dispatch transfers
        # ------------------------------------------------------------------ #
        shipping_cost = 0.0
        for t in action.transfers:
            s["inventory"][t.source_node][t.sku] -= t.quantity
            route_cost = SHIPPING_COSTS.get(t.source_node, {}).get(t.destination_node, 0.0)
            shipping_cost += t.quantity * route_cost
            transit_days = TRANSIT_TIMES.get(t.source_node, {}).get(t.destination_node, 1)
            s["in_transit"].append({
                "source_node":      t.source_node,
                "destination_node": t.destination_node,
                "sku":              t.sku,
                "quantity":         t.quantity,
                "arrives_on_day":   s["day"] + transit_days,
            })

        # Track task-1 specific metric
        if self._task_id == 1:
            for t in action.transfers:
                if t.source_node == "central" and t.destination_node == "west" and t.sku == "SKU_A":
                    prev = s.get("task1_transferred", 0)
                    s["task1_transferred"] = prev + t.quantity

        s["cost_accumulator"]["shipping"] += shipping_cost

        # ------------------------------------------------------------------ #
        # 3. Short-circuit if not advancing time
        # ------------------------------------------------------------------ #
        if not action.advance_time:
            reward = self._compute_reward(
                action=action,
                shipping_cost=shipping_cost,
                fulfilled_units=0,
                stockout_units=0,
                holding_cost=0.0,
                is_invalid=False,
            )
            return StepResult(
                observation=self._get_observation(),
                reward=reward,
                done=s["day"] >= s["horizon"],
                info={"shipping_cost": shipping_cost},
            )

        # ------------------------------------------------------------------ #
        # 4. Arrive shipments (arrives_on_day == current_day)
        # ------------------------------------------------------------------ #
        # Task 3: on day 3 inject supply shock and freeze supplier_1 shipments
        if self._task_id == 3 and s["day"] == 3:
            alert = "Supplier 1 offline — shipments delayed indefinitely"
            if alert not in s["alerts"]:
                s["alerts"].append(alert)
            # Remove all in-transit supplier_1 shipments
            s["in_transit"] = [
                sh for sh in s["in_transit"]
                if sh["source_node"] != "supplier_1"
            ]

        still_in_transit = []
        for shipment in s["in_transit"]:
            if shipment["arrives_on_day"] == s["day"]:
                dst = shipment["destination_node"]
                sku = shipment["sku"]
                if dst in s["inventory"]:
                    s["inventory"][dst][sku] = s["inventory"][dst].get(sku, 0) + shipment["quantity"]
                # else: supplier→supplier routes are dropped silently
            else:
                still_in_transit.append(shipment)
        s["in_transit"] = still_in_transit

        # ------------------------------------------------------------------ #
        # 5. Realise demand via rng, fulfil, log shortfalls
        # ------------------------------------------------------------------ #
        fulfilled_units = 0
        stockout_units = 0

        for wh_id in WAREHOUSES:
            pattern = s["demand_pattern"][wh_id]
            for sku in SKUS:
                base_qty = pattern.get(sku, 0)
                jitter = 1.0 + 0.2 * (s["rng"].random() * 2 - 1)
                demanded = max(0, round(base_qty * jitter))

                available = s["inventory"][wh_id].get(sku, 0)
                fulfilled = min(demanded, available)
                shortfall = demanded - fulfilled

                s["inventory"][wh_id][sku] = available - fulfilled  # clamp handled by min()
                fulfilled_units += fulfilled
                stockout_units += shortfall

                s["demand_log"].append({
                    "day":          s["day"],
                    "warehouse_id": wh_id,
                    "sku":          sku,
                    "demanded":     demanded,
                    "fulfilled":    fulfilled,
                    "shortfall":    shortfall,
                })

        # ------------------------------------------------------------------ #
        # 6. Charge holding costs
        # ------------------------------------------------------------------ #
        holding_cost = 0.0
        for wh_id in WAREHOUSES:
            rate = HOLDING_COSTS.get(wh_id, 0.0)
            for sku in SKUS:
                holding_cost += s["inventory"][wh_id].get(sku, 0) * rate
        s["cost_accumulator"]["holding"] += holding_cost
        s["cost_accumulator"]["unfulfilled"] += stockout_units

        # ------------------------------------------------------------------ #
        # 7. _compute_reward()
        # ------------------------------------------------------------------ #
        reward = self._compute_reward(
            action=action,
            shipping_cost=shipping_cost,
            fulfilled_units=fulfilled_units,
            stockout_units=stockout_units,
            holding_cost=holding_cost,
            is_invalid=False,
        )

        # ------------------------------------------------------------------ #
        # 8. day += 1
        # ------------------------------------------------------------------ #
        s["day"] += 1

        # ------------------------------------------------------------------ #
        # 9. done = day >= horizon
        # ------------------------------------------------------------------ #
        done = s["day"] >= s["horizon"]

        # ------------------------------------------------------------------ #
        # 10. return StepResult
        # ------------------------------------------------------------------ #
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={
                "shipping_cost":   shipping_cost,
                "holding_cost":    holding_cost,
                "fulfilled_units": fulfilled_units,
                "stockout_units":  stockout_units,
            },
        )

    # ------------------------------------------------------------------
    # Reward computation (Process Supervision Enhanced)
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: Action,
        shipping_cost: float,
        fulfilled_units: int,
        stockout_units: int,
        holding_cost: float,
        is_invalid: bool,
    ) -> "ProcessSupervisionReward":
        """
        Enhanced process-supervision reward with per-step intermediate signals.

        Rewards multiple dimensions:
          1. Fulfillment: +2.0 per unit fulfilled, -5.0 per stockout
          2. Costs: -1.0 × (holding + shipping) costs
          3. Inventory balance: +0.5 if transfers are balancing, -0.5 if exacerbating imbalance
          4. Forecast alignment: +1.0 × alignment score (0.0-1.0) with demand forecast
          5. Proactive prevention: +2.0 if action prevents a forecast stockout
          6. Safety margin: +0.5 if maintaining 5-day safety buffer
          7. Temporal incentive: bonus for early intervention (higher on earlier days)
          8. Penalties: -50.0 for invalid actions, -10.0 for safety violations
        """
        from models import ProcessSupervisionReward

        s = self._internal_state

        # ────────────────────────────────────────────────────────────────
        # 1. Core fulfillment & cost signals
        # ────────────────────────────────────────────────────────────────
        fulfilled_reward = 2.0 * fulfilled_units
        stockout_penalty = -5.0 * stockout_units
        holding_penalty = -1.0 * holding_cost
        shipping_penalty = -1.0 * shipping_cost
        invalid_penalty = -50.0 if is_invalid else 0.0

        # ────────────────────────────────────────────────────────────────
        # 2. Inventory Balance Score (are transfers from excess to deficit?)
        # ────────────────────────────────────────────────────────────────
        balance_score = self._compute_inventory_balance_score(action)
        balance_reward = 0.5 * balance_score  # ∈ [-0.5, +0.5]

        # ────────────────────────────────────────────────────────────────
        # 3. Demand Forecast Alignment
        # ────────────────────────────────────────────────────────────────
        forecast_alignment = self._compute_forecast_alignment(action)
        forecast_reward = 1.0 * forecast_alignment  # ∈ [0.0, 1.0]

        # ────────────────────────────────────────────────────────────────
        # 4. Proactive Stockout Prevention
        # ────────────────────────────────────────────────────────────────
        prevention_score = self._compute_stockout_prevention_score(action)
        prevention_bonus = 2.0 * prevention_score  # ∈ [0.0, 2.0]

        # ────────────────────────────────────────────────────────────────
        # 5. Safety Margin Bonus (maintaining buffer stock)
        # ────────────────────────────────────────────────────────────────
        safety_margin = self._compute_safety_margin()
        safety_bonus = 0.5 if safety_margin > 0.2 else 0.0  # 5-day buffer minimum

        # ────────────────────────────────────────────────────────────────
        # 6. Temporal Incentive Bonus (reward early intervention)
        # ────────────────────────────────────────────────────────────────
        horizon = s["horizon"]
        days_remaining = horizon - s["day"]
        temporal_bonus = max(0.0, (days_remaining / max(1, horizon)) * 1.0)

        # ────────────────────────────────────────────────────────────────
        # 7. Safety Violation Penalty (routing errors, impossible actions)
        # ────────────────────────────────────────────────────────────────
        safety_violation_penalty = 0.0
        if not is_invalid:  # only check if action passed validation
            safety_violation_penalty = self._compute_safety_violation_penalty(action)

        # ────────────────────────────────────────────────────────────────
        # Composite Score
        # ────────────────────────────────────────────────────────────────
        total = (
            fulfilled_reward
            + stockout_penalty
            + holding_penalty
            + shipping_penalty
            + invalid_penalty
            + balance_reward
            + forecast_reward
            + prevention_bonus
            + safety_bonus
            + temporal_bonus
            + safety_violation_penalty
        )

        return ProcessSupervisionReward(
            # Base metrics
            fulfilled_demand_reward=fulfilled_reward,
            stockout_penalty=stockout_penalty,
            holding_cost_penalty=holding_penalty,
            shipping_cost_penalty=shipping_penalty,
            # Process supervision metrics
            inventory_balance_reward=balance_reward,
            demand_forecast_accuracy_reward=forecast_reward,
            proactive_restocking_bonus=prevention_bonus,
            safety_margin_bonus=safety_bonus,
            temporal_incentive_bonus=temporal_bonus,
            # Penalties
            invalid_action_penalty=invalid_penalty,
            safety_violation_penalty=safety_violation_penalty,
            # Composite
            total=total,
        )

    def _compute_inventory_balance_score(self, action: Action) -> float:
        """
        Score ∈ [-1.0, +1.0] based on whether transfers reduce imbalance.
        
        +1.0 if transfers are from high-inventory to low-inventory warehouses.
        -1.0 if transfers exacerbate imbalance (high stocks getting higher).
         0.0 if transfers have no effect on balance.
        """
        if not action.transfers:
            return 0.0

        s = self._internal_state
        inv = s["inventory"]

        score = 0.0
        for t in action.transfers:
            if t.source_node not in WAREHOUSES or t.destination_node not in WAREHOUSES:
                continue

            src_inv = inv[t.source_node].get(t.sku, 0)
            dst_inv = inv[t.destination_node].get(t.sku, 0)

            # If src has much more than dst, this is favorable
            imbalance_before = (src_inv - dst_inv) / max(1, src_inv + dst_inv)
            imbalance_after = (
                (src_inv - t.quantity) - (dst_inv + t.quantity)
            ) / max(1, src_inv + t.quantity + dst_inv + t.quantity)

            # Reward for reducing imbalance
            improvement = imbalance_before - imbalance_after
            score += min(1.0, max(-1.0, improvement))

        return min(1.0, max(-1.0, score / len(action.transfers)))

    def _compute_forecast_alignment(self, action: Action) -> float:
        """
        Score ∈ [0.0, 1.0] measuring how well transfers align with demand forecast.
        
        High score if transfers move stock toward high-demand warehouses.
        """
        if not action.transfers:
            return 0.5  # neutral if no action

        s = self._internal_state
        day = s["day"]
        forecast = s["demand_pattern"]  # next-day demand pattern

        score = 0.0
        transfer_count = 0

        for t in action.transfers:
            if t.destination_node not in WAREHOUSES:
                continue

            # Get forecasted demand at destination in next 3 days
            dst_forecast = forecast.get(t.destination_node, {}).get(t.sku, 1)
            # Get current stock at destination
            current_stock = s["inventory"][t.destination_node].get(t.sku, 0)

            # Check if this transfer helps meet forecast demand
            stock_after_transfer = current_stock + t.quantity
            demand_ratio = min(1.0, current_stock / max(1, dst_forecast))
            demand_ratio_after = min(1.0, stock_after_transfer / max(1, dst_forecast))

            # Reward if we're bringing stock ratio closer to 1.0
            alignment = max(0.0, min(1.0, 1.0 - abs(demand_ratio_after - 1.0)))
            score += alignment
            transfer_count += 1

        return score / transfer_count if transfer_count > 0 else 0.5

    def _compute_stockout_prevention_score(self, action: Action) -> float:
        """
        Score ∈ [0.0, 1.0] measuring how well transfers prevent future stockouts.
        
        High score if transfers provide stock to warehouses with low forecasted availability.
        """
        if not action.transfers:
            return 0.0

        s = self._internal_state
        forecast = s["demand_pattern"]
        inv = s["inventory"]

        prevention_score = 0.0
        transfer_count = 0

        for t in action.transfers:
            if t.destination_node not in WAREHOUSES:
                continue

            dst_inv = inv[t.destination_node].get(t.sku, 0)
            dst_forecast = forecast.get(t.destination_node, {}).get(t.sku, 1)

            # If destination would run out without this transfer
            if dst_inv < dst_forecast:
                # Score: how much does transfer help?
                new_inv = dst_inv + t.quantity
                if new_inv >= dst_forecast:
                    prevention_score += 1.0  # Fully prevents stockout
                else:
                    prevention_score += min(1.0, new_inv / max(1, dst_forecast))

            transfer_count += 1

        return prevention_score / transfer_count if transfer_count > 0 else 0.0

    def _compute_safety_margin(self) -> float:
        """
        Ratio ∈ [0.0, ∞) measuring inventory adequacy.
        
        returns: total_inventory / (avg_daily_demand × forecast_horizon)
        
        > 1.0 is healthy (excess inventory)
        ~ 0.2  is 5-day safety buffer
        < 0.1  is danger zone
        """
        s = self._internal_state
        inv = s["inventory"]
        demand = s["demand_pattern"]

        total_inv = sum(
            qty for wh in inv.values() for qty in wh.values()
        )

        avg_daily_demand = sum(
            qty for wh in demand.values() for qty in wh.values()
        ) / 3.0  # 3 days average

        if avg_daily_demand == 0:
            return 5.0

        return total_inv / (avg_daily_demand * _FORECAST_HORIZON)

    def _compute_safety_violation_penalty(self, action: Action) -> float:
        """
        Penalty ∈ [-10.0, 0.0] for risky decisions (even if technically valid).
        
        Examples:
          - Transferring >50% of warehouse stock in one day (depleting)
          - Routing to warehouse already at capacity
          - Empty stock lines with upcoming demand
        """
        if not action.transfers:
            return 0.0

        s = self._internal_state
        inv = s["inventory"]
        forecast = s["demand_pattern"]
        penalty = 0.0

        for t in action.transfers:
            if t.source_node not in WAREHOUSES:
                continue

            # Check for dangerous depletion
            current_stock = inv[t.source_node].get(t.sku, 0)
            if t.quantity > 0.5 * current_stock and current_stock > 0:
                penalty -= 2.0  # risky depletion

            # Check if destination will be over-provisioned (waste)
            if t.destination_node in WAREHOUSES:
                dst_stock = inv[t.destination_node].get(t.sku, 0)
                dst_forecast = forecast.get(t.destination_node, {}).get(t.sku, 1)
                if dst_stock + t.quantity > 3 * dst_forecast:
                    penalty -= 1.0  # excessive buffer

        return max(-10.0, penalty)

    # ------------------------------------------------------------------
    # Grader stubs  (Dev 3)
    # ------------------------------------------------------------------

    def grade_task_1(self) -> dict[str, Any]:
        """
        Easy: Simple Restock.

        Score 1.0 if the agent transferred >= 50 units of SKU_A from Central
        to West before calling advance_time=True on Day 1 (tracked via
        _internal_state["task1_transferred"]).
        Score 0.0 otherwise.

        Returns
        -------
        dict with keys: "score" (float in [0.0, 1.0]), "transferred" (int).
        """
        s = self._internal_state
        transferred: int = s.get("task1_transferred", 0)
        score: float = 1.0 if transferred >= 50 else 0.0
        return {"score": score, "transferred": transferred}

    def grade_task_2(self) -> dict[str, Any]:
        """
        Medium: Cost-Optimised Rebalancing (5-day episode).

        Start at 1.0.
          -0.20 per stockout-day (any SKU at any warehouse on any day).
          -0.05 per $100 of (actual_total_cost - baseline_optimal_cost).
        baseline_optimal_cost = 320.0 (pre-calculated).
        Clamped to [0.0, 1.0].

        Returns
        -------
        dict with keys: "score", "stockout_days", "actual_cost", "cost_excess".
        """
        BASELINE_OPTIMAL_COST: float = 500.0
        s = self._internal_state
        demand_log: list[dict] = s.get("demand_log", []) or []

        # Count stockout-days: any entry with shortfall > 0 is one stockout-day
        stockout_days: int = sum(1 for entry in demand_log if entry["shortfall"] > 0)

        acc = s.get("cost_accumulator", {"holding": 0.0, "shipping": 0.0, "unfulfilled": 0.0})
        actual_total_cost: float = acc.get("holding", 0.0) + acc.get("shipping", 0.0)

        cost_excess = max(0.0, actual_total_cost - BASELINE_OPTIMAL_COST)
        score = 1.0 - (0.04 * stockout_days) - (0.02 * (cost_excess / 100.0))
        score = max(0.0, min(1.0, score))

        return {
            "score":        score,
            "stockout_days": stockout_days,
            "actual_cost":  actual_total_cost,
            "cost_excess":  cost_excess,
        }

    def grade_task_3(self) -> dict[str, Any]:
        """
        Hard: Supply Shock (7-day episode).

        On Day 3 the environment injects:
          "Supplier 1 offline — shipments delayed indefinitely"
        and freezes all supplier_1 in-transit shipments.

        score = (total_revenue - total_costs) / MAX_POSSIBLE_PROFIT
        total_revenue = fulfilled_units × $10.0 per unit
        MAX_POSSIBLE_PROFIT = 2800.0 (hardcoded baseline)
        Clamped to [0.0, 1.0].

        Returns
        -------
        dict with keys: "score", "total_revenue", "total_costs", "profit".
        """
        MAX_POSSIBLE_PROFIT: float = 2800.0
        REVENUE_PER_UNIT: float = 10.0
        s = self._internal_state
        demand_log: list[dict] = s.get("demand_log", []) or []

        total_fulfilled: int = sum(entry["fulfilled"] for entry in demand_log)
        total_revenue: float = total_fulfilled * REVENUE_PER_UNIT

        acc = s.get("cost_accumulator", {"holding": 0.0, "shipping": 0.0, "unfulfilled": 0.0})
        total_costs: float = acc.get("holding", 0.0) + acc.get("shipping", 0.0)

        profit: float = total_revenue - total_costs
        score = profit / MAX_POSSIBLE_PROFIT
        score = max(0.0, min(1.0, score))

        return {
            "score":         score,
            "total_revenue": total_revenue,
            "total_costs":   total_costs,
            "profit":        profit,
        }

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        """Construct an Observation from _internal_state (read-only)."""
        s = self._internal_state
        day: int = s["day"]

        warehouse_states: list[WarehouseState] = []
        for wh_id in WAREHOUSES:
            incoming = [
                IncomingShipment(
                    source_node=t["source_node"],
                    sku=t["sku"],
                    quantity=t["quantity"],
                    arrives_on_day=t["arrives_on_day"],
                )
                for t in s["in_transit"]
                if t["destination_node"] == wh_id
            ]
            warehouse_states.append(
                WarehouseState(
                    warehouse_id=wh_id,
                    current_inventory=dict(s["inventory"][wh_id]),
                    incoming_shipments=incoming,
                )
            )

        forecast: list[DemandForecast] = []
        for offset in range(1, _FORECAST_HORIZON + 1):
            fday = day + offset
            for wh_id in WAREHOUSES:
                pattern = s["demand_pattern"][wh_id]
                for sku in SKUS:
                    base_qty = pattern.get(sku, 0)
                    # Deterministic ±20 % jitter — hash-based so state() never mutates rng
                    jitter = 1.0 + 0.2 * (((hash((fday, wh_id, sku)) % 100) - 50) / 100.0)
                    forecast.append(
                        DemandForecast(
                            warehouse_id=wh_id,
                            sku=sku,
                            day=fday,
                            quantity=max(0, round(base_qty * jitter)),
                        )
                    )

        return Observation(
            current_day=day,
            warehouses=warehouse_states,
            shipping_rates=copy.deepcopy(SHIPPING_COSTS),
            transit_times=copy.deepcopy(TRANSIT_TIMES),
            demand_forecast=forecast,
            active_alerts=list(s["alerts"]),
        )

    def _validate_action(self, action: Action) -> list[str]:
        """
        Validate an Action against current inventory.
        Returns a list of human-readable violation strings.
        Empty list → action is fully valid.
        """
        violations: list[str] = []
        s = self._internal_state
        committed: dict[tuple[str, str], int] = {}

        for idx, t in enumerate(action.transfers):
            label = (
                f"Transfer[{idx}] "
                f"({t.source_node}→{t.destination_node} {t.sku} ×{t.quantity})"
            )

            if t.source_node not in NODES:
                violations.append(f"{label}: unknown source_node '{t.source_node}'")
                continue
            if t.destination_node not in NODES:
                violations.append(f"{label}: unknown destination_node '{t.destination_node}'")
                continue
            if t.source_node == t.destination_node:
                violations.append(f"{label}: source and destination are the same node")
                continue
            if t.sku not in SKUS:
                violations.append(f"{label}: unknown SKU '{t.sku}'")
                continue
            if t.quantity <= 0:
                violations.append(f"{label}: quantity must be > 0, got {t.quantity}")
                continue
            if t.source_node not in WAREHOUSES:
                violations.append(
                    f"{label}: source_node '{t.source_node}' is not a warehouse"
                )
                continue
            if t.destination_node not in TRANSIT_TIMES.get(t.source_node, {}):
                violations.append(
                    f"{label}: no route from '{t.source_node}' to '{t.destination_node}'"
                )
                continue

            key = (t.source_node, t.sku)
            committed[key] = committed.get(key, 0) + t.quantity
            available = s["inventory"][t.source_node].get(t.sku, 0)
            if committed[key] > available:
                violations.append(
                    f"Insufficient {t.sku} at {t.source_node}: "
                    f"need {committed[key]} (cumulative), have {available}"
                )

        return violations