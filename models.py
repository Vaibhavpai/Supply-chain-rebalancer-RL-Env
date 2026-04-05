"""
models.py — Pydantic v2 schemas for Supply Chain Inventory Rebalancer RL Environment.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shipment / Inventory
# ---------------------------------------------------------------------------

class IncomingShipment(BaseModel):
    source_node: str
    sku: str
    quantity: int
    arrives_on_day: int


class WarehouseState(BaseModel):
    warehouse_id: str                          # "east" | "central" | "west"
    current_inventory: dict[str, int]          # SKU_A..SKU_E → qty
    incoming_shipments: list[IncomingShipment]


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class DemandForecast(BaseModel):
    warehouse_id: str
    sku: str
    day: int
    quantity: int


# ---------------------------------------------------------------------------
# Observation (env → agent)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    current_day: int
    warehouses: list[WarehouseState]
    shipping_rates: dict[str, dict[str, float]]   # node → node → $/unit
    transit_times: dict[str, dict[str, int]]      # node → node → days
    demand_forecast: list[DemandForecast]          # next 3 days
    active_alerts: list[str]                       # e.g. "supplier_1 disrupted"


# ---------------------------------------------------------------------------
# Action (agent → env)
# ---------------------------------------------------------------------------

class Transfer(BaseModel):
    source_node: str
    destination_node: str
    sku: str
    quantity: int


class Action(BaseModel):
    transfers: list[Transfer] = Field(default_factory=list)
    advance_time: bool = True


# ---------------------------------------------------------------------------
# State Snapshot (for transparency + debugging)
# ---------------------------------------------------------------------------

class StateSnapshot(BaseModel):
    """Immutable snapshot of environment state at any point."""
    current_day: int
    total_inventory: dict[str, int]          # SKU → total across all warehouses
    inventory_per_warehouse: dict[str, dict[str, int]]  # warehouse → sku → qty
    total_in_transit: int                    # units currently inflight
    unfulfilled_demand_so_far: int           # cumulative stockouts
    fulfilled_demand_so_far: int             # cumulative fulfilled


# ---------------------------------------------------------------------------
# Enhanced Reward Breakdown (Process Supervision)
# ---------------------------------------------------------------------------

class StepRewardMetrics(BaseModel):
    """Per-step intermediate metrics for process supervision."""
    # Core fulfillment metrics
    fulfilled_units: int                     # units fulfilled this step
    stockout_units: int                      # units unfulfilled this step
    
    # Cost metrics
    shipping_cost: float                     # $/unit × qty shipped
    holding_cost: float                      # carrying cost incurred
    
    # Decision quality metrics
    inventory_balance_score: float           # [0.0, 1.0] how balanced inventory is
    demand_forecast_alignment: float         # [0.0, 1.0] how well transfers align with forecast
    stockout_prevention_score: float         # [0.0, 1.0] how well action prevents future stockouts
    safety_margin_ratio: float               # inventory / (forecasted_demand * horizon)


class ProcessSupervisionReward(BaseModel):
    """Process supervision breakdown with per-step intermediate rewards."""
    # Base fulfillment rewards
    fulfilled_demand_reward: float           # +2.0 × fulfilled_units
    stockout_penalty: float                  # -5.0 × stockout_units
    
    # Cost penalties
    holding_cost_penalty: float              # -1.0 × holding_cost
    shipping_cost_penalty: float             # -1.0 × shipping_cost
    
    # Process supervision rewards (per-step intermediate signals)
    inventory_balance_reward: float          # +0.5 if balanced, -0.5 if imbalanced
    demand_forecast_accuracy_reward: float   # +1.0 × alignment score
    proactive_restocking_bonus: float        # +2.0 if action prevents future stockout
    safety_margin_bonus: float               # +0.5 if maintaining safety buffer (5 days supply)
    temporal_incentive_bonus: float          # +day_offset bonus for early intervention
    
    # Validation penalties
    invalid_action_penalty: float            # -50.0 if action violated constraints
    safety_violation_penalty: float          # -10.0 for routing errors / unsafe actions
    
    # Composite score
    total: float                             # sum of all components


class RewardBreakdown(BaseModel):
    """Legacy compatibility wrapper; use ProcessSupervisionReward for RL training."""
    fulfilled_demand_reward: float
    holding_cost_penalty: float
    shipping_cost_penalty: float
    invalid_action_penalty: float
    total: float


class StepResult(BaseModel):
    observation: Observation
    reward: ProcessSupervisionReward        # enhanced reward for RL training
    reward_legacy: RewardBreakdown | None = None  # for backward compatibility
    done: bool
    info: dict