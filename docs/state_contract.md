# `_internal_state` Contract

> **Audience:** Dev 2 (implements `step()`) and Dev 3 (implements graders).  
> **Owner:** Dev 1.  Do not add new keys without updating this doc.

---

## Overview

`SupplyChainEnv._internal_state` is a plain `dict[str, Any]` that holds the
complete mutable simulation state.  It is the single source of truth — the
`Observation` returned to the agent is a read-only projection of this dict.

All keys are initialised to `None` in `__init__` and populated to their
working types on every call to `reset()`.

---

## Key Reference

| Key | Python type | Set by | Description |
|-----|-------------|--------|-------------|
| `day` | `int` | `reset()` / `step()` | Current simulation day, 0-indexed. Incremented at the **end** of each `step()` call when `advance_time=True`. |
| `inventory` | `dict[str, dict[str, int]]` | `reset()` / `step()` | On-hand stock keyed `inventory[warehouse_id][sku]`. Outer keys are always the three warehouse IDs (`"east"`, `"central"`, `"west"`). Inner keys are the five SKUs. Values are non-negative integers. **Deduct on dispatch; credit on arrival.** |
| `in_transit` | `list[dict]` | `step()` | Shipments currently moving between nodes. Each entry is a plain dict with keys: `source_node: str`, `destination_node: str`, `sku: str`, `quantity: int`, `arrives_on_day: int`. Entries are removed when `arrives_on_day <= day` at the start of the arrival phase. |
| `demand_log` | `list[dict]` | `step()` | Append-only record of every demand realisation. Each entry: `day: int`, `warehouse_id: str`, `sku: str`, `demanded: int`, `fulfilled: int`, `shortfall: int`. Used by Dev 3 graders to compute fill-rate and service-level metrics. |
| `cost_accumulator` | `dict[str, float]` | `step()` | Running totals of costs across the entire episode. Sub-keys: `"holding"` (Σ units × HOLDING_COSTS rate), `"shipping"` (Σ units × SHIPPING_COSTS rate), `"unfulfilled"` (Σ shortfall units × penalty rate). Dev 3 reads these for `grade_task_*()`. |
| `alerts` | `list[str]` | `reset()` / `step()` | Active operational alerts surfaced verbatim in every `Observation`. Dev 2 may append or clear entries as scenario events trigger (e.g. lift supplier disruption on day 7 for task 2). |
| `horizon` | `int` | `reset()` | Total episode length in days. Episode is `done` when `day >= horizon`. |
| `rng` | `random.Random` | `reset()` | Seeded random instance for stochastic demand sampling in `step()`. **Never call `rng` outside `step()`** — doing so breaks determinism for callers who interleave `state()` calls. |
| `demand_pattern` | `dict[str, dict[str, int]]` | `reset()` | Base daily demand keyed `demand_pattern[warehouse_id][sku]`. Treat as read-only; stochastic jitter is applied at runtime in `step()` and `_get_observation()`. |

---

## Invariants Dev 2 Must Maintain

1. `inventory[wh][sku] >= 0` at all times — clamp to 0, never go negative.
2. Entries are removed from `in_transit` **before** demand is realised on the same day (so arriving stock can fill that day's demand).
3. `rng` is the **only** source of randomness.  Do not import or call any other random source.
4. `day` is incremented **after** reward is computed and **before** `StepResult` is returned.
5. All `cost_accumulator` sub-keys must remain `float` (not `int`).

---

## Invariants Dev 3 Must Maintain

1. Graders are **read-only** — never mutate `_internal_state`.
2. All grader methods return `dict[str, Any]` with a `"score"` key in `[0.0, 1.0]`.
3. Graders should handle the case where `demand_log` is empty (episode not yet run).

---

## Execution Order Inside `step()` (Dev 2 reference)

```
1. _validate_action()          → collect violations, price penalties
2. Dispatch valid transfers     → deduct inventory, push to in_transit
3. [short-circuit if advance_time=False]
4. Arrive shipments             → credit inventory, remove from in_transit
5. Realise demand               → sample via rng, fulfill, append demand_log
6. Charge holding costs         → accumulate cost_accumulator["holding"]
7. _compute_reward()            → build RewardBreakdown
8. day += 1
9. done = day >= horizon
10. return StepResult
```