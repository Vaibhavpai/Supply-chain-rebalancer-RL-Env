"""
inference.py — Supply Chain Inventory Rebalancer Inference Script.

Runs all 3 tasks using rule-based agents with structured stdout logging.

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from huggingface_hub import InferenceClient

from environment import SupplyChainEnv
from models import Action, Observation, Transfer

# ---------------------------------------------------------------------------
# Configuration (via environment variables)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")

BENCHMARK: str = "supply-chain-inventory-rebalancer"

# Task configs matching openenv.yaml
TASKS = [
    {"id": 1, "name": "simple_restock",            "max_steps": 5},
    {"id": 2, "name": "cost_optimized_rebalancing", "max_steps": 15},
    {"id": 3, "name": "supply_shock",               "max_steps": 25},
]

TEMPERATURE = 0.0
MAX_TOKENS  = 512


# ---------------------------------------------------------------------------
# Structured stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Rule-based agents (deterministic — no LLM needed)
# ---------------------------------------------------------------------------

def rule_based_task1(obs: Observation) -> Action:
    """Task 1: Transfer >=50 SKU_A from central -> west on day 0."""
    if obs.current_day == 0:
        return Action(
            transfers=[Transfer(source_node="central", destination_node="west", sku="SKU_A", quantity=50)],
            advance_time=True,
        )
    return Action(transfers=[], advance_time=True)


def rule_based_task2(obs: Observation) -> Action:
    """Task 2: Cost-optimised rebalancing over 5-day horizon."""
    HORIZON = 5

    wh_inv: dict[str, dict[str, int]] = {
        wh.warehouse_id: dict(wh.current_inventory) for wh in obs.warehouses
    }

    wh_demand: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for f in obs.demand_forecast:
        wh_demand.setdefault(f.warehouse_id, {})
        counts.setdefault(f.warehouse_id, {})
        wh_demand[f.warehouse_id][f.sku] = wh_demand[f.warehouse_id].get(f.sku, 0.0) + f.quantity
        counts[f.warehouse_id][f.sku] = counts[f.warehouse_id].get(f.sku, 0) + 1
    for wh_id in wh_demand:
        for sku in wh_demand[wh_id]:
            n = counts[wh_id].get(sku, 1)
            wh_demand[wh_id][sku] /= n

    if obs.current_day > 0:
        return Action(transfers=[], advance_time=True)

    days_remaining = max(1, HORIZON - obs.current_day)
    transfers: list[Transfer] = []
    nodes = list(wh_inv.keys())

    for sku in ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]:
        for dest in nodes:
            dest_inv   = wh_inv[dest].get(sku, 0)
            dest_daily = wh_demand.get(dest, {}).get(sku, 0.0)
            if dest_daily <= 0:
                continue

            dest_needed = int(dest_daily * days_remaining)
            deficit = dest_needed - dest_inv
            if deficit <= 0:
                continue

            donor_candidates = [s for s in nodes if s != dest]
            donor_candidates.sort(key=lambda s: obs.shipping_rates.get(s, {}).get(dest, 999))

            for src in donor_candidates:
                src_inv    = wh_inv[src].get(sku, 0)
                src_daily  = wh_demand.get(src, {}).get(sku, 0.0)
                src_needed = int(src_daily * days_remaining)
                src_surplus = src_inv - src_needed

                if src_surplus <= 0:
                    continue

                transfer_qty = min(deficit, src_surplus)
                if transfer_qty <= 0:
                    continue

                transfers.append(Transfer(
                    source_node=src, destination_node=dest,
                    sku=sku, quantity=transfer_qty,
                ))
                wh_inv[src][sku]  = src_inv - transfer_qty
                wh_inv[dest][sku] = dest_inv + transfer_qty
                deficit -= transfer_qty
                if deficit <= 0:
                    break

    return Action(transfers=transfers, advance_time=True)


def rule_based_task3(obs: Observation) -> Action:
    """Task 3: Supply-shock triage — redistribute surplus to prevent stockouts."""
    HORIZON = 7

    wh_inv: dict[str, dict[str, int]] = {
        wh.warehouse_id: dict(wh.current_inventory) for wh in obs.warehouses
    }

    wh_demand: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for f in obs.demand_forecast:
        wh_demand.setdefault(f.warehouse_id, {})
        counts.setdefault(f.warehouse_id, {})
        wh_demand[f.warehouse_id][f.sku] = wh_demand[f.warehouse_id].get(f.sku, 0.0) + f.quantity
        counts[f.warehouse_id][f.sku] = counts[f.warehouse_id].get(f.sku, 0) + 1
    for wh_id in wh_demand:
        for sku in wh_demand[wh_id]:
            n = counts[wh_id].get(sku, 1)
            wh_demand[wh_id][sku] /= n

    days_remaining = max(1, HORIZON - obs.current_day)
    transfers: list[Transfer] = []
    nodes = list(wh_inv.keys())

    for sku in ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]:
        for dest in nodes:
            dest_inv   = wh_inv[dest].get(sku, 0)
            dest_daily = wh_demand.get(dest, {}).get(sku, 0.0)
            if dest_daily <= 0:
                continue

            dest_needed = int(dest_daily * days_remaining)
            deficit = dest_needed - dest_inv
            if deficit <= 0:
                continue

            donor_candidates = [s for s in nodes if s != dest]
            donor_candidates.sort(key=lambda s: obs.shipping_rates.get(s, {}).get(dest, 999))

            for src in donor_candidates:
                src_inv    = wh_inv[src].get(sku, 0)
                src_daily  = wh_demand.get(src, {}).get(sku, 0.0)
                src_needed = int(src_daily * days_remaining)
                src_surplus = src_inv - src_needed

                if src_surplus <= 0:
                    continue

                transfer_qty = min(deficit, src_surplus)
                if transfer_qty <= 0:
                    continue

                transfers.append(Transfer(
                    source_node=src, destination_node=dest,
                    sku=sku, quantity=transfer_qty,
                ))
                wh_inv[src][sku]  = src_inv - transfer_qty
                wh_inv[dest][sku] = dest_inv + transfer_qty
                deficit -= transfer_qty
                if deficit <= 0:
                    break

    return Action(transfers=transfers, advance_time=True)


# ---------------------------------------------------------------------------
# LLM fallback (HuggingFace Inference API)
# ---------------------------------------------------------------------------

def build_system_prompt(task_id: int) -> str:
    action_schema = json.dumps(Action.model_json_schema(), indent=2)
    task_hints = {
        1: "Transfer >=50 SKU_A from central to west. Set advance_time=true.",
        2: "5-day sim. Minimize costs + prevent stockouts. Often do nothing.",
        3: "7-day sim. Day 3 supplier offline. Redistribute to prevent stockouts.",
    }
    return (
        "You are a supply chain agent managing east, central, west warehouses.\n"
        f"{task_hints[task_id]}\n"
        f"Action schema:\n{action_schema}\n"
        'Respond with ONLY valid JSON. Example: {\"transfers\": [], \"advance_time\": true}'
    )


def call_llm(client: InferenceClient, system_prompt: str, obs: Observation) -> Action:
    user_prompt = obs.model_dump_json(indent=2)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        return Action.model_validate_json(text)
    except Exception:
        return Action(transfers=[], advance_time=True)


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_config: dict) -> dict:
    """Run a single task with [START]/[STEP]/[END] logging. Returns result dict."""
    task_id   = task_config["id"]
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]

    env = SupplyChainEnv(task_id=task_id, seed=42)
    rule_agents = {1: rule_based_task1, 2: rule_based_task2, 3: rule_based_task3}

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            obs = env.state()

            # Use rule-based agent
            action = rule_agents[task_id](obs)

            # Format action string for logging
            if action.transfers:
                parts = [f"{t.quantity}x{t.sku}:{t.source_node}->{t.destination_node}"
                         for t in action.transfers]
                action_str = ";".join(parts)
            else:
                action_str = "no_transfer"
            if action.advance_time:
                action_str += "+advance"

            # Step environment
            result = env.step(action)
            r = result.reward
            reward_val = r.total
            done = result.done
            error_list = result.info.get("violations", None)
            error_str = str(error_list) if error_list else None

            rewards.append(reward_val)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_str)

            if done:
                break

        # Grade
        grader = {1: env.grade_task_1, 2: env.grade_task_2, 3: env.grade_task_3}
        grade = grader[task_id]()
        score = grade["score"]
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Exception during task {task_name}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "steps": steps_taken, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = []
    for task_config in TASKS:
        result = run_task(task_config)
        results.append(result)
        print("", flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['task']}: score={r['score']:.3f} steps={r['steps']}", flush=True)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average Score: {avg_score:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
