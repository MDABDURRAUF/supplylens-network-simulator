"""A small deterministic, daily supply-network simulation.

This is an illustrative planning model, not a representation of a real
company. Read README.md for the operational and cost assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    kind: str = "supplier closure"
    target: str = "Supplier A"
    start_day: int = 15
    duration: int = 28
    severity: float = 2.0


@dataclass
class Result:
    daily: pd.DataFrame
    detail: pd.DataFrame
    events: pd.DataFrame
    metrics: dict


def sample_network() -> dict[str, pd.DataFrame]:
    """Return a synthetic network; coordinates are schematic, not geographic."""
    suppliers = pd.DataFrame([
        ("Supplier A", 0.08, 0.13, 70.0), ("Supplier B", 0.08, 0.31, 70.0),
        ("Supplier C", 0.08, 0.50, 70.0), ("Supplier D", 0.08, 0.69, 70.0),
        ("Supplier E", 0.08, 0.87, 70.0),
    ], columns=["supplier", "x", "y", "daily_capacity"])
    warehouses = pd.DataFrame([
        ("West DC", "West", 0.55, 0.20), ("Central DC", "Central", 0.55, 0.50),
        ("East DC", "East", 0.55, 0.80),
    ], columns=["warehouse", "region", "x", "y"])
    names = suppliers.supplier.tolist()
    products = pd.DataFrame([
        (f"SKU-{i+1:02d}", names[i % 5], names[(i+1) % 5], 8.0 + i, 9.5 + i)
        for i in range(10)
    ], columns=["sku", "primary_supplier", "backup_supplier", "unit_cost", "backup_unit_cost"])
    inventory = []
    lanes = []
    for w, warehouse in enumerate(warehouses.warehouse):
        for i, p in products.iterrows():
            demand = float(3 + ((i + 2*w) % 4))
            inventory.append((warehouse, p.sku, 10*demand, demand))
            lanes.append((p.primary_supplier, warehouse, p.sku, 4 + ((i+w) % 3)))
            lanes.append((p.backup_supplier, warehouse, p.sku, 5 + ((i+w) % 3)))
    return {"suppliers": suppliers, "warehouses": warehouses, "products": products,
            "inventory": pd.DataFrame(inventory, columns=["warehouse", "sku", "on_hand", "daily_demand"]),
            "lanes": pd.DataFrame(lanes, columns=["supplier", "warehouse", "sku", "lead_time"])}


def _validate(network: dict, scenario: Scenario, strategy: str, days: int) -> None:
    if not isinstance(scenario, Scenario):
        raise ValueError("scenario must be a Scenario")
    if scenario.kind not in {"none", "supplier closure", "warehouse outage", "transport delay", "demand surge"}:
        raise ValueError("unknown scenario kind")
    if strategy not in {"baseline", "backup", "transfer"}:
        raise ValueError("unknown recovery strategy")
    if not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError("days must be an integer between 1 and 365")
    if not isinstance(scenario.start_day, int) or not 1 <= scenario.start_day <= days:
        raise ValueError("start_day must be a day within the simulation")
    if not isinstance(scenario.duration, int) or scenario.duration < 1:
        raise ValueError("duration must be a positive integer")
    if not math.isfinite(float(scenario.severity)) or scenario.severity < 1:
        raise ValueError("severity must be at least 1")
    supplier_names = set(network["suppliers"].supplier)
    warehouse_names = set(network["warehouses"].warehouse)
    if scenario.kind in {"supplier closure", "transport delay"} and scenario.target not in supplier_names:
        raise ValueError("target must be an existing supplier")
    if scenario.kind in {"warehouse outage", "demand surge"} and scenario.target not in warehouse_names:
        raise ValueError("target must be an existing warehouse")
    if scenario.kind == "transport delay" and not float(scenario.severity).is_integer():
        raise ValueError("transport delay severity must be a whole number of days")


def simulate(scenario: Scenario, strategy: str = "baseline", days: int = 90,
             network: dict[str, pd.DataFrame] | None = None) -> Result:
    """Simulate daily receipts, demand, transfers, and replenishment.

    An outage defers inbound shipments; orders already in transit are not
    canceled by a supplier closure. Demand not immediately filled is split
    equally between lost sales and backlog. Backlog is served first next day.
    """
    network = sample_network() if network is None else network
    _validate(network, scenario, strategy, days)
    suppliers = network["suppliers"].set_index("supplier")
    products = network["products"].set_index("sku")
    warehouse_region = network["warehouses"].set_index("warehouse")["region"].to_dict()
    inv = network["inventory"].set_index(["warehouse", "sku"])
    on_hand = inv["on_hand"].astype(float).to_dict()
    demand_rate = inv["daily_demand"].astype(float).to_dict()
    keys = list(on_hand)
    backlog = {key: 0.0 for key in keys}
    lanes = network["lanes"].set_index(["supplier", "warehouse", "sku"])["lead_time"].to_dict()
    warehouse_names = network["warehouses"].warehouse.tolist()
    pipeline: list[dict] = []
    events: list[dict] = []
    detail: list[dict] = []
    daily: list[dict] = []
    last_day = scenario.start_day + scenario.duration - 1

    def active(day: int) -> bool:
        return scenario.kind != "none" and scenario.start_day <= day <= last_day

    def unavailable(warehouse: str, day: int) -> bool:
        return active(day) and scenario.kind == "warehouse outage" and warehouse == scenario.target

    def emit(day: int, event: str, source: str, destination: str, sku: str,
             quantity: float, cost: float, arrival_day: int) -> None:
        events.append(dict(day=day, event=event, source=source, destination=destination,
                           sku=sku, quantity=quantity, cost=cost, arrival_day=arrival_day))

    for day in range(1, days+1):
        records = {key: dict(day=day, warehouse=key[0], region=warehouse_region[key[0]], sku=key[1],
                             demand=0.0, fulfilled=0.0, served_backorders=0.0,
                             new_backorders=0.0, lost_units=0.0, on_hand=0.0,
                             open_backorders=0.0, arrivals=0.0, procurement_cost=0.0,
                             transfer_cost=0.0) for key in keys}
        # Due shipments wait in transit until the destination is open.
        still_in_transit = []
        for shipment in pipeline:
            key = (shipment["destination"], shipment["sku"])
            if shipment["arrival_day"] <= day and not unavailable(key[0], day):
                on_hand[key] += shipment["quantity"]
                records[key]["arrivals"] += shipment["quantity"]
                emit(day, "arrival", shipment["source"], key[0], key[1], shipment["quantity"], 0.0, day)
            else:
                still_in_transit.append(shipment)
        pipeline = still_in_transit

        for key in keys:
            warehouse, sku = key
            d = demand_rate[key]
            if active(day) and scenario.kind == "demand surge" and warehouse == scenario.target:
                d *= scenario.severity
            record = records[key]
            record["demand"] = d
            if not unavailable(warehouse, day):
                served_old = min(backlog[key], on_hand[key])
                backlog[key] -= served_old
                on_hand[key] -= served_old
                served_new = min(d, on_hand[key])
                on_hand[key] -= served_new
            else:
                served_old = served_new = 0.0
            shortfall = d - served_new
            record["served_backorders"] = served_old
            record["fulfilled"] = served_new
            record["new_backorders"] = shortfall * 0.5
            record["lost_units"] = shortfall * 0.5
            backlog[key] += shortfall * 0.5

        # Transfer policy activates only from the disruption's start, and uses
        # already-owned stock. It does not invent supply or ship from outages.
        if strategy == "transfer" and scenario.kind != "none" and day >= scenario.start_day:
            for recipient, sku in keys:
                if unavailable(recipient, day):
                    continue
                key = (recipient, sku)
                inbound_transfer = sum(s["quantity"] for s in pipeline
                                       if s["destination"] == recipient and s["sku"] == sku and s["type"] == "transfer")
                need = max(0.0, 3*demand_rate[key] + backlog[key] - on_hand[key] - inbound_transfer)
                for donor in warehouse_names:
                    if need <= 1e-9:
                        break
                    donor_key = (donor, sku)
                    if donor == recipient or donor_key not in on_hand or unavailable(donor, day):
                        continue
                    surplus = max(0.0, on_hand[donor_key] - 9*demand_rate[donor_key] - backlog[donor_key])
                    qty = min(need, surplus)
                    if qty <= 1e-9:
                        continue
                    on_hand[donor_key] -= qty
                    need -= qty
                    pipeline.append(dict(source=donor, destination=recipient, sku=sku,
                                         quantity=qty, arrival_day=day+1, type="transfer"))
                    cost = qty * 0.5
                    records[key]["transfer_cost"] += cost
                    emit(day, "transfer", donor, recipient, sku, qty, cost, day+1)

        # Shared supplier capacity is reset each day. Existing in-transit
        # inventory counts towards the reorder target, including transfers.
        remaining_capacity = suppliers["daily_capacity"].astype(float).to_dict()
        for warehouse, sku in keys:
            key = (warehouse, sku)
            inbound = sum(s["quantity"] for s in pipeline if s["destination"] == warehouse and s["sku"] == sku)
            need = max(0.0, 14*demand_rate[key] + backlog[key] - on_hand[key] - inbound)
            product = products.loc[sku]
            candidates = [(product.primary_supplier, float(product.unit_cost))]
            if strategy == "backup" and scenario.kind != "none" and day >= scenario.start_day:
                candidates.append((product.backup_supplier, float(product.backup_unit_cost)))
            for supplier, unit_cost in candidates:
                if need <= 1e-9:
                    break
                if active(day) and scenario.kind == "supplier closure" and supplier == scenario.target:
                    continue
                lane = (supplier, warehouse, sku)
                if lane not in lanes:
                    continue
                qty = min(need, remaining_capacity[supplier])
                if qty <= 1e-9:
                    continue
                remaining_capacity[supplier] -= qty
                need -= qty
                lead = int(lanes[lane])
                if active(day) and scenario.kind == "transport delay" and supplier == scenario.target:
                    lead += int(scenario.severity)
                arrival_day = day + max(1, lead)
                pipeline.append(dict(source=supplier, destination=warehouse, sku=sku,
                                     quantity=qty, arrival_day=arrival_day, type="order"))
                cost = qty * unit_cost
                records[key]["procurement_cost"] += cost
                emit(day, "order", supplier, warehouse, sku, qty, cost, arrival_day)

        for key in keys:
            records[key]["on_hand"] = on_hand[key]
            records[key]["open_backorders"] = backlog[key]
            detail.append(records[key])
        day_rows = list(records.values())
        cols = ["demand", "fulfilled", "served_backorders", "new_backorders", "lost_units",
                "on_hand", "open_backorders", "arrivals", "procurement_cost", "transfer_cost"]
        totals = {col: sum(row[col] for row in day_rows) for col in cols}
        totals["shortage_locations"] = len({row["warehouse"] for row in day_rows
                                             if row["demand"] - row["fulfilled"] > 1e-9})
        daily.append(dict(day=day, **totals))

    daily_df = pd.DataFrame(daily)
    detail_df = pd.DataFrame(detail)
    event_df = pd.DataFrame(events, columns=["day", "event", "source", "destination", "sku", "quantity", "cost", "arrival_day"])
    first_recovery = None
    streak = 0
    for row in daily_df.itertuples(index=False):
        if scenario.kind != "none" and row.day > last_day and row.shortage_locations == 0 and row.open_backorders < 1e-9:
            streak += 1
            if streak == 7:
                first_recovery = row.day - 6
                break
        else:
            streak = 0
    metrics = dict(demand=float(daily_df.demand.sum()), fulfilled=float(daily_df.fulfilled.sum()),
                   served_backorders=float(daily_df.served_backorders.sum()),
                   lost_units=float(daily_df.lost_units.sum()), open_backorders=float(daily_df.open_backorders.iloc[-1]),
                   fill_rate=float(daily_df.fulfilled.sum() / daily_df.demand.sum()) if daily_df.demand.sum() else 0.0,
                   shortage_days=int((daily_df.shortage_locations > 0).sum()),
                   affected_locations=int(detail_df.loc[detail_df.demand - detail_df.fulfilled > 1e-9, "warehouse"].nunique()),
                   procurement_cost=float(daily_df.procurement_cost.sum()), transfer_cost=float(daily_df.transfer_cost.sum()),
                   recovery_day=first_recovery)
    return Result(daily_df, detail_df, event_df, metrics)
