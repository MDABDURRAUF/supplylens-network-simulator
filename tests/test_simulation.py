"""Behavioral checks for the demo simulation's accounting and scenarios."""

import pandas as pd
import pytest

from network_simulator.simulation import Scenario, sample_network, simulate


def test_sample_network_has_expected_connected_size():
    network = sample_network()
    assert len(network["suppliers"]) == 5
    assert len(network["warehouses"]) == 3
    assert len(network["products"]) == 10
    assert len(network["inventory"]) == 30
    assert not network["inventory"].duplicated(["warehouse", "sku"]).any()
    assert set(network["inventory"]["warehouse"]) == set(network["warehouses"]["warehouse"])
    assert set(network["inventory"]["sku"]) == set(network["products"]["sku"])


@pytest.mark.parametrize("strategy", ["baseline", "backup", "transfer"])
def test_demand_and_backlog_accounting(strategy):
    result = simulate(Scenario(kind="supplier closure", target="Supplier A", start_day=5, duration=15), strategy, days=30)
    detail = result.detail
    assert not detail.empty
    for column in ["demand", "fulfilled", "served_backorders", "new_backorders", "lost_units", "on_hand", "open_backorders", "arrivals"]:
        assert (detail[column] >= -1e-8).all(), column
    for row in detail.itertuples():
        assert row.demand == pytest.approx(row.fulfilled + row.new_backorders + row.lost_units)
    ordered = detail.sort_values(["warehouse", "sku", "day"])
    prior = ordered.groupby(["warehouse", "sku"])["open_backorders"].shift(fill_value=0)
    assert list(ordered["open_backorders"]) == pytest.approx(
        list(prior + ordered["new_backorders"] - ordered["served_backorders"])
    )
    assert result.metrics["fill_rate"] == pytest.approx(detail["fulfilled"].sum() / detail["demand"].sum())
    for column in ["demand", "fulfilled", "served_backorders", "new_backorders", "lost_units", "on_hand", "open_backorders", "arrivals", "procurement_cost", "transfer_cost"]:
        by_day = detail.groupby("day")[column].sum()
        actual = result.daily.set_index("day")[column]
        assert list(actual.loc[by_day.index]) == pytest.approx(list(by_day)), column


def test_warehouse_outage_blocks_service_and_receipts_only_in_window():
    network = sample_network()
    warehouse = network["warehouses"].iloc[0]["warehouse"]
    scenario = Scenario(kind="warehouse outage", target=warehouse, start_day=5, duration=4)
    result = simulate(scenario, "baseline", days=20, network=network)
    affected = result.detail[(result.detail["warehouse"] == warehouse) & result.detail["day"].between(5, 8)]
    assert not affected.empty
    assert affected["demand"].sum() > 0
    assert affected["fulfilled"].sum() == pytest.approx(0)
    assert affected["served_backorders"].sum() == pytest.approx(0)
    assert affected["arrivals"].sum() == pytest.approx(0)
    assert affected["new_backorders"].sum() > 0


def test_demand_surge_targets_one_warehouse_and_only_its_window():
    network = sample_network()
    warehouse = network["warehouses"].iloc[0]["warehouse"]
    control = simulate(Scenario(kind="demand surge", target=warehouse, start_day=5, duration=4, severity=1), "baseline", days=12, network=network)
    surge = simulate(Scenario(kind="demand surge", target=warehouse, start_day=5, duration=4, severity=2), "baseline", days=12, network=network)
    base = control.detail.set_index(["day", "warehouse", "sku"])["demand"].sort_index()
    actual = surge.detail.set_index(["day", "warehouse", "sku"])["demand"].sort_index()
    for index in base.index:
        day, location, _ = index
        multiplier = 2 if location == warehouse and 5 <= day <= 8 else 1
        assert actual[index] == pytest.approx(base[index] * multiplier)


def test_simulation_is_reproducible_and_does_not_mutate_inputs():
    network = sample_network()
    original = {key: value.copy(deep=True) for key, value in network.items()}
    scenario = Scenario(kind="supplier closure", target="Supplier A", start_day=5, duration=10)
    first = simulate(scenario, "backup", days=30, network=network)
    second = simulate(scenario, "backup", days=30, network=network)
    pd.testing.assert_frame_equal(first.daily, second.daily)
    pd.testing.assert_frame_equal(first.detail, second.detail)
    assert first.metrics == second.metrics
    for key, frame in network.items():
        pd.testing.assert_frame_equal(frame, original[key])


@pytest.mark.parametrize("strategy", ["baseline", "backup", "transfer"])
def test_stock_conservation_and_shared_supplier_capacity(strategy):
    network = sample_network()
    result = simulate(Scenario(kind="supplier closure", target="Supplier A", start_day=5, duration=15), strategy, days=30, network=network)
    detail = result.detail.sort_values(["warehouse", "sku", "day"]).copy()
    initial = network["inventory"].set_index(["warehouse", "sku"])["on_hand"]
    previous = detail.groupby(["warehouse", "sku"])["on_hand"].shift()
    initial_values = [initial.loc[(row.warehouse, row.sku)] for row in detail.itertuples()]
    previous = previous.fillna(pd.Series(initial_values, index=detail.index))
    transfers = result.events[result.events.event == "transfer"]
    outbound = transfers.groupby(["day", "source", "sku"])["quantity"].sum()
    sent = [outbound.get((row.day, row.warehouse, row.sku), 0.0) for row in detail.itertuples()]
    expected = previous + detail["arrivals"] - detail["fulfilled"] - detail["served_backorders"] - sent
    assert list(detail["on_hand"]) == pytest.approx(list(expected))

    orders = result.events[result.events.event == "order"]
    capacities = network["suppliers"].set_index("supplier")["daily_capacity"]
    for (day, supplier), quantity in orders.groupby(["day", "source"])["quantity"].sum().items():
        assert quantity <= capacities[supplier] + 1e-8, (day, supplier)
    closed_orders = orders[(orders.source == "Supplier A") & orders.day.between(5, 19)]
    assert closed_orders.empty


def test_transport_delay_adds_days_to_new_orders_from_target_supplier():
    network = sample_network()
    control = simulate(Scenario(kind="none", start_day=1, duration=1), days=8, network=network)
    delayed = simulate(Scenario(kind="transport delay", target="Supplier A", start_day=1, duration=1, severity=3), days=8, network=network)
    keys = ["day", "source", "destination", "sku"]
    before = control.events[(control.events.event == "order") & (control.events.day == 1)].set_index(keys)
    after = delayed.events[(delayed.events.event == "order") & (delayed.events.day == 1)].set_index(keys)
    assert len(before) > 0
    assert before.index.equals(after.index)
    for index in before.index:
        assert after.loc[index, "quantity"] == pytest.approx(before.loc[index, "quantity"])
        assert after.loc[index, "arrival_day"] == before.loc[index, "arrival_day"] + (3 if index[1] == "Supplier A" else 0)


def test_recovery_requires_seven_observed_clear_days_after_disruption():
    scenario = Scenario(kind="supplier closure", target="Supplier A", start_day=5, duration=10)
    result = simulate(scenario, "baseline", days=60)
    daily = result.daily.set_index("day")
    valid_starts = []
    for start in range(scenario.start_day + scenario.duration, 61 - 6):
        window = daily.loc[start:start + 6]
        if (window["shortage_locations"] == 0).all() and (window["open_backorders"] < 1e-9).all():
            valid_starts.append(start)
    assert result.metrics["recovery_day"] == (valid_starts[0] if valid_starts else None)
