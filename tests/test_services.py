"""Checks for the services model: calendar, capacity, queues, responses, analysis and outputs."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import network_simulator as ns
from network_simulator import ServiceScenario as S
from network_simulator.cli import main


def tiny(staff=1, demand=10.0, sla=0, patience=1, remote=False, days_open="Every day", **site):
    """One site, one service line, one hour per request: easy to follow by hand."""
    row = {"site": "A", "city": "Austin", "state": "TX", "region": "South", "staff": staff, "hours_per_staff_day": 8,
           "days_open": days_open, "remote_share": 0.0, "overtime_max": 0.25, "labor_cost_per_hour": 40,
           "closed_on_federal_holidays": False, **site}
    return {
        "sites": pd.DataFrame([row]),
        "services": pd.DataFrame([{"service": "Job", "handle_hours": 1.0, "sla_days": sla, "patience_days": patience,
                                   "revenue_per_request": 100, "late_penalty": 10, "remote_eligible": remote, "priority": 1}]),
        "demand": pd.DataFrame([{"site": "A", "service": "Job", "daily_requests": demand}]),
        "dependencies": pd.DataFrame([{"site": "A", "vendor": "Software", "capacity_share": 0.5}]),
        "settings": {"start_date": "2026-01-05", "weekday_factor": [1.0] * 7},
    }


def two_sites(remote=True):
    net = tiny(demand=6.0, remote=remote)
    b = net["sites"].iloc[0].to_dict() | {"site": "B", "city": "Boston", "state": "MA", "region": "Northeast"}
    net["sites"] = pd.concat([net["sites"], pd.DataFrame([b])], ignore_index=True)
    net["demand"] = pd.DataFrame([{"site": s, "service": "Job", "daily_requests": 6.0} for s in ("A", "B")])
    net["dependencies"] = net["dependencies"].iloc[0:0]
    return net


@pytest.fixture(scope="module")
def clinics():
    return ns.urgent_care_network()


@pytest.fixture(scope="module")
def hurricane(clinics):
    return ns.presets(clinics)["Hurricane landfall (South)"]


# ------------------------------------------------------------------ US data
def test_federal_holidays_and_observed_dates():
    h = {name: day for day, name in ns.federal_holidays(2026).items()}
    assert len(h) == 11
    assert h["Birthday of Martin Luther King, Jr."] == dt.date(2026, 1, 19)
    assert h["Washington's Birthday"] == dt.date(2026, 2, 16)
    assert h["Memorial Day"] == dt.date(2026, 5, 25)
    assert h["Labor Day"] == dt.date(2026, 9, 7)
    assert h["Columbus Day"] == dt.date(2026, 10, 12)
    assert h["Thanksgiving Day"] == dt.date(2026, 11, 26)
    table = ns.holiday_table(2026).set_index("holiday")
    # 4 July 2026 is a Saturday, observed on Friday 3 July.
    assert table.loc["Independence Day", "weekday"] == "Saturday"
    assert table.loc["Independence Day", "observed"] == pd.Timestamp("2026-07-03")
    assert {d.year for d in ns.federal_holidays([2026, 2027])} == {2026, 2027}
    assert ns.federal_holidays(2027)[dt.date(2027, 5, 31)] == "Memorial Day"  # last Monday of May


def test_census_regions_cover_every_state():
    states = [s for group in ns.CENSUS_REGIONS.values() for s in group]
    assert len(states) == len(set(states)) == 51  # 50 states and DC
    assert ns.region_of("tx") == "South" and ns.region_of("MA") == "Northeast"
    assert ns.region_of("OH") == "Midwest" and ns.region_of("CA") == "West"
    with pytest.raises(ValueError, match="not a US state"):
        ns.region_of("ZZ")


# ---------------------------------------------------------------- queue logic
def test_queue_follows_hand_calculation():
    # 8 hours of capacity, 10 one-hour jobs a day, same-day target, customers wait one extra day.
    d = ns.simulate_services(tiny(), None, "baseline", days=7).detail.set_index("day")
    assert d.loc[1, ["served_on_time", "served_late", "backlog", "abandoned"]].tolist() == [8, 0, 2, 0]
    assert d.loc[2, ["served_on_time", "served_late", "backlog", "abandoned"]].tolist() == [6, 2, 4, 0]
    assert d.loc[5, ["served_on_time", "served_late", "backlog"]].tolist() == [0, 8, 10]
    assert d.loc[6, ["served_late", "abandoned", "backlog"]].tolist() == [8, 2, 10]  # the two oldest give up
    assert d.loc[6, "wait_days"] == 8  # eight jobs each waited one day


@pytest.mark.parametrize("strategy", ns.STRATEGIES)
def test_request_accounting(clinics, hurricane, strategy):
    r = ns.simulate_services(clinics, hurricane, strategy, days=60)
    d = r.detail
    for c in ["arrivals", "served", "served_on_time", "served_late", "abandoned", "backlog", "served_by_other_site"]:
        assert (d[c] >= -1e-9).all(), c
    np.testing.assert_allclose(d["served"], d["served_on_time"] + d["served_late"])
    # Every request is served, abandoned or still waiting.
    by = d.groupby(["site", "service"])
    ending = d[d["day"] == 60].set_index(["site", "service"])["backlog"]
    np.testing.assert_allclose(by["arrivals"].sum(), by["served"].sum() + by["abandoned"].sum() + ending.reindex(by["arrivals"].sum().index))
    for c in ["arrivals", "served", "abandoned", "backlog", "lost_revenue", "sla_penalty"]:
        np.testing.assert_allclose(r.daily.set_index("day")[c], d.groupby("day")[c].sum(), err_msg=c)
    m = r.metrics
    assert m["service_level"] == pytest.approx(d["served_on_time"].sum() / d["arrivals"].sum())
    assert m["total_impact"] == pytest.approx(m["lost_revenue"] + m["sla_penalties"] + m["overtime_cost"] + m["temp_cost"] + m["handoff_cost"])
    # Hours used never exceed hours available (plus overtime and temporary staff).
    s = r.sites
    assert (s["used_hours"] + s["shared_out_hours"] <= s["available_hours"] + 1e-6).all()
    assert (s["overtime_hours"] <= s["available_hours"] * 0.20 + 1e-6).all()


def test_normal_operation_meets_every_target():
    for make in ns.SAMPLE_NETWORKS.values():
        m = ns.simulate_services(make(), None, "combined", days=60).metrics
        assert m["service_level"] == pytest.approx(1.0) and m["abandoned"] == 0 and m["total_impact"] == 0
        assert m["recovery_day"] is None  # nothing to recover from; responses stay switched off


# ------------------------------------------------------------------ calendar
def test_opening_days_and_federal_holidays():
    net = ns.support_center_network()  # Mon-Sat, closed on federal holidays; day 1 is Monday 5 Jan 2026
    r = ns.simulate_services(net, None, "baseline", days=21)
    by_day = r.detail.groupby("day")["arrivals"].sum()
    hours = r.sites.groupby("day")["base_hours"].sum()
    for day in (7, 14, 21):  # Sundays
        assert by_day[day] == 0 and hours[day] == 0
    assert by_day[15] == 0 and hours[15] == 0  # Monday 19 Jan 2026, Martin Luther King Jr. Day
    assert by_day[16] > 0
    clinics = ns.simulate_services(ns.urgent_care_network(), None, "baseline", days=21)
    assert (clinics.detail.groupby("day")["arrivals"].sum() > 0).all()  # clinics open every day
    monday, saturday = r.detail[r.detail["day"] == 1]["arrivals"].sum(), r.detail[r.detail["day"] == 6]["arrivals"].sum()
    assert saturday == pytest.approx(monday * 0.45 / 1.15)  # the network's weekday pattern


# ---------------------------------------------------------------- disruptions
def test_site_outage_blocks_on_site_work_but_not_remote_staff():
    net = two_sites(remote=False)
    r = ns.simulate_services(net, S("site outage", "A", 3, 2, 1.0), "combined", days=6).detail
    a = r[r["site"] == "A"].set_index("day")
    assert a.loc[[3, 4], "served"].sum() == 0  # on-site work cannot be done anywhere else
    assert (r[r["site"] == "B"]["served_on_time"] == 6).all()
    # With 25% of staff remote-capable and remote-eligible work, 2 of 8 hours keep going.
    net = two_sites(remote=True)
    net["sites"]["remote_share"] = 0.25
    a = ns.simulate_services(net, S("site outage", "A", 3, 2, 1.0), "baseline", days=6).detail.query("site == 'A'").set_index("day")
    assert a.loc[3, "served"] == pytest.approx(2.0)
    half = ns.simulate_services(net, S("site outage", "A", 3, 2, 0.5), "baseline", days=6).sites.query("site == 'A'").set_index("day")
    assert half.loc[3, "available_hours"] == pytest.approx(8 * 0.5 + 8 * 0.5 * 0.25)


def test_staff_shortage_vendor_outage_and_surge_sizes():
    net = tiny(demand=4.0)
    s = ns.simulate_services(net, S("staff shortage", "A", 2, 2, 0.25), "baseline", days=4).sites.set_index("day")
    assert s.loc[1, "available_hours"] == 8 and s.loc[2, "available_hours"] == 6 and s.loc[4, "available_hours"] == 8
    v = ns.simulate_services(net, S("vendor outage", "Software", 2, 1, 1.0), "baseline", days=3).sites.set_index("day")
    assert v.loc[2, "available_hours"] == pytest.approx(4.0)  # the site depends on the vendor for half its capacity
    half = ns.simulate_services(net, S("vendor outage", "Software", 2, 1, 0.5), "baseline", days=3).sites.set_index("day")
    assert half.loc[2, "available_hours"] == pytest.approx(6.0)
    d = ns.simulate_services(net, S("demand surge", "A", 2, 2, 1.5), "baseline", days=4).detail.set_index("day")
    assert d["arrivals"].tolist() == [4, 6, 6, 4]
    both = [S("staff shortage", "A", 2, 1, 0.5), S("vendor outage", "Software", 2, 1, 1.0)]
    assert ns.simulate_services(net, both, "baseline", days=3).sites.set_index("day").loc[2, "available_hours"] == pytest.approx(2.0)


def test_targets_resolve_to_sites_states_and_regions(clinics):
    assert ns.resolve_sites(clinics, "Houston Clinic") == ["Houston Clinic"]
    assert ns.resolve_sites(clinics, "FL") == ["Miami Clinic"]
    assert set(ns.resolve_sites(clinics, "South")) == {"Houston Clinic", "Miami Clinic", "Atlanta Clinic"}
    assert len(ns.resolve_sites(clinics, ns.ALL_SITES)) == 6
    with pytest.raises(ValueError, match="not a site, state or region"):
        ns.resolve_sites(clinics, "Mars")
    r = ns.simulate_services(clinics, S("site outage", "South", 10, 3, 1.0), "baseline", days=20).sites
    closed = r[r["day"].between(10, 12) & (r["available_hours"] < r["base_hours"])]
    assert set(closed["site"]) == {"Houston Clinic", "Miami Clinic", "Atlanta Clinic"}


# ------------------------------------------------------------------ responses
def test_overtime_adds_bounded_capacity_at_time_and_a_half():
    net = tiny(demand=7.0)
    scenario = S("demand surge", "A", 2, 3, 2.0)  # 14 jobs a day against 8 hours
    base = ns.simulate_services(net, scenario, "baseline", days=8)
    over = ns.simulate_services(net, scenario, "overtime", days=8)
    s = over.sites.set_index("day")
    assert s.loc[1, "overtime_hours"] == 0  # not before the disruption
    assert s.loc[2, "overtime_hours"] == pytest.approx(2.0)  # 25% of 8 hours
    assert over.metrics["overtime_cost"] == pytest.approx(s["overtime_hours"].sum() * 40 * 1.5)
    assert over.metrics["served"] > base.metrics["served"] and over.metrics["abandoned"] < base.metrics["abandoned"]
    assert base.metrics["response_cost"] == 0


def test_load_sharing_only_moves_remote_eligible_work():
    scenario = S("site outage", "A", 2, 2, 1.0)
    shared = ns.simulate_services(two_sites(remote=True), scenario, "load sharing", days=6)
    d = shared.detail.set_index(["site", "day"])
    # B has 2 spare hours a day and uses them on A's queue.
    assert d.loc[("A", 2), "served_by_other_site"] == pytest.approx(2.0) and d.loc[("A", 2), "served"] == pytest.approx(2.0)
    assert (shared.detail.query("site == 'B'")["served_on_time"] == 6).all()  # B's own customers are served first
    assert shared.metrics["handoff_cost"] == pytest.approx(shared.metrics["served_by_other_site"] * 2.0)
    assert shared.sites.set_index(["site", "day"]).loc[("B", 2), "shared_out_hours"] == pytest.approx(2.0)
    on_site = ns.simulate_services(two_sites(remote=False), scenario, "load sharing", days=6)
    assert on_site.metrics["served_by_other_site"] == 0 and on_site.metrics["handoff_cost"] == 0


def test_temp_staff_arrive_after_the_lead_time_and_are_billed_per_hour():
    net = tiny(demand=7.0)
    net["settings"] |= {"temp_pool_hours_per_day": 10, "temp_lead_days": 2, "temp_cost_multiplier": 1.5, "temp_productivity": 0.8}
    r = ns.simulate_services(net, S("staff shortage", "A", 3, 6, 0.5), "temp staff", days=10)
    s = r.sites.set_index("day")
    assert s.loc[[3, 4], "temp_hours_paid"].sum() == 0 and s.loc[5, "temp_hours_paid"] == 10
    assert s.loc[5, "temp_hours_used"] <= 10 * 0.8 + 1e-9  # a temporary hour does 80% of a regular hour's work
    assert r.metrics["temp_cost"] == pytest.approx(s["temp_hours_paid"].sum() * 40 * 1.5)
    none = ns.simulate_services(tiny(demand=7.0), S("staff shortage", "A", 3, 6, 0.5), "temp staff", days=10)
    assert none.metrics["temp_cost"] == 0  # no agency pool configured


def test_responses_never_reduce_service(clinics):
    for name, scenarios in ns.presets(clinics).items():
        table, _ = ns.compare_strategies(clinics, scenarios, days=60)
        t = table.set_index("strategy")
        assert (t["service_level"] >= t.loc["baseline", "service_level"] - 1e-9).all(), name
        assert (t["abandoned"] <= t.loc["baseline", "abandoned"] + 1e-6).all(), name
        assert t.loc["baseline", "net_benefit_vs_baseline"] == 0 and t.loc["baseline", "response_cost"] == 0
        np.testing.assert_allclose(t["net_benefit_vs_baseline"], t.loc["baseline", "total_impact"] - t["total_impact"])


# ------------------------------------------------------------------- analysis
def test_comparison_and_best_strategy(clinics, hurricane):
    table, results = ns.compare_strategies(clinics, hurricane, days=90)
    assert list(table["strategy"]) == list(ns.STRATEGIES) and set(results) == set(ns.STRATEGIES)
    best = ns.best_strategy(table)
    assert best["total_impact"] == table["total_impact"].min()
    assert best["strategy"] == "load sharing"  # other regions' clinics pick up telehealth and follow-ups
    row = table.set_index("strategy").loc["load sharing"]
    base = table.set_index("strategy").loc["baseline"]
    protected = base["lost_revenue"] + base["sla_penalties"] - row["lost_revenue"] - row["sla_penalties"]
    assert row["protected_per_dollar"] == pytest.approx(protected / row["response_cost"])
    assert np.isnan(base["protected_per_dollar"])
    assert (table["disruption_cost"] == table["total_impact"]).all()  # normal operation costs nothing here


def test_site_criticality_scenario_matrix_and_staffing(clinics, hurricane):
    crit = ns.site_criticality(clinics, duration=5, days=45)
    assert crit["extra_impact"].is_monotonic_decreasing and len(crit) == 6
    assert crit.iloc[0]["site"] == "Houston Clinic"  # the largest clinic
    assert crit["absorbed"].between(0, 1).all()
    shared = ns.site_criticality(clinics, duration=5, days=45, strategy="load sharing")
    assert shared["extra_impact"].sum() < crit["extra_impact"].sum()

    some = dict(list(ns.presets(clinics).items())[:2])
    matrix = ns.scenario_matrix(clinics, some, days=45)
    assert len(matrix) == 2 * len(ns.STRATEGIES) and (matrix.groupby("disruption")["best"].sum() == 1).all()

    sweep = ns.staffing_sweep(clinics, hurricane, changes=(-0.1, 0.0, 0.1), days=60)
    assert sweep["labor_cost"].is_monotonic_increasing and sweep["service_level_disrupted"].is_monotonic_increasing
    today = sweep[sweep["staffing_change"] == 0.0].iloc[0]
    assert today["labor_cost_change"] == 0 and today["net_normal"] == 0 and today["net_disrupted"] == 0
    assert sweep.iloc[-1]["labor_cost_change"] == pytest.approx(0.1 * today["labor_cost"])
    assert ns.scale_staff(clinics, 1.5)["sites"]["staff"].sum() == pytest.approx(1.5 * clinics["sites"]["staff"].sum())
    with pytest.raises(ValueError):
        ns.scale_staff(clinics, 0)


def test_monte_carlo_is_reproducible_and_varies(clinics, hurricane):
    runs, summary = ns.monte_carlo(clinics, hurricane, "baseline", days=45, n_runs=6, seed=3)
    again, _ = ns.monte_carlo(clinics, hurricane, "baseline", days=45, n_runs=6, seed=3)
    pd.testing.assert_frame_equal(runs, again)
    assert runs["total_impact"].nunique() > 1
    s = summary.set_index("measure")
    assert (s["p05"] <= s["median"]).all() and (s["median"] <= s["p95"]).all()
    expected = ns.simulate_services(clinics, hurricane, "baseline", days=45).metrics["service_level"]
    assert abs(s.loc["service_level", "median"] - expected) < 0.03
    a = ns.simulate_services(clinics, hurricane, "baseline", days=30)
    b = ns.simulate_services(clinics, hurricane, "baseline", days=30)
    pd.testing.assert_frame_equal(a.daily, b.daily)  # deterministic without a seed


# --------------------------------------------------------- inputs and outputs
def test_inputs_are_not_changed_and_networks_round_trip(tmp_path, clinics, hurricane):
    before = {k: v.copy(deep=True) for k, v in clinics.items() if isinstance(v, pd.DataFrame)}
    first = ns.simulate_services(clinics, hurricane, "combined", days=40).metrics
    for k, frame in before.items():
        pd.testing.assert_frame_equal(clinics[k], frame)
    loaded = ns.load_network(ns.save_network(clinics, tmp_path / "net"))
    assert loaded["name"] == clinics["name"]
    assert ns.simulate_services(loaded, hurricane, "combined", days=40).metrics == first


def test_validation():
    net = tiny()
    for bad, match in [(S("meteor", "A"), "unknown scenario kind"), (S("site outage", "A", 0, 3), "start_day"),
                       (S("site outage", "A", 3, 0), "duration"), (S("site outage", "A", 3, 2, 1.5), "between 0 and 1"),
                       (S("demand surge", "A", 3, 2, 0.5), "at least 1"), (S("vendor outage", "Nobody", 3, 2), "not a vendor"),
                       (S("staff shortage", "Nowhere", 3, 2, 0.2), "not a site")]:
        with pytest.raises(ValueError, match=match):
            ns.simulate_services(net, bad, "baseline", days=10)
    with pytest.raises(ValueError, match="unknown strategy"):
        ns.simulate_services(net, None, "pray", days=10)
    with pytest.raises(ValueError, match="days must be"):
        ns.simulate_services(net, None, "baseline", days=0)
    broken = tiny()
    broken["sites"] = broken["sites"].drop(columns="staff")
    with pytest.raises(ValueError, match="sites is missing columns: staff"):
        ns.validate_network(broken)
    broken = tiny(days_open="Sometimes")
    with pytest.raises(ValueError, match="days_open"):
        ns.validate_network(broken)
    broken = tiny()
    broken["demand"] = pd.DataFrame([{"site": "Z", "service": "Job", "daily_requests": 1}])
    with pytest.raises(ValueError, match="not defined"):
        ns.validate_network(broken)
    broken = tiny()
    broken["services"]["handle_hours"] = 0
    with pytest.raises(ValueError, match="handle_hours"):
        ns.validate_network(broken)


def test_brief_and_cli(tmp_path, capsys, clinics, hurricane):
    table, _ = ns.compare_strategies(clinics, hurricane, days=90)
    normal = ns.simulate_services(clinics, None, "baseline", 90).metrics
    text = ns.brief_summary(clinics, hurricane, table, normal)
    base = table.set_index("strategy").loc["baseline"]
    assert f"{base['abandoned']:,.0f} customers give up" in text and "The best response is load sharing" in text
    html = ns.build_brief(clinics, hurricane, table, normal, ns.site_criticality(clinics, days=45))
    for needle in ("Response plans compared", "Which site matters most", "Houston Clinic", "not a forecast", 'class="best"'):
        assert needle in html

    assert main(["list"]) == 0
    listing = capsys.readouterr().out
    assert "Urgent-care clinics" in listing and "Hurricane landfall (South)" in listing
    brief, csv, folder = tmp_path / "b.html", tmp_path / "c.csv", tmp_path / "net"
    assert main(["run", "--preset", "Hurricane landfall (South)", "--brief", str(brief), "--csv", str(csv), "--days", "60"]) == 0
    assert "Lowest total impact" in capsys.readouterr().out and brief.stat().st_size > 4000 and len(pd.read_csv(csv)) == 5
    assert main(["export", "--network", "Customer-support centers", "--dir", str(folder)]) == 0
    assert main(["run", "--folder", str(folder), "--preset", "Seasonal demand peak (all sites)", "--days", "45"]) == 0
    assert main(["run", "--preset", "Alien invasion"]) == 1
    assert "unknown preset" in capsys.readouterr().err
