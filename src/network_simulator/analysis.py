"""Questions asked of a service network: which response, which site matters most, how much staff.

Every comparison runs the same network and disruption under different
choices. Differences are scenario estimates under the stated assumptions, not
measured savings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .services import STRATEGIES, ServiceResult, ServiceScenario, simulate_services

_KEEP = ["service_level", "abandonment_rate", "average_wait_days", "abandoned", "served_late", "peak_backlog", "lost_revenue",
         "sla_penalties", "response_cost", "total_impact", "recovery_day"]


def compare_strategies(network: dict, scenarios, days: int = 90, strategies=STRATEGIES) -> tuple[pd.DataFrame, dict[str, ServiceResult]]:
    """Run every response plan against the same disruption.

    Returns a table (one row per plan) and the full results by plan name.

    ``disruption_cost``          total impact minus the total impact of a normal
                                 period with no disruption and no response
    ``net_benefit_vs_baseline``  how much lower the total impact is than doing
                                 nothing; negative means the response cost
                                 more than it protected
    ``protected_per_dollar``     revenue and penalties saved per dollar spent
                                 on the response
    """
    normal = simulate_services(network, None, "baseline", days).metrics
    results = {s: simulate_services(network, scenarios, s, days) for s in strategies}
    base = results["baseline"].metrics if "baseline" in results else simulate_services(network, scenarios, "baseline", days).metrics
    rows = []
    for name, result in results.items():
        m = result.metrics
        protected = (base["lost_revenue"] + base["sla_penalties"]) - (m["lost_revenue"] + m["sla_penalties"])
        rows.append({"strategy": name, **{k: m[k] for k in _KEEP},
                     "disruption_cost": m["total_impact"] - normal["total_impact"],
                     "net_benefit_vs_baseline": base["total_impact"] - m["total_impact"],
                     "protected_per_dollar": protected / m["response_cost"] if m["response_cost"] > 0 else np.nan})
    return pd.DataFrame(rows), results


def best_strategy(comparison: pd.DataFrame) -> pd.Series:
    """The plan with the lowest total impact (lost revenue + penalties + response cost)."""
    return comparison.sort_values(["total_impact", "response_cost"]).iloc[0]


def site_criticality(network: dict, duration: int = 5, days: int = 60, start_day: int = 15, strategy: str = "baseline") -> pd.DataFrame:
    """Close each site in turn for ``duration`` days and rank them by what it costs.

    ``extra_impact`` is the total impact above a normal period. ``absorbed``
    is the share of the site's own demand over the closure that was still
    served on time (by remote staff or, with a sharing strategy, other sites).
    """
    normal = simulate_services(network, None, "baseline", days).metrics
    rows = []
    for site in network["sites"]["site"]:
        result = simulate_services(network, ServiceScenario("site outage", site, start_day, duration, 1.0), strategy, days)
        d = result.detail
        hit = d[(d["site"] == site) & d["day"].between(start_day, start_day + duration - 1)]
        rows.append({"site": site, "state": network["sites"].set_index("site").loc[site, "state"],
                     "extra_impact": result.metrics["total_impact"] - normal["total_impact"],
                     "service_level": result.metrics["service_level"], "abandoned": result.metrics["abandoned"],
                     "absorbed": float(hit["served_on_time"].sum() / hit["arrivals"].sum()) if hit["arrivals"].sum() else np.nan,
                     "recovery_day": result.metrics["recovery_day"]})
    return pd.DataFrame(rows).sort_values("extra_impact", ascending=False).reset_index(drop=True)


def scenario_matrix(network: dict, named_scenarios: dict, days: int = 90, strategies=STRATEGIES) -> pd.DataFrame:
    """Every named disruption against every response plan.

    One row per (disruption, plan) with service level, total impact and
    response cost, plus ``best`` marking the lowest-impact plan for each
    disruption.
    """
    rows = []
    for name, scenarios in named_scenarios.items():
        table, _ = compare_strategies(network, scenarios, days, strategies)
        winner = best_strategy(table)["strategy"]
        for r in table.itertuples(index=False):
            rows.append({"disruption": name, "strategy": r.strategy, "service_level": r.service_level,
                         "total_impact": r.total_impact, "response_cost": r.response_cost,
                         "net_benefit_vs_baseline": r.net_benefit_vs_baseline, "best": r.strategy == winner})
    return pd.DataFrame(rows)


def scale_staff(network: dict, factor: float) -> dict:
    """A copy of the network with every site's staff multiplied by ``factor``."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    out = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in network.items()}
    out["sites"]["staff"] = out["sites"]["staff"] * factor
    return out


def staffing_sweep(network: dict, scenarios=None, changes=(-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15), days: int = 90,
                   strategy: str = "baseline") -> pd.DataFrame:
    """What more or fewer staff would do, in normal times and under a disruption.

    For each staffing change: the yearly-equivalent labor cost difference is
    not computed; instead ``labor_cost_change`` is the change in scheduled
    labor cost over the simulated period, to be set against ``impact_normal``
    (lost revenue and penalties with no disruption) and ``impact_disrupted``.
    ``net_normal`` and ``net_disrupted`` add the labor change to each impact
    relative to today's staffing, so negative means better off overall.
    """
    rows = []
    for change in changes:
        net = scale_staff(network, 1 + change)
        normal = simulate_services(net, None, "baseline", days).metrics
        row = {"staffing_change": change, "labor_cost": normal["scheduled_labor_cost"], "service_level_normal": normal["service_level"],
               "impact_normal": normal["total_impact"]}
        if scenarios:
            hit = simulate_services(net, scenarios, strategy, days).metrics
            row.update(service_level_disrupted=hit["service_level"], impact_disrupted=hit["total_impact"])
        rows.append(row)
    table = pd.DataFrame(rows)
    today = table.loc[(table["staffing_change"].abs()).idxmin()]
    table["labor_cost_change"] = table["labor_cost"] - today["labor_cost"]
    table["net_normal"] = table["labor_cost_change"] + table["impact_normal"] - today["impact_normal"]
    if scenarios:
        table["net_disrupted"] = table["labor_cost_change"] + table["impact_disrupted"] - today["impact_disrupted"]
    return table


def monte_carlo(network: dict, scenarios=None, strategy: str = "baseline", days: int = 90, n_runs: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the simulation with random day-to-day demand.

    Returns ``(runs, summary)``: one row per run, and the 5th, 50th and 95th
    percentile of service level, abandoned requests and total impact. Only
    demand varies (Poisson around its expected value); the disruption itself
    is the same in every run.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    rows = []
    for i in range(n_runs):
        m = simulate_services(network, scenarios, strategy, days, demand_seed=seed + i).metrics
        rows.append({"run": i, "service_level": m["service_level"], "abandoned": m["abandoned"], "total_impact": m["total_impact"],
                     "peak_backlog": m["peak_backlog"]})
    runs = pd.DataFrame(rows)
    summary = runs.drop(columns="run").quantile([0.05, 0.5, 0.95]).T
    summary.columns = ["p05", "median", "p95"]
    return runs, summary.reset_index().rename(columns={"index": "measure"})


# ------------------------------------------------------------------ files
_TABLES = ("sites", "services", "demand", "dependencies")


def save_network(network: dict, folder: str | Path) -> Path:
    """Write a network as CSV files plus ``settings.json`` in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for table in _TABLES:
        if table in network and network[table] is not None:
            network[table].to_csv(folder / f"{table}.csv", index=False)
    meta = {"name": network.get("name", ""), "description": network.get("description", ""), "settings": network.get("settings") or {}}
    (folder / "settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def load_network(folder: str | Path) -> dict:
    """Read a network written by ``save_network`` (or hand-made files in the same layout)."""
    from .services import validate_network

    folder = Path(folder)
    network: dict = {}
    for table in _TABLES:
        path = folder / f"{table}.csv"
        if path.exists():
            network[table] = pd.read_csv(path)
    meta = json.loads((folder / "settings.json").read_text(encoding="utf-8")) if (folder / "settings.json").exists() else {}
    network.update(name=meta.get("name", folder.name), description=meta.get("description", ""), settings=meta.get("settings", {}))
    network.setdefault("dependencies", pd.DataFrame(columns=["site", "vendor", "capacity_share"]))
    validate_network(network)
    return network
