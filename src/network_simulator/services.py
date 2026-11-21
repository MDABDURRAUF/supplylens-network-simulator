"""A daily simulation of a service-delivery network.

Goods networks hold inventory; service networks cannot. A clinic, support
center or field team has staff hours that are used on the day or lost, and
customers who wait in a queue and eventually give up. This module models that:

* **sites** with staff, opening days and a share of staff able to work remotely;
* **service lines** with a handling time, a service-level target (days), a
  patience limit after which waiting customers leave, revenue and a late penalty;
* **demand** per site and service line, following a weekday pattern and the
  US federal holiday calendar;
* **dependencies** on outside vendors (a records system, a phone carrier).

Disruptions: a site outage, a staff shortage, a vendor outage or a demand
surge, aimed at one site, a state, a US Census region or every site. Several
can be combined in one run.

Responses: overtime, load sharing between sites (for work that can be done
remotely), temporary staff, or all three.

The model is deterministic by default (expected values) and every number in
it is a planning assumption. It is not a forecast of a real disruption.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .us import closed_for_holiday, federal_holidays

KINDS = ("site outage", "staff shortage", "vendor outage", "demand surge")
STRATEGIES = ("baseline", "overtime", "load sharing", "temp staff", "combined")
ALL_SITES = "All sites"

DEFAULT_SETTINGS = {
    "start_date": "2026-01-05",        # day 1 of the simulation (a Monday)
    "weekday_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6],  # demand multiplier Mon..Sun
    "overtime_multiplier": 1.5,        # pay rate for overtime hours relative to regular hours
    "temp_pool_hours_per_day": 0.0,    # temporary staff hours an agency can supply each day
    "temp_lead_days": 3,               # days between asking for temporary staff and their first shift
    "temp_cost_multiplier": 1.6,       # agency bill rate relative to the site's labor cost
    "temp_productivity": 0.8,          # output of a temporary hour relative to a regular hour
    "handoff_cost_per_request": 2.0,   # cost of routing one request to another site
}
OPEN_DAYS = {"Mon-Fri": {0, 1, 2, 3, 4}, "Mon-Sat": {0, 1, 2, 3, 4, 5}, "Every day": {0, 1, 2, 3, 4, 5, 6}}


@dataclass(frozen=True)
class ServiceScenario:
    """One disruption.

    ``kind``      one of ``KINDS``
    ``target``    a site, a state code, a Census region, ``"All sites"``; or a
                  vendor name for a vendor outage
    ``severity``  share of capacity lost (0-1) for a site outage, staff
                  shortage or vendor outage; demand multiplier (>= 1) for a surge
    """

    kind: str = "site outage"
    target: str = ALL_SITES
    start_day: int = 15
    duration: int = 5
    severity: float = 1.0
    label: str = ""

    @property
    def end_day(self) -> int:
        return self.start_day + self.duration - 1

    def describe(self) -> str:
        if self.label:
            return self.label
        size = f"x{self.severity:g} demand" if self.kind == "demand surge" else f"{self.severity:.0%} of capacity"
        return f"{self.kind.capitalize()} at {self.target}: {size}, day {self.start_day} for {self.duration} days"


@dataclass
class ServiceResult:
    daily: pd.DataFrame     # network totals per day
    detail: pd.DataFrame    # per day, site and service line
    sites: pd.DataFrame     # per day and site: hours available and used
    metrics: dict
    scenarios: list = field(default_factory=list)
    strategy: str = "baseline"


def settings_of(network: dict) -> dict:
    """The network's settings with defaults filled in."""
    return {**DEFAULT_SETTINGS, **(network.get("settings") or {})}


def resolve_sites(network: dict, target: str) -> list[str]:
    """Sites matched by a target: a site name, a state code, a region, or ``"All sites"``."""
    sites = network["sites"]
    if target == ALL_SITES:
        return sites["site"].tolist()
    for column in ("site", "state", "region"):
        hit = sites.loc[sites[column].astype(str) == str(target), "site"].tolist()
        if hit:
            return hit
    raise ValueError(f"target '{target}' is not a site, state or region in this network")


def validate_network(network: dict) -> None:
    """Raise ``ValueError`` describing the first problem found in a network."""
    need = {
        "sites": ["site", "state", "region", "staff", "hours_per_staff_day", "days_open", "remote_share", "overtime_max",
                  "labor_cost_per_hour", "closed_on_federal_holidays"],
        "services": ["service", "handle_hours", "sla_days", "patience_days", "revenue_per_request", "late_penalty",
                     "remote_eligible", "priority"],
        "demand": ["site", "service", "daily_requests"],
    }
    for table, columns in need.items():
        if table not in network or not isinstance(network[table], pd.DataFrame):
            raise ValueError(f"network is missing the '{table}' table")
        missing = [c for c in columns if c not in network[table].columns]
        if missing:
            raise ValueError(f"{table} is missing columns: {', '.join(missing)}")
    sites, services, demand = network["sites"], network["services"], network["demand"]
    if sites.empty or services.empty or demand.empty:
        raise ValueError("sites, services and demand must each have at least one row")
    if sites["site"].duplicated().any() or services["service"].duplicated().any():
        raise ValueError("site and service names must be unique")
    if demand.duplicated(["site", "service"]).any():
        raise ValueError("demand must have one row per site and service")
    if not set(demand["site"]) <= set(sites["site"]) or not set(demand["service"]) <= set(services["service"]):
        raise ValueError("demand refers to a site or service that is not defined")
    if not set(sites["days_open"]) <= set(OPEN_DAYS):
        raise ValueError(f"days_open must be one of: {', '.join(OPEN_DAYS)}")
    checks = [(sites, ["staff", "hours_per_staff_day", "labor_cost_per_hour", "overtime_max"], 0, None),
              (sites, ["remote_share"], 0, 1), (services, ["sla_days", "patience_days", "revenue_per_request", "late_penalty"], 0, None),
              (demand, ["daily_requests"], 0, None)]
    for frame, columns, low, high in checks:
        for c in columns:
            values = pd.to_numeric(frame[c], errors="coerce")
            if values.isna().any() or (values < low).any() or (high is not None and (values > high).any()):
                raise ValueError(f"{c} must be a number of at least {low}" + (f" and at most {high}" if high is not None else ""))
    if (pd.to_numeric(services["handle_hours"], errors="coerce").fillna(0) <= 0).any():
        raise ValueError("handle_hours must be greater than zero")
    deps = network.get("dependencies")
    if deps is not None and len(deps):
        if not {"site", "vendor", "capacity_share"} <= set(deps.columns):
            raise ValueError("dependencies needs the columns: site, vendor, capacity_share")
        if not set(deps["site"]) <= set(sites["site"]):
            raise ValueError("dependencies refers to a site that is not defined")
        if not deps["capacity_share"].between(0, 1).all():
            raise ValueError("capacity_share must be between 0 and 1")


def _validate_run(network: dict, scenarios: list[ServiceScenario], strategy: str, days: int) -> None:
    validate_network(network)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'; choose from: {', '.join(STRATEGIES)}")
    if not isinstance(days, int) or not 1 <= days <= 366:
        raise ValueError("days must be an integer between 1 and 366")
    deps = network.get("dependencies")
    vendors = set(deps["vendor"]) if deps is not None and len(deps) else set()
    for s in scenarios:
        if not isinstance(s, ServiceScenario):
            raise ValueError("each scenario must be a ServiceScenario")
        if s.kind not in KINDS:
            raise ValueError(f"unknown scenario kind '{s.kind}'")
        if not isinstance(s.start_day, int) or not 1 <= s.start_day <= days:
            raise ValueError("start_day must be a day within the simulation")
        if not isinstance(s.duration, int) or s.duration < 1:
            raise ValueError("duration must be a positive integer")
        if not math.isfinite(float(s.severity)):
            raise ValueError("severity must be a finite number")
        if s.kind == "demand surge":
            if s.severity < 1:
                raise ValueError("a demand surge needs a multiplier of at least 1")
        elif not 0 <= s.severity <= 1:
            raise ValueError("severity must be between 0 and 1 (share of capacity lost)")
        if s.kind == "vendor outage":
            if s.target not in vendors:
                raise ValueError(f"'{s.target}' is not a vendor in this network's dependencies")
        else:
            resolve_sites(network, s.target)


def simulate_services(network: dict, scenarios: ServiceScenario | list[ServiceScenario] | None = None,
                      strategy: str = "baseline", days: int = 90, demand_seed: int | None = None) -> ServiceResult:
    """Simulate the network day by day.

    Each day, in order: requests arrive; each site serves its own queue
    (service lines in priority order, oldest request first) using regular
    hours, then temporary staff, then overtime; with load sharing, sites with
    hours left serve other sites' remotely-doable requests; finally, requests
    that have waited longer than their patience limit leave.

    Capacity on a day is ``staff x hours`` if the site is open, reduced by
    staff shortages and vendor outages. In a site outage of severity ``v`` only
    ``1 - v`` of hours can do on-site work; the remote-capable share of the
    rest keeps working, on remotely-doable service lines only.

    A response plan is active from the first day of the first disruption
    until the disruptions are over and the queue has cleared.

    With ``demand_seed`` set, daily arrivals are drawn from a Poisson
    distribution around the expected demand instead of being the expected value.
    """
    scenarios = [] if scenarios is None else [scenarios] if isinstance(scenarios, ServiceScenario) else list(scenarios)
    _validate_run(network, scenarios, strategy, days)
    cfg = settings_of(network)
    rng = np.random.default_rng(demand_seed) if demand_seed is not None else None
    site_rows = network["sites"].set_index("site")
    svc_rows = network["services"].sort_values(["priority", "service"]).set_index("service")
    site_names, svc_names = site_rows.index.tolist(), svc_rows.index.tolist()
    rate = {(r.site, r.service): float(r.daily_requests) for r in network["demand"].itertuples(index=False)}
    deps = network.get("dependencies")
    dep = {(r.site, r.vendor): float(r.capacity_share) for r in deps.itertuples(index=False)} if deps is not None and len(deps) else {}
    start_date = pd.Timestamp(cfg["start_date"])
    holidays = federal_holidays(range(start_date.year, (start_date + pd.Timedelta(days=days)).year + 1))
    weekday_factor = list(cfg["weekday_factor"])

    targets = {i: (resolve_sites(network, s.target) if s.kind != "vendor outage" else
                   [site for (site, vendor) in dep if vendor == s.target]) for i, s in enumerate(scenarios)}
    affected_sites = sorted({site for sites in targets.values() for site in sites})
    first_start = min((s.start_day for s in scenarios), default=None)
    last_end = max((s.end_day for s in scenarios), default=0)
    use_overtime = strategy in ("overtime", "combined")
    use_sharing = strategy in ("load sharing", "combined")
    use_temp = strategy in ("temp staff", "combined") and cfg["temp_pool_hours_per_day"] > 0

    queues = {(site, svc): deque() for site in site_names for svc in svc_names if (site, svc) in rate}
    handle = svc_rows["handle_hours"].astype(float).to_dict()
    sla = svc_rows["sla_days"].astype(float).to_dict()
    patience = svc_rows["patience_days"].astype(float).to_dict()
    remote_ok = svc_rows["remote_eligible"].astype(bool).to_dict()
    detail, site_log, daily = [], [], []
    backlog_yesterday = 0.0

    for day in range(1, days + 1):
        date = start_date + pd.Timedelta(days=day - 1)
        dow = date.weekday()
        responding = first_start is not None and day >= first_start and (day <= last_end or backlog_yesterday > 1e-9)
        rec = {key: dict(day=day, date=date, site=key[0], service=key[1], arrivals=0.0, served=0.0, served_on_time=0.0,
                         served_late=0.0, served_by_other_site=0.0, abandoned=0.0, backlog=0.0, wait_days=0.0)
               for key in queues}
        pools, site_rec = {}, {}

        # ---- capacity for the day
        temp_share = {}
        if use_temp and responding and day >= first_start + int(cfg["temp_lead_days"]) and affected_sites:
            weights = {s: float(site_rows.loc[s, "staff"]) for s in affected_sites}
            total = sum(weights.values()) or 1.0
            temp_share = {s: cfg["temp_pool_hours_per_day"] * w / total for s, w in weights.items()}
        for site in site_names:
            row = site_rows.loc[site]
            is_open = dow in OPEN_DAYS[row["days_open"]] and not (bool(row["closed_on_federal_holidays"]) and closed_for_holiday(date, holidays))
            base = float(row["staff"]) * float(row["hours_per_staff_day"]) if is_open else 0.0
            staff_left, outage, vendor_lost, surge = 1.0, 0.0, 0.0, 1.0
            for i, s in enumerate(scenarios):
                if not s.start_day <= day <= s.end_day or site not in targets[i]:
                    continue
                if s.kind == "staff shortage":
                    staff_left *= 1 - s.severity
                elif s.kind == "site outage":
                    outage = max(outage, s.severity)
                elif s.kind == "vendor outage":
                    vendor_lost += s.severity * dep.get((site, s.target), 0.0)
                else:
                    surge *= s.severity
            available = base * staff_left * (1 - min(1.0, vendor_lost))
            general = available * (1 - outage)
            remote_only = available * outage * float(row["remote_share"])
            ot = float(row["overtime_max"]) if use_overtime and responding else 0.0
            # Temporary staff work on site, so an outage keeps them out too.
            temp_hours = temp_share.get(site, 0.0) if is_open else 0.0
            pools[site] = {"general": general, "remote": remote_only, "temp": temp_hours * cfg["temp_productivity"] * (1 - outage),
                           "ot_general": general * ot, "ot_remote": remote_only * ot}
            site_rec[site] = dict(day=day, date=date, site=site, open=is_open, base_hours=base, available_hours=general + remote_only,
                                  used_hours=0.0, overtime_hours=0.0, temp_hours_paid=temp_hours, temp_hours_used=0.0,
                                  shared_out_hours=0.0, overtime_cost=0.0, handoff_cost=0.0,
                                  temp_cost=temp_hours * float(row["labor_cost_per_hour"]) * cfg["temp_cost_multiplier"],
                                  scheduled_labor_cost=base * float(row["labor_cost_per_hour"]))
            # ---- arrivals
            for svc in svc_names:
                key = (site, svc)
                if key not in queues or not is_open:
                    continue
                expected = rate[key] * weekday_factor[dow] * surge
                qty = float(rng.poisson(expected)) if rng is not None else expected
                if qty > 0:
                    queues[key].append([day, qty])
                    rec[key]["arrivals"] = qty

        def serve(key: tuple, hours_by_pool: dict, order: list[str], log: dict, shared: bool = False) -> None:
            """Serve the queue at ``key`` oldest first from the named pools."""
            queue, h = queues[key], handle[key[1]]
            for pool in order:
                while queue and hours_by_pool[pool] > 1e-9:
                    arrived, qty = queue[0]
                    done = min(qty, hours_by_pool[pool] / h)
                    hours_by_pool[pool] -= done * h
                    waited = day - arrived
                    r = rec[key]
                    r["served"] += done
                    r["wait_days"] += done * waited
                    r["served_on_time" if waited <= sla[key[1]] else "served_late"] += done
                    if shared:
                        r["served_by_other_site"] += done
                        log["shared_out_hours"] += done * h
                        rec_site = site_rec[key[0]]
                        rec_site["handoff_cost"] += done * cfg["handoff_cost_per_request"]
                    elif pool.startswith("ot_"):
                        log["overtime_hours"] += done * h
                    elif pool == "temp":
                        log["temp_hours_used"] += done * h
                    else:
                        log["used_hours"] += done * h
                    if done >= qty - 1e-12:
                        queue.popleft()
                    else:
                        queue[0][1] = qty - done

        # ---- each site serves its own queue: regular hours, then temporary staff, then overtime
        for site in site_names:
            for svc in svc_names:
                key = (site, svc)
                if key in queues:
                    order = ["remote", "general", "temp", "ot_remote", "ot_general"] if remote_ok[svc] else ["general", "temp", "ot_general"]
                    serve(key, pools[site], order, site_rec[site])
            site_rec[site]["overtime_cost"] = (site_rec[site]["overtime_hours"] * float(site_rows.loc[site, "labor_cost_per_hour"])
                                               * cfg["overtime_multiplier"])

        # ---- load sharing: spare regular hours serve other sites' remotely-doable requests, oldest first
        if use_sharing and responding:
            waiting = sorted(((queues[(site, svc)][0][0], svc_rows.loc[svc, "priority"], site, svc)
                              for (site, svc) in queues if remote_ok[svc] and queues[(site, svc)]),
                             key=lambda t: (t[1], t[0]))
            for _, _, site, svc in waiting:
                for donor in sorted((d for d in site_names if d != site), key=lambda d: -(pools[d]["general"] + pools[d]["remote"])):
                    if not queues[(site, svc)]:
                        break
                    if pools[donor]["general"] + pools[donor]["remote"] > 1e-9:
                        serve((site, svc), pools[donor], ["remote", "general"], site_rec[donor], shared=True)

        # ---- customers who have waited past their patience leave
        for key, queue in queues.items():
            while queue and day - queue[0][0] >= patience[key[1]]:
                rec[key]["abandoned"] += queue.popleft()[1]
            rec[key]["backlog"] = sum(q for _, q in queue)

        detail.extend(rec.values())
        site_log.extend(site_rec.values())
        cols = ["arrivals", "served", "served_on_time", "served_late", "served_by_other_site", "abandoned", "backlog", "wait_days"]
        totals = {c: sum(r[c] for r in rec.values()) for c in cols}
        for c in ("overtime_hours", "temp_hours_paid", "overtime_cost", "temp_cost", "handoff_cost", "scheduled_labor_cost"):
            totals[c] = sum(r[c] for r in site_rec.values())
        daily.append(dict(day=day, date=date, **totals))
        backlog_yesterday = totals["backlog"]

    daily_df, detail_df, sites_df = pd.DataFrame(daily), pd.DataFrame(detail), pd.DataFrame(site_log)
    revenue = detail_df["service"].map(svc_rows["revenue_per_request"].astype(float))
    penalty = detail_df["service"].map(svc_rows["late_penalty"].astype(float))
    detail_df["lost_revenue"] = detail_df["abandoned"] * revenue
    detail_df["sla_penalty"] = detail_df["served_late"] * penalty
    daily_df["lost_revenue"] = detail_df.groupby("day")["lost_revenue"].sum().to_numpy()
    daily_df["sla_penalty"] = detail_df.groupby("day")["sla_penalty"].sum().to_numpy()

    recovery, streak = None, 0
    if scenarios:
        for row in daily_df.itertuples(index=False):
            if row.day > last_end and row.abandoned < 1e-9 and row.served_late < 1e-9:
                streak += 1
                if streak == 7:
                    recovery = row.day - 6
                    break
            else:
                streak = 0
    arrivals, served = float(daily_df["arrivals"].sum()), float(daily_df["served"].sum())
    m = dict(
        requests=arrivals,
        served=served,
        served_on_time=float(daily_df["served_on_time"].sum()),
        served_late=float(daily_df["served_late"].sum()),
        abandoned=float(daily_df["abandoned"].sum()),
        ending_backlog=float(daily_df["backlog"].iloc[-1]),
        peak_backlog=float(daily_df["backlog"].max()),
        service_level=float(daily_df["served_on_time"].sum() / arrivals) if arrivals else 1.0,
        abandonment_rate=float(daily_df["abandoned"].sum() / arrivals) if arrivals else 0.0,
        average_wait_days=float(daily_df["wait_days"].sum() / served) if served else 0.0,
        served_by_other_site=float(daily_df["served_by_other_site"].sum()),
        lost_revenue=float(detail_df["lost_revenue"].sum()),
        sla_penalties=float(detail_df["sla_penalty"].sum()),
        overtime_cost=float(daily_df["overtime_cost"].sum()),
        temp_cost=float(daily_df["temp_cost"].sum()),
        handoff_cost=float(daily_df["handoff_cost"].sum()),
        scheduled_labor_cost=float(daily_df["scheduled_labor_cost"].sum()),
        recovery_day=recovery,
    )
    m["response_cost"] = m["overtime_cost"] + m["temp_cost"] + m["handoff_cost"]
    m["total_impact"] = m["lost_revenue"] + m["sla_penalties"] + m["response_cost"]
    return ServiceResult(daily_df, detail_df, sites_df, m, scenarios, strategy)
