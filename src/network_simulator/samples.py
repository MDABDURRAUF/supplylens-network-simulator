"""Illustrative US service networks and disruption presets.

The networks are made up. Staffing, demand, revenue and vendor figures are
chosen to be plausible, not taken from any real organization. The presets
describe the *kind* of event; their durations and severities are assumptions
to be edited, not forecasts.
"""

from __future__ import annotations

import pandas as pd

from .services import ALL_SITES, ServiceScenario
from .us import region_of

_SITE_COLUMNS = ["site", "city", "state", "staff", "hours_per_staff_day", "days_open", "remote_share", "overtime_max",
                 "labor_cost_per_hour", "closed_on_federal_holidays"]
_SERVICE_COLUMNS = ["service", "handle_hours", "sla_days", "patience_days", "revenue_per_request", "late_penalty",
                    "remote_eligible", "priority"]


# Approximate city coordinates, used only to draw the map.
_CITY_COORDINATES = {
    "Houston": (29.76, -95.37), "Miami": (25.76, -80.19), "Atlanta": (33.75, -84.39), "Chicago": (41.88, -87.63),
    "Boston": (42.36, -71.06), "Phoenix": (33.45, -112.07), "Dallas": (32.78, -96.80), "Tampa": (27.95, -82.46),
    "Columbus": (39.96, -83.00), "Salt Lake City": (40.76, -111.89), "Charlotte": (35.23, -80.84), "Nashville": (36.16, -86.78),
    "Minneapolis": (44.98, -93.27), "Denver": (39.74, -104.99), "Sacramento": (38.58, -121.49),
}


def _network(name, description, sites, services, demand, dependencies, settings) -> dict:
    site_df = pd.DataFrame(sites, columns=_SITE_COLUMNS)
    site_df.insert(3, "region", site_df["state"].map(region_of))
    site_df["lat"] = site_df["city"].map(lambda c: _CITY_COORDINATES[c][0])
    site_df["lon"] = site_df["city"].map(lambda c: _CITY_COORDINATES[c][1])
    return {
        "name": name,
        "description": description,
        "sites": site_df,
        "services": pd.DataFrame(services, columns=_SERVICE_COLUMNS),
        "demand": pd.DataFrame(demand, columns=["site", "service", "daily_requests"]),
        "dependencies": pd.DataFrame(dependencies, columns=["site", "vendor", "capacity_share"]),
        "settings": settings,
    }


def urgent_care_network() -> dict:
    """Six urgent-care clinics in four Census regions, open every day."""
    sites = [
        # site              city        st   staff hrs  open         remote ot    $/hr closed on holidays
        ("Houston Clinic",  "Houston",  "TX", 14, 8, "Every day", 0.15, 0.20, 62, False),
        ("Miami Clinic",    "Miami",    "FL", 12, 8, "Every day", 0.15, 0.20, 60, False),
        ("Atlanta Clinic",  "Atlanta",  "GA", 12, 8, "Every day", 0.15, 0.20, 58, False),
        ("Chicago Clinic",  "Chicago",  "IL", 13, 8, "Every day", 0.15, 0.20, 64, False),
        ("Boston Clinic",   "Boston",   "MA", 11, 8, "Every day", 0.15, 0.20, 68, False),
        ("Phoenix Clinic",  "Phoenix",  "AZ", 10, 8, "Every day", 0.15, 0.20, 57, False),
    ]
    services = [
        # service             hours  sla patience revenue late  remote priority
        ("Walk-in visit",      0.75,  0,  1,       165,   0,   False, 1),
        ("Telehealth visit",   0.40,  0,  2,        75,   0,   True,  2),
        ("Results follow-up",  0.25,  2,  6,        20,   8,   True,  3),
    ]
    per_staff = {"Walk-in visit": 5.6, "Telehealth visit": 3.2, "Results follow-up": 3.6}
    demand = [(s[0], svc, round(s[3] * rate, 1)) for s in sites for svc, rate in per_staff.items()]
    dependencies = [(s[0], "Health records system", 0.70) for s in sites] + [(s[0], "Phone and internet carrier", 0.35) for s in sites]
    settings = {"start_date": "2026-01-05", "weekday_factor": [1.10, 1.0, 0.95, 0.95, 1.0, 1.0, 0.95],
                "temp_pool_hours_per_day": 64, "temp_lead_days": 3, "temp_cost_multiplier": 1.7, "temp_productivity": 0.8,
                "handoff_cost_per_request": 3.0}
    return _network("Urgent-care clinics", "Six clinics; walk-in visits must be done on site, telehealth and follow-ups can be "
                    "handled by any clinic.", sites, services, demand, dependencies, settings)


def support_center_network() -> dict:
    """Four customer-support centers, mostly remote-capable work, closed on federal holidays."""
    sites = [
        ("Dallas Center",         "Dallas",         "TX", 60, 7.5, "Mon-Sat", 0.70, 0.15, 31, True),
        ("Tampa Center",          "Tampa",          "FL", 45, 7.5, "Mon-Sat", 0.70, 0.15, 29, True),
        ("Columbus Center",       "Columbus",       "OH", 50, 7.5, "Mon-Sat", 0.70, 0.15, 30, True),
        ("Salt Lake City Center", "Salt Lake City", "UT", 40, 7.5, "Mon-Sat", 0.70, 0.15, 30, True),
    ]
    services = [
        ("Phone support",      0.20, 0, 1, 18, 0,  True, 1),
        ("Email and chat",     0.25, 1, 4, 12, 3,  True, 2),
        ("Claims processing",  0.60, 5, 15, 40, 25, True, 3),
    ]
    per_staff = {"Phone support": 14.0, "Email and chat": 8.0, "Claims processing": 2.6}
    demand = [(s[0], svc, round(s[3] * rate, 1)) for s in sites for svc, rate in per_staff.items()]
    dependencies = [(s[0], "Cloud contact-center platform", 0.85) for s in sites] + [(s[0], "Phone and internet carrier", 0.50) for s in sites]
    settings = {"start_date": "2026-01-05", "weekday_factor": [1.15, 1.05, 1.0, 1.0, 0.95, 0.45, 0.0],
                "temp_pool_hours_per_day": 300, "temp_lead_days": 5, "temp_cost_multiplier": 1.5, "temp_productivity": 0.7,
                "handoff_cost_per_request": 0.5}
    return _network("Customer-support centers", "Four centers; all work can be rerouted between centers or done from home.",
                    sites, services, demand, dependencies, settings)


def field_service_network() -> dict:
    """Five field-service branches: work happens at the customer's premises and cannot be shared."""
    sites = [
        ("Charlotte Branch",   "Charlotte",   "NC", 18, 8, "Mon-Sat", 0.0, 0.25, 44, True),
        ("Nashville Branch",   "Nashville",   "TN", 15, 8, "Mon-Sat", 0.0, 0.25, 42, True),
        ("Minneapolis Branch", "Minneapolis", "MN", 16, 8, "Mon-Sat", 0.0, 0.25, 46, True),
        ("Denver Branch",      "Denver",      "CO", 14, 8, "Mon-Sat", 0.0, 0.25, 45, True),
        ("Sacramento Branch",  "Sacramento",  "CA", 15, 8, "Mon-Sat", 0.0, 0.25, 52, True),
    ]
    services = [
        ("Emergency repair",       2.5, 0, 1, 420, 0,  False, 1),
        ("Scheduled maintenance",  1.5, 3, 10, 190, 40, False, 2),
        ("Remote diagnostics",     0.5, 1, 5,  60, 10, True,  3),
    ]
    per_staff = {"Emergency repair": 1.1, "Scheduled maintenance": 2.2, "Remote diagnostics": 1.6}
    demand = [(s[0], svc, round(s[3] * rate, 1)) for s in sites for svc, rate in per_staff.items()]
    dependencies = [(s[0], "Dispatch and scheduling software", 0.60) for s in sites] + [(s[0], "Parts distributor", 0.40) for s in sites]
    settings = {"start_date": "2026-01-05", "weekday_factor": [1.1, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
                "temp_pool_hours_per_day": 80, "temp_lead_days": 4, "temp_cost_multiplier": 1.6, "temp_productivity": 0.75,
                "handoff_cost_per_request": 4.0}
    return _network("Field-service branches", "Five branches; repairs and maintenance happen at the customer's premises, so only "
                    "remote diagnostics can be shared between branches.", sites, services, demand, dependencies, settings)


SAMPLE_NETWORKS = {
    "Urgent-care clinics": urgent_care_network,
    "Customer-support centers": support_center_network,
    "Field-service branches": field_service_network,
}


def presets(network: dict, start_day: int = 15) -> dict[str, list[ServiceScenario]]:
    """Named disruptions that fit this network, each a list of one or more scenarios.

    Regional presets appear only if the network has a site in that region,
    and the vendor preset only if the network has vendor dependencies.
    """
    regions = set(network["sites"]["region"])
    out: dict[str, list[ServiceScenario]] = {}
    if "South" in regions:
        out["Hurricane landfall (South)"] = [
            ServiceScenario("site outage", "South", start_day, 5, 1.0, "Sites in the South closed for 5 days"),
            ServiceScenario("staff shortage", "South", start_day + 5, 7, 0.25, "A quarter of staff in the South unavailable for the following week"),
            ServiceScenario("demand surge", "South", start_day + 5, 10, 1.3, "Demand in the South 30% higher for 10 days after reopening"),
        ]
    if "Midwest" in regions or "Northeast" in regions:
        hit = "Midwest" if "Midwest" in regions else "Northeast"
        out[f"Winter storm ({hit})"] = [
            ServiceScenario("site outage", hit, start_day, 3, 0.8, f"Sites in the {hit} lose 80% of on-site capacity for 3 days")]
    if "West" in regions:
        out["Wildfire smoke (West)"] = [
            ServiceScenario("site outage", "West", start_day, 4, 0.6, "Sites in the West lose 60% of on-site capacity for 4 days"),
            ServiceScenario("staff shortage", "West", start_day, 4, 0.15, "15% of staff in the West unavailable for 4 days")]
    out["Respiratory illness wave (all sites)"] = [
        ServiceScenario("staff shortage", ALL_SITES, start_day, 21, 0.18, "18% of staff out sick everywhere for 3 weeks"),
        ServiceScenario("demand surge", ALL_SITES, start_day, 21, 1.15, "Demand 15% higher everywhere for 3 weeks")]
    deps = network.get("dependencies")
    if deps is not None and len(deps):
        vendor = deps.groupby("vendor")["capacity_share"].sum().idxmax()
        out[f"{vendor} outage"] = [ServiceScenario("vendor outage", vendor, start_day, 2, 1.0, f"{vendor} down for 2 days")]
    out["Seasonal demand peak (all sites)"] = [
        ServiceScenario("demand surge", ALL_SITES, start_day, 30, 1.35, "Demand 35% higher everywhere for 30 days")]
    return out
