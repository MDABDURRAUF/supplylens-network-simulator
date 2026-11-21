"""SupplyLens Network Simulator: disruption scenarios for goods networks and US service networks."""

from .analysis import (
    best_strategy,
    compare_strategies,
    load_network,
    monte_carlo,
    save_network,
    scale_staff,
    scenario_matrix,
    site_criticality,
    staffing_sweep,
)
from .report import brief_summary, build_brief, write_brief
from .samples import SAMPLE_NETWORKS, field_service_network, presets, support_center_network, urgent_care_network
from .services import (
    ALL_SITES,
    KINDS,
    STRATEGIES,
    ServiceResult,
    ServiceScenario,
    resolve_sites,
    simulate_services,
    validate_network,
)
from .simulation import Result, Scenario, sample_network, simulate
from .us import CENSUS_REGIONS, federal_holidays, holiday_table, region_of

__version__ = "0.2.0"

__all__ = [
    # goods network (original model)
    "Result", "Scenario", "sample_network", "simulate",
    # service network
    "ALL_SITES", "KINDS", "STRATEGIES", "ServiceResult", "ServiceScenario", "resolve_sites", "simulate_services",
    "validate_network",
    # analysis and outputs
    "best_strategy", "compare_strategies", "monte_carlo", "scenario_matrix", "site_criticality", "staffing_sweep",
    "scale_staff", "load_network", "save_network", "brief_summary", "build_brief", "write_brief",
    # United States data and samples
    "CENSUS_REGIONS", "federal_holidays", "holiday_table", "region_of", "SAMPLE_NETWORKS", "presets",
    "urgent_care_network", "support_center_network", "field_service_network",
]
