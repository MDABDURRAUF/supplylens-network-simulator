"""Command line for the services model.

    network-sim list
    network-sim run --network "Urgent-care clinics" --preset "Hurricane landfall (South)" --brief brief.html
    network-sim export --network "Customer-support centers" --dir my_network
    network-sim run --folder my_network --preset "Seasonal demand peak (all sites)"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .analysis import best_strategy, compare_strategies, load_network, save_network, site_criticality
from .report import brief_summary, build_brief, money, write_brief
from .samples import SAMPLE_NETWORKS, presets
from .services import simulate_services


def _network(args) -> dict:
    if getattr(args, "folder", None):
        return load_network(args.folder)
    if args.network not in SAMPLE_NETWORKS:
        raise ValueError(f"unknown network '{args.network}'; choose from: {', '.join(SAMPLE_NETWORKS)}")
    return SAMPLE_NETWORKS[args.network]()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="network-sim", description="Disruption scenarios for US service networks.")
    p.add_argument("--version", action="version", version=f"network-sim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the sample networks and the disruption presets for each")

    r = sub.add_parser("run", help="compare every response plan against one disruption")
    r.add_argument("--network", default="Urgent-care clinics", help="a sample network name")
    r.add_argument("--folder", help="a network folder written by 'export' (overrides --network)")
    r.add_argument("--preset", required=True, help="a disruption preset name (see 'list')")
    r.add_argument("--days", type=int, default=90)
    r.add_argument("--start-day", type=int, default=15)
    r.add_argument("--brief", help="write a continuity brief to this HTML file")
    r.add_argument("--csv", help="write the comparison table to this CSV file")

    e = sub.add_parser("export", help="write a sample network as editable CSV files")
    e.add_argument("--network", default="Urgent-care clinics")
    e.add_argument("--dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            for name, make in SAMPLE_NETWORKS.items():
                net = make()
                print(f"{name}: {len(net['sites'])} sites, {net['sites']['staff'].sum():,.0f} staff. {net['description']}")
                for preset in presets(net):
                    print(f"    - {preset}")
            return 0
        network = _network(args)
        if args.command == "export":
            print(f"Wrote {save_network(network, args.dir)} (sites.csv, services.csv, demand.csv, dependencies.csv, settings.json)")
            return 0

        available = presets(network, args.start_day)
        if args.preset not in available:
            raise ValueError(f"unknown preset '{args.preset}' for this network; choose from: {', '.join(available)}")
        scenarios = available[args.preset]
        comparison, _ = compare_strategies(network, scenarios, args.days)
        normal = simulate_services(network, None, "baseline", args.days).metrics
        print(brief_summary(network, scenarios, comparison, normal))
        print()
        show = comparison[["strategy", "service_level", "abandoned", "response_cost", "total_impact", "net_benefit_vs_baseline"]]
        with pd.option_context("display.width", 200):
            print(show.to_string(index=False, formatters={
                "service_level": "{:.1%}".format, "abandoned": "{:,.0f}".format, "response_cost": money,
                "total_impact": money, "net_benefit_vs_baseline": money}))
        print(f"\nLowest total impact: {best_strategy(comparison)['strategy']}")
        if args.csv:
            comparison.to_csv(args.csv, index=False)
            print(f"Wrote {args.csv}")
        if args.brief:
            crit = site_criticality(network, days=min(args.days, 60))
            print(f"Wrote {write_brief(build_brief(network, scenarios, comparison, normal, crit), args.brief)}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
