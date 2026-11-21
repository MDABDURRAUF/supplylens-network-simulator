"""A continuity brief: what a disruption does to a service network and which response to prefer."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .analysis import best_strategy


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def recovery_text(day, scenarios) -> str:
    if day is None or pd.isna(day):
        return "not within the period simulated"
    end = max(s.end_day for s in scenarios) if scenarios else 0
    return f"day {int(day)} ({int(day) - end - 1} days after the disruption ends)" if scenarios else f"day {int(day)}"


def brief_summary(network: dict, scenarios: list, comparison: pd.DataFrame, normal: dict) -> str:
    """One paragraph: the damage if nothing is done and the response that limits it best."""
    table = comparison.set_index("strategy")
    base = table.loc["baseline"]
    best = best_strategy(comparison)
    text = (f"With no response, on-time service falls from {pct(normal['service_level'])} to {pct(base['service_level'])}, "
            f"{base['abandoned']:,.0f} customers give up waiting, and the disruption costs an estimated "
            f"{money(base['disruption_cost'])} in lost revenue and late penalties; service is back to normal on "
            f"{recovery_text(base['recovery_day'], scenarios)}.")
    if best["strategy"] == "baseline" or best["net_benefit_vs_baseline"] <= 0:
        return text + " None of the response plans pays for itself in this scenario: each costs more than the revenue it protects."
    return text + (f" The best response is {best['strategy']}: it costs {money(best['response_cost'])}, lifts on-time service to "
                   f"{pct(best['service_level'])} and leaves the business an estimated {money(best['net_benefit_vs_baseline'])} "
                   f"better off than doing nothing.")


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
tr.best td{background:#e8f6f3;font-weight:600}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""


def _table(df: pd.DataFrame, formats: dict | None = None, highlight: str | None = None) -> str:
    if df is None or df.empty:
        return '<p class="note">Nothing to show.</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = "".join(f'<td class="num">{escape(formats[c](v))}</td>' if c in formats else f"<td>{escape(str(v))}</td>"
                        for c, v in row.items())
        cls = ' class="best"' if highlight is not None and str(row.iloc[0]) == highlight else ""
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_brief(network: dict, scenarios: list, comparison: pd.DataFrame, normal: dict, criticality: pd.DataFrame | None = None,
                title: str | None = None) -> str:
    """The continuity brief as a single HTML page.

    ``comparison`` comes from ``compare_strategies``, ``normal`` is the metrics
    of a run with no disruption, and ``criticality`` (optional) comes from
    ``site_criticality``.
    """
    name = network.get("name", "Service network")
    best = best_strategy(comparison)
    base = comparison.set_index("strategy").loc["baseline"]
    body = f'<p class="lead">{escape(brief_summary(network, scenarios, comparison, normal))}</p>'
    body += '<div class="kpis">' + "".join(
        f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in [
            ("on-time service, normal", pct(normal["service_level"])), ("on-time service, no response", pct(base["service_level"])),
            ("customers lost, no response", f"{base['abandoned']:,.0f}"), ("cost of the disruption", money(base["disruption_cost"])),
            ("best response", str(best["strategy"])), ("better off by", money(best["net_benefit_vs_baseline"]))]) + "</div>"

    body += "<h2>The disruption</h2><ul>" + "".join(f"<li>{escape(s.describe())}</li>" for s in scenarios) + "</ul>"
    sites = network["sites"]
    body += (f'<p class="note">{escape(name)}: {len(sites)} sites in {sites["state"].nunique()} states, '
             f'{sites["staff"].sum():,.0f} staff, {len(network["services"])} service lines.</p>')

    body += "<h2>Response plans compared</h2>"
    cols = ["strategy", "service_level", "abandoned", "average_wait_days", "lost_revenue", "sla_penalties", "response_cost",
            "total_impact", "net_benefit_vs_baseline", "recovery_day"]
    body += _table(comparison[cols], {
        "service_level": pct, "abandoned": lambda v: f"{v:,.0f}", "average_wait_days": lambda v: f"{v:.2f}", "lost_revenue": money,
        "sla_penalties": money, "response_cost": money, "total_impact": money, "net_benefit_vs_baseline": money,
        "recovery_day": lambda v: "not observed" if v is None or pd.isna(v) else f"day {int(v)}"}, highlight=str(best["strategy"]))
    body += ('<p class="note">Total impact = lost revenue + late penalties + cost of the response. The highlighted row has the '
             "lowest total impact.</p>")

    if criticality is not None and len(criticality):
        body += "<h2>Which site matters most</h2><p class=\"note\">Each site closed in turn; extra impact above a normal period.</p>"
        body += _table(criticality[["site", "state", "extra_impact", "service_level", "abandoned", "absorbed"]],
                       {"extra_impact": money, "service_level": pct, "abandoned": lambda v: f"{v:,.0f}", "absorbed": pct})

    body += ('<h2>How to read this</h2><p class="caveat">This is a scenario, not a forecast. The network, demand, revenue per '
             "request, staffing costs and the size and length of the disruption are assumptions. Demand is taken at its expected "
             "value each day. Differences between plans are estimates under those assumptions, not measured savings.</p>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title or 'Continuity brief')}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{escape(title or 'Continuity brief: ' + name)}</h1><p class=\"note\">SupplyLens Network Simulator, services model</p>"
            f"{body}</main></body></html>")


def write_brief(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
