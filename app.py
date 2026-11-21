"""SupplyLens Network Simulator dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import network_simulator as ns
from network_simulator.report import money, pct, recovery_text

st.set_page_config(page_title="SupplyLens Network Simulator", layout="wide")

RED, AMBER, GREEN, BLUE, GREY = "#d1495b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
PLAN_COLORS = {"baseline": GREY, "overtime": AMBER, "load sharing": BLUE, "temp staff": "#7b6fb0", "combined": GREEN}
PLAN_LABELS = {"baseline": "No response", "overtime": "Overtime", "load sharing": "Load sharing", "temp staff": "Temporary staff",
               "combined": "All three"}
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Network Simulator")
mode = st.sidebar.radio("Model", ["Service network (US)", "Goods network"])

# =============================================================== GOODS MODEL
if mode == "Goods network":
    st.title("Goods network: disruption and recovery")
    st.caption("The original SupplyLens model: five suppliers, three warehouses and ten products, all synthetic.")
    network = ns.sample_network()
    kinds = {"Supplier closure": "supplier closure", "Warehouse outage": "warehouse outage", "Transport delay": "transport delay",
             "Demand surge": "demand surge"}
    c = st.sidebar
    kind = kinds[c.selectbox("Disruption", list(kinds))]
    targets = network["suppliers"]["supplier"] if kind in ("supplier closure", "transport delay") else network["warehouses"]["warehouse"]
    target = c.selectbox("Affected supplier" if kind in ("supplier closure", "transport delay") else "Affected warehouse", list(targets))
    days = c.slider("Days simulated", 30, 180, 90, 10)
    start = c.slider("Starts on day", 1, days, min(15, days))
    duration = c.slider("Lasts (days)", 1, max(1, days - start + 1), min(28, days - start + 1))
    severity = (c.slider("Extra transit days", 1, 30, 7) if kind == "transport delay"
                else c.slider("Demand multiplier", 1.0, 5.0, 2.0, 0.1) if kind == "demand surge" else 2.0)
    scenario = ns.Scenario(kind=kind, target=target, start_day=int(start), duration=int(duration), severity=float(severity))
    labels = {"baseline": "Baseline", "backup": "Backup sourcing", "transfer": "Inventory transfers"}
    colors = {"baseline": GREY, "backup": GREEN, "transfer": AMBER}
    try:
        results = {k: ns.simulate(scenario, strategy=k, days=int(days), network=network) for k in labels}
    except ValueError as exc:
        st.error(f"Check the scenario: {exc}")
        st.stop()
    rows = [{"Plan": labels[k], "Immediate fill rate": r.metrics["fill_rate"], "Lost units": r.metrics["lost_units"],
             "Ending backlog": r.metrics["open_backorders"], "Shortage days": r.metrics["shortage_days"],
             "Procurement cost": r.metrics["procurement_cost"], "Transfer cost": r.metrics["transfer_cost"],
             "Recovery day": r.metrics["recovery_day"] if r.metrics["recovery_day"] is not None else "Not observed"}
            for k, r in results.items()]
    table = pd.DataFrame(rows)
    best = table.sort_values("Lost units").iloc[0]
    st.info(f"**{best['Plan']}** loses the fewest units ({best['Lost units']:,.0f}) with an immediate fill rate of "
            f"{best['Immediate fill rate']:.1%}.")
    st.dataframe(table, hide_index=True, column_config={"Immediate fill rate": PCT, "Procurement cost": USD, "Transfer cost": USD})
    cols = st.columns(3)
    for col, (column, title) in zip(cols, [("on_hand", "Inventory on hand"), ("open_backorders", "Open backorders"), ("lost_units", "Lost units per day")]):
        fig = go.Figure()
        for k, r in results.items():
            fig.add_scatter(x=r.daily["day"], y=r.daily[column], name=labels[k], line=dict(color=colors[k], width=2))
        fig.add_vrect(x0=start, x1=start + duration - 1, fillcolor=RED, opacity=0.07, line_width=0)
        fig.update_layout(title=title, xaxis_title="Day", legend=dict(orientation="h", y=-0.3))
        with col:
            show(fig, 320)
    st.download_button("Download comparison (CSV)", table.to_csv(index=False), "goods_network_comparison.csv", "text/csv")
    st.caption("Costs are procurement and transfer charges only and are not a like-for-like measure of savings; see the README.")
    st.stop()

# ============================================================ SERVICES MODEL
name = st.sidebar.selectbox("Network", list(ns.SAMPLE_NETWORKS))
if st.session_state.get("net_name") != name:
    st.session_state.update(net_name=name, network=ns.SAMPLE_NETWORKS[name](), net_version=0)
network = st.session_state["network"]
version = st.session_state["net_version"]
page = st.sidebar.radio("View", ["Scenario", "Stress matrix", "Sites", "Staffing", "Network data", "How it works"])
st.sidebar.caption("Illustrative network. Staffing, demand and revenue are made-up planning assumptions.")
days = st.sidebar.slider("Days simulated", 30, 180, 90, 10)


@st.cache_data(max_entries=32, show_spinner=False)
def compare(_network, key, scenarios, days):
    return ns.compare_strategies(_network, list(scenarios), days)


@st.cache_data(max_entries=16, show_spinner=False)
def normal_run(_network, key, days):
    return ns.simulate_services(_network, None, "baseline", days).metrics


@st.cache_data(max_entries=16, show_spinner=False)
def criticality(_network, key, duration, days, strategy):
    return ns.site_criticality(_network, duration=duration, days=days, strategy=strategy)


@st.cache_data(max_entries=8, show_spinner="Running every disruption against every response plan...")
def matrix(_network, key, days):
    return ns.scenario_matrix(_network, ns.presets(_network), days)


key = (name, version)
normal = normal_run(network, key, days)

# ================================================================== SCENARIO
if page == "Scenario":
    st.title(f"{name}: what a disruption costs and how to respond")
    available = ns.presets(network)
    choice = st.selectbox("Disruption", list(available) + ["Build my own"])
    if choice == "Build my own":
        c = st.columns(5)
        kind = c[0].selectbox("Type", ns.KINDS)
        deps = network["dependencies"]
        if kind == "vendor outage":
            target = c[1].selectbox("Vendor", sorted(deps["vendor"].unique()))
        else:
            sites = network["sites"]
            target = c[1].selectbox("Where", [ns.ALL_SITES] + sorted(sites["region"].unique()) + sorted(sites["state"].unique()) + sites["site"].tolist())
        start = c[2].number_input("Starts on day", 1, days, min(15, days))
        duration = c[3].number_input("Lasts (days)", 1, days, 5)
        severity = (c[4].slider("Demand multiplier", 1.0, 3.0, 1.4, 0.05) if kind == "demand surge"
                    else c[4].slider("Capacity lost", 0.05, 1.0, 1.0 if kind != "staff shortage" else 0.25, 0.05))
        scenarios = [ns.ServiceScenario(kind, target, int(start), int(duration), float(severity))]
    else:
        scenarios = available[choice]
    for s in scenarios:
        st.markdown(f"- {s.describe()}")
    try:
        table, results = compare(network, key, tuple(scenarios), days)
    except ValueError as exc:
        st.error(f"Check the scenario: {exc}")
        st.stop()
    t = table.set_index("strategy")
    base, best = t.loc["baseline"], ns.best_strategy(table)
    st.info(ns.brief_summary(network, scenarios, table, normal))

    k = st.columns(5)
    k[0].metric("On-time service", pct(base["service_level"]), f"{(base['service_level'] - normal['service_level']) * 100:+.1f} pts vs normal",
                help="Share of requests served within their target, with no response")
    k[1].metric("Customers lost", f"{base['abandoned']:,.0f}", help="Requests abandoned after waiting too long, with no response")
    k[2].metric("Cost of the disruption", money(base["disruption_cost"]), help="Lost revenue and late penalties above a normal period")
    k[3].metric("Best response", PLAN_LABELS[best["strategy"]], f"costs {money(best['response_cost'])}", delta_color="off")
    k[4].metric("Better off by", money(best["net_benefit_vs_baseline"]), help="Total impact avoided against doing nothing. An estimate.")

    left, right = st.columns(2)
    with left:
        st.subheader("What each response plan costs in total")
        parts = table.melt(id_vars="strategy", value_vars=["lost_revenue", "sla_penalties", "response_cost"], var_name="part", value_name="amount")
        parts["plan"] = parts["strategy"].map(PLAN_LABELS)
        parts["part"] = parts["part"].map({"lost_revenue": "Lost revenue", "sla_penalties": "Late penalties", "response_cost": "Cost of the response"})
        fig = px.bar(parts, x="plan", y="amount", color="part", labels={"plan": "", "amount": "Total impact ($)"},
                     color_discrete_map={"Lost revenue": RED, "Late penalties": AMBER, "Cost of the response": BLUE},
                     category_orders={"plan": [PLAN_LABELS[s] for s in ns.STRATEGIES]})
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 360)
    with right:
        st.subheader("Customers waiting, day by day")
        fig = go.Figure()
        for s, r in results.items():
            fig.add_scatter(x=r.daily["date"], y=r.daily["backlog"], name=PLAN_LABELS[s], line=dict(color=PLAN_COLORS[s], width=2))
        start_date = results["baseline"].daily["date"].iloc[0]
        for s in scenarios:
            fig.add_vrect(x0=start_date + pd.Timedelta(days=s.start_day - 1), x1=start_date + pd.Timedelta(days=s.end_day),
                          fillcolor=RED, opacity=0.06, line_width=0)
        fig.update_layout(yaxis_title="Requests waiting at end of day", legend=dict(orientation="h", y=-0.2))
        show(fig, 360)
    st.caption("Total impact = lost revenue + late penalties + cost of the response. A response is worth it only if it lowers the total. "
               "Shaded bands mark the disruption.")

    st.subheader("Response plans compared")
    view = table.assign(plan=table["strategy"].map(PLAN_LABELS))[["plan", "service_level", "abandoned", "average_wait_days", "lost_revenue",
                                                                   "sla_penalties", "response_cost", "total_impact", "net_benefit_vs_baseline",
                                                                   "protected_per_dollar", "recovery_day"]]
    st.dataframe(view, hide_index=True, column_config={
        "service_level": st.column_config.ProgressColumn("On-time service", format="percent", min_value=0, max_value=1),
        "abandoned": st.column_config.NumberColumn("Customers lost", format="%.0f"),
        "average_wait_days": st.column_config.NumberColumn("Average wait (days)", format="%.2f"),
        "lost_revenue": USD, "sla_penalties": USD, "response_cost": USD, "total_impact": USD,
        "net_benefit_vs_baseline": st.column_config.NumberColumn("Better off than no response", format="dollar"),
        "protected_per_dollar": st.column_config.NumberColumn("Protected per $1 spent", format="$%.2f"),
        "recovery_day": st.column_config.NumberColumn("Back to normal (day)", format="%d")})

    st.subheader("Where it hurts")
    plan = st.radio("Show", list(ns.STRATEGIES), format_func=PLAN_LABELS.get, horizontal=True)
    d = results[plan].detail
    by_site = d.groupby("site").agg(arrivals=("arrivals", "sum"), on_time=("served_on_time", "sum"), abandoned=("abandoned", "sum"),
                                    lost_revenue=("lost_revenue", "sum")).reset_index()
    by_site["service_level"] = by_site["on_time"] / by_site["arrivals"]
    by_site = by_site.merge(network["sites"][["site", "city", "state", "region", "lat", "lon"]], on="site")
    left, right = st.columns([3, 2])
    with left:
        fig = px.scatter_geo(by_site, lat="lat", lon="lon", size="arrivals", color="service_level", hover_name="site", scope="usa",
                             color_continuous_scale="RdYlGn", range_color=[min(0.8, by_site["service_level"].min()), 1.0],
                             hover_data={"lat": False, "lon": False, "state": True, "abandoned": ":,.0f", "service_level": ":.1%", "arrivals": ":,.0f"},
                             labels={"service_level": "On-time service"})
        fig.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)", landcolor="rgba(128,128,128,0.15)"))
        show(fig, 360)
    with right:
        by_service = d.groupby("service").agg(arrivals=("arrivals", "sum"), on_time=("served_on_time", "sum"), abandoned=("abandoned", "sum"),
                                              by_other=("served_by_other_site", "sum")).reset_index()
        by_service["service_level"] = by_service["on_time"] / by_service["arrivals"]
        st.dataframe(by_service[["service", "service_level", "abandoned", "by_other"]], hide_index=True, column_config={
            "service_level": st.column_config.ProgressColumn("On-time service", format="percent", min_value=0, max_value=1),
            "abandoned": st.column_config.NumberColumn("Customers lost", format="%.0f"),
            "by_other": st.column_config.NumberColumn("Served by another site", format="%.0f")})
        st.dataframe(by_site[["site", "state", "service_level", "abandoned", "lost_revenue"]].sort_values("service_level"), hide_index=True,
                     column_config={"service_level": PCT, "abandoned": st.column_config.NumberColumn(format="%.0f"), "lost_revenue": USD})

    with st.expander("How sure is this? Run it with random day-to-day demand"):
        st.caption("The figures above use expected demand each day. This repeats the run with demand varying at random around it.")
        c1, c2 = st.columns([1, 3])
        runs_n = c1.slider("Runs", 10, 100, 30, 10)
        if c1.button("Run"):
            runs, summary = ns.monte_carlo(network, scenarios, plan, days, n_runs=runs_n, seed=1)
            c2.dataframe(summary, hide_index=True)
            c2.caption(f"{PLAN_LABELS[plan]}: in 90% of runs, on-time service was between {pct(summary.set_index('measure').loc['service_level', 'p05'])} "
                       f"and {pct(summary.set_index('measure').loc['service_level', 'p95'])}.")

    st.subheader("Take it with you")
    dl = st.columns(3)
    crit = criticality(network, key, 5, min(days, 60), "baseline")
    dl[0].download_button("Continuity brief (HTML)", ns.build_brief(network, scenarios, table, normal, crit), "continuity_brief.html",
                          "text/html", width="stretch", type="primary")
    dl[1].download_button("Plan comparison (CSV)", table.to_csv(index=False), "plan_comparison.csv", "text/csv", width="stretch")
    dl[2].download_button("Daily detail (CSV)", results[plan].detail.to_csv(index=False), f"daily_detail_{plan.replace(' ', '_')}.csv",
                          "text/csv", width="stretch")

# ============================================================= STRESS MATRIX
elif page == "Stress matrix":
    st.title("Every disruption against every response")
    m = matrix(network, key, days)
    m["plan"] = m["strategy"].map(PLAN_LABELS)
    order = [PLAN_LABELS[s] for s in ns.STRATEGIES]
    measure = st.radio("Show", ["Total impact ($)", "On-time service"], horizontal=True)
    column, scale, fmt = ("total_impact", "Reds", ",.0f") if measure.startswith("Total") else ("service_level", "RdYlGn", ".1%")
    grid = m.pivot(index="disruption", columns="plan", values=column)[order]
    fig = px.imshow(grid, color_continuous_scale=scale, aspect="auto", text_auto=fmt, labels={"color": measure, "x": "", "y": ""})
    show(fig, 90 + 60 * len(grid))
    best = m[m["best"]][["disruption", "plan", "service_level", "total_impact", "response_cost", "net_benefit_vs_baseline"]]
    st.subheader("Best response for each disruption")
    st.dataframe(best.rename(columns={"plan": "best response"}), hide_index=True, column_config={
        "service_level": PCT, "total_impact": USD, "response_cost": USD,
        "net_benefit_vs_baseline": st.column_config.NumberColumn("Better off than no response", format="dollar")})
    never = [PLAN_LABELS[s] for s in ns.STRATEGIES if s != "baseline" and not m[(m["strategy"] == s) & m["best"]].any().any()]
    if never:
        st.caption(f"Never the cheapest overall for this network: {', '.join(never)}. They may still be chosen to protect service at a cost.")
    st.download_button("Download matrix (CSV)", m.to_csv(index=False), "stress_matrix.csv", "text/csv")

# ===================================================================== SITES
elif page == "Sites":
    st.title("Which site matters most")
    c = st.columns(2)
    duration = c[0].slider("Close each site in turn for (days)", 1, 21, 5)
    strategy = c[1].selectbox("Response", list(ns.STRATEGIES), format_func=PLAN_LABELS.get)
    crit = criticality(network, key, duration, min(days, 90), strategy)
    worst = crit.iloc[0]
    st.info(f"Losing **{worst['site']}** for {duration} days costs the most: {money(worst['extra_impact'])} above a normal period, with "
            f"{worst['abandoned']:,.0f} customers lost. {pct(worst['absorbed'], 0)} of its demand during the closure was still served on time.")
    left, right = st.columns(2)
    with left:
        fig = px.bar(crit, x="extra_impact", y="site", orientation="h", color_discrete_sequence=[RED],
                     labels={"extra_impact": "Extra impact above normal ($)", "site": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig, 340)
    with right:
        fig = px.bar(crit, x="absorbed", y="site", orientation="h", color_discrete_sequence=[GREEN],
                     labels={"absorbed": "Demand still served on time during the closure", "site": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=crit["site"].tolist()[::-1])
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        show(fig, 340)
    st.dataframe(crit, hide_index=True, column_config={"extra_impact": USD, "service_level": PCT, "absorbed": PCT,
                                                       "abandoned": st.column_config.NumberColumn("Customers lost", format="%.0f")})
    st.caption("Try 'Load sharing': it shows how much of a closed site's work the rest of the network can pick up, which depends on "
               "how much of the work can be done remotely.")

# ================================================================== STAFFING
elif page == "Staffing":
    st.title("How much staff is the right buffer")
    available = ns.presets(network)
    choice = st.selectbox("Test against", list(available))
    sweep = ns.staffing_sweep(network, available[choice], changes=tuple(np.round(np.arange(-0.15, 0.151, 0.05), 2)), days=days)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=sweep["staffing_change"], y=sweep["service_level_normal"], name="Normal period", line_color=GREY, mode="lines+markers")
        fig.add_scatter(x=sweep["staffing_change"], y=sweep["service_level_disrupted"], name="With the disruption", line_color=RED, mode="lines+markers")
        fig.update_layout(xaxis_title="Change in staff", yaxis_title="On-time service", legend=dict(orientation="h", y=-0.25))
        fig.update_xaxes(tickformat="+.0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 360)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=sweep["staffing_change"], y=sweep["net_normal"], name="Normal period", line_color=GREY, mode="lines+markers")
        fig.add_scatter(x=sweep["staffing_change"], y=sweep["net_disrupted"], name="With the disruption", line_color=RED, mode="lines+markers")
        fig.add_hline(y=0, line_color=GREY, line_dash="dot")
        fig.update_layout(xaxis_title="Change in staff", yaxis_title="Net cost against today's staffing ($)", legend=dict(orientation="h", y=-0.25))
        fig.update_xaxes(tickformat="+.0%")
        show(fig, 360)
    best_normal = sweep.loc[sweep["net_normal"].idxmin()]
    best_hit = sweep.loc[sweep["net_disrupted"].idxmin()]
    st.info(f"Over {days} days: in a normal period the cheapest staffing is **{best_normal['staffing_change']:+.0%}** against today; "
            f"if this disruption happens it is **{best_hit['staffing_change']:+.0%}**. Net cost adds the change in payroll to the change in "
            "lost revenue and late penalties, so below zero is better than today.")
    st.dataframe(sweep, hide_index=True, column_config={
        "staffing_change": st.column_config.NumberColumn(format="percent"), "labor_cost": USD, "labor_cost_change": USD,
        "service_level_normal": PCT, "service_level_disrupted": PCT, "impact_normal": USD, "impact_disrupted": USD,
        "net_normal": USD, "net_disrupted": USD})
    st.caption("Extra staff are paid every day but only earn their keep when demand exceeds capacity. This shows that trade-off for one disruption.")

# ============================================================== NETWORK DATA
elif page == "Network data":
    st.title("Network data")
    st.caption(network["description"] + " Edit the tables and apply to re-run everything with your numbers; changes last until you switch network.")
    sites = st.data_editor(network["sites"], hide_index=True, key=f"sites_{name}_{version}", disabled=["site", "region"])
    services = st.data_editor(network["services"], hide_index=True, key=f"services_{name}_{version}", disabled=["service"])
    demand = st.data_editor(network["demand"], hide_index=True, key=f"demand_{name}_{version}", disabled=["site", "service"])
    deps = st.data_editor(network["dependencies"], hide_index=True, key=f"deps_{name}_{version}", disabled=["site", "vendor"])
    c = st.columns(4)
    if c[0].button("Apply edits", type="primary"):
        edited = {**network, "sites": sites, "services": services, "demand": demand, "dependencies": deps}
        try:
            ns.validate_network(edited)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.update(network=edited, net_version=version + 1)
            st.rerun()
    if c[1].button("Reset to the sample"):
        st.session_state.update(network=ns.SAMPLE_NETWORKS[name](), net_version=version + 1)
        st.rerun()
    st.subheader("Normal period")
    k = st.columns(4)
    k[0].metric("On-time service", pct(normal["service_level"]))
    k[1].metric("Requests", f"{normal['requests']:,.0f}")
    k[2].metric("Scheduled labor cost", money(normal["scheduled_labor_cost"]))
    k[3].metric("Lost revenue and penalties", money(normal["total_impact"]))
    st.subheader("Settings")
    st.json(ns.services.settings_of(network), expanded=False)
    start = pd.Timestamp(ns.services.settings_of(network)["start_date"])
    st.subheader(f"US federal holidays, {start.year}")
    st.dataframe(ns.holiday_table(start.year), hide_index=True, column_config={
        "date": st.column_config.DateColumn(format="DD MMM YYYY"), "observed": st.column_config.DateColumn(format="DD MMM YYYY")})
    st.caption("Sites marked closed_on_federal_holidays are closed on these dates and on the observed weekday.")
    st.subheader("Download the network")
    d = st.columns(4)
    for col, table in zip(d, ("sites", "services", "demand", "dependencies")):
        col.download_button(f"{table}.csv", network[table].to_csv(index=False), f"{table}.csv", "text/csv", width="stretch")

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.markdown(
        """
**Services are not goods.** A warehouse can hold stock for a bad week. A clinic or support center cannot: staff hours
are used on the day or lost, and customers who are not served wait in a queue and eventually leave. This model
tracks hours and waiting customers instead of inventory.

**Each day**
1. Requests arrive at each site, following the weekday pattern. Sites closed that day (weekends, US federal holidays) take none.
2. Each site serves its own queue, most important service line first and oldest request first, using regular hours,
   then temporary staff, then overtime.
3. With load sharing, sites with hours to spare serve other sites' requests, but only work that can be done remotely.
4. Customers who have waited longer than their patience limit leave. Their revenue is lost.

**Disruptions** can hit one site, a state, a Census region or every site:
- *Site outage*: on-site work stops; the remote-capable share of staff keeps working on remotely-doable service lines.
- *Staff shortage*: a share of staff is unavailable.
- *Vendor outage*: sites lose the share of capacity that depends on that vendor.
- *Demand surge*: demand is multiplied.

**Responses** start on the first day of the disruption and run until the queue has cleared:
- *Overtime*: extra hours up to each site's limit, paid at time and a half by default.
- *Load sharing*: a small handoff cost per request routed to another site.
- *Temporary staff*: an agency pool, available after a lead time, billed for every hour whether used or not, and less productive.

**Total impact** = lost revenue + late penalties + cost of the response. The best response is the one with the lowest
total, which is not always the one with the best service.

**Limits.** The networks are illustrative. Demand is taken at its expected value unless you run the random-demand check.
Disruption sizes and lengths are assumptions you set, not forecasts. Travel between sites, skills, and learning curves
for temporary staff beyond a single productivity factor are not modelled. Differences between plans are scenario
estimates, not measured savings.
"""
    )
