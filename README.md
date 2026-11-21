# SupplyLens Network Simulator

**Stress-test a network against a disruption and compare what each response would cost and protect.**
Two models: a **service network** model for US service businesses (clinics, support centers, field service) and
the original **goods network** model (suppliers, warehouses, inventory).

All networks and figures included are illustrative. Results are scenario estimates under stated assumptions,
not forecasts of a real disruption and not measured savings.

## Why a services model

Most of the US economy is services. Private services-producing industries were 73% of US GDP in the third
quarter of 2025 ([BEA data via FRED](https://fred.stlouisfed.org/data/VAPGDPSPI)), and service-providing
industries have accounted for roughly 81% of nonfarm payroll employment since 2000
([BLS](https://www.bls.gov/opub/ted/2024/a-look-at-a-long-term-trend-for-the-bureaus-birthday.htm)).

Supply chain tools are built around inventory, and a service business has none. A clinic, a support center or
a repair branch cannot stock up before a storm: staff hours are used on the day or lost, and customers who are
not served wait and then leave. The services model tracks **hours and waiting customers** instead of stock.

## What the services model answers

| Question | Where |
|---|---|
| What does this disruption cost us if we do nothing? | **Scenario** view; `simulate_services` |
| Which response is worth paying for: overtime, sharing work between sites, temporary staff, or all three? | **Scenario** view; `compare_strategies` |
| Which response wins for each kind of event? | **Stress matrix** view; `scenario_matrix` |
| Which site can we least afford to lose, and how much of its work can the others pick up? | **Sites** view; `site_criticality` |
| Is extra staff a cheaper buffer than responding after the fact? | **Staffing** view; `staffing_sweep` |
| How much does day-to-day randomness change the answer? | random-demand check; `monte_carlo` |

Outputs: a printable **continuity brief** (HTML), the plan comparison and daily detail as CSV, and a command line
that produces them without opening the dashboard.

## United States focus

- **Geography.** Sites carry a state; disruptions can target a site, a state, a US Census region or every site.
  All 50 states and DC are mapped to the four Census regions.
- **Calendar.** Sites follow opening days (Mon–Fri, Mon–Sat, every day) and, if set, close on the eleven US
  federal holidays, including the observed weekday when a holiday falls at the weekend.
- **Event presets** shaped like US events: hurricane landfall in the South, a winter storm in the Midwest or
  Northeast, wildfire smoke in the West, a respiratory illness wave, an outage at a critical vendor, and a
  seasonal demand peak. Their sizes and lengths are editable assumptions.
- **Labor.** Overtime is costed at time and a half by default, the overtime rate the Fair Labor Standards Act
  requires for covered non-exempt employees; temporary staff are billed at an agency multiplier.
- **Three sample networks**: urgent-care clinics (6 sites), customer-support centers (4), field-service
  branches (5), each in several states.

## Example

Urgent-care clinics, hurricane landfall in the South (three clinics closed for five days, then a week of staff
shortage and higher demand), 90 days simulated:

| Response | On-time service | Customers lost | Cost of response | Total impact | Better off than no response |
|---|---|---|---|---|---|
| No response | 92.6% | 3,084 | $0 | $240,190 | |
| Overtime | 94.7% | 1,835 | $54,816 | $228,837 | $11,354 |
| **Load sharing** | 95.5% | 1,103 | $7,778 | **$162,142** | **$78,049** |
| Temporary staff | 94.4% | 1,902 | $91,552 | $266,835 | -$26,644 |
| All three | 98.6% | 851 | $108,246 | $248,694 | -$8,503 |

Routing telehealth visits and follow-ups to clinics in other regions is cheap and protects the most. Doing
everything gives the best service but costs more than it protects. These numbers come from a made-up network
and show the method, not a finding about any real organization.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-network-simulator.git
cd supplylens-network-simulator
pip install -e ".[app,dev]"
streamlit run app.py
```

Command line:

```bash
network-sim list                                   # sample networks and the disruptions for each
network-sim run --preset "Hurricane landfall (South)" --brief brief.html --csv comparison.csv
network-sim export --network "Customer-support centers" --dir my_network     # editable CSV files
network-sim run --folder my_network --preset "Seasonal demand peak (all sites)"
```

Python:

```python
import network_simulator as ns

network = ns.urgent_care_network()
scenarios = ns.presets(network)["Hurricane landfall (South)"]

comparison, results = ns.compare_strategies(network, scenarios, days=90)
print(ns.best_strategy(comparison))

# Or build a disruption yourself: half of Florida's on-site capacity lost for four days from day 20.
own = ns.ServiceScenario("site outage", "FL", start_day=20, duration=4, severity=0.5)
print(ns.simulate_services(network, own, strategy="load sharing").metrics)

print(ns.site_criticality(network, duration=5))
```

## How the services model works

A network is five tables (see `examples/urgent_care_network/`):

| Table | Holds |
|---|---|
| `sites` | Site, state, staff, hours per staff day, opening days, share of staff who can work remotely, overtime limit, labor cost, whether it closes on federal holidays |
| `services` | Service line, hours per request, service-level target (days), patience (days before a waiting customer leaves), revenue, late penalty, whether it can be done remotely, priority |
| `demand` | Average daily requests per site and service line |
| `dependencies` | Share of a site's capacity that depends on each outside vendor |
| `settings` | Start date, weekday demand pattern, overtime and temporary-staff terms, handoff cost |

Each simulated day:

1. Requests arrive at open sites, following the weekday pattern.
2. Each site serves its own queue, highest-priority service line first and oldest request first, using regular
   hours, then temporary staff, then overtime.
3. With load sharing, sites with hours left serve other sites' requests, for remotely-doable service lines only.
4. Customers who have waited longer than their patience leave, and their revenue is lost.

Disruptions: **site outage** (on-site work stops; the remote-capable share of staff keeps working on
remotely-doable lines), **staff shortage**, **vendor outage** (sites lose the dependent share of capacity) and
**demand surge**. Several can be combined in one run.

    total impact = lost revenue + late penalties + cost of the response

The preferred response is the one with the lowest total impact, which is often not the one with the best service.

| Measure | Definition |
|---|---|
| On-time service | Requests served within their target, divided by all requests |
| Customers lost | Requests abandoned after waiting past their patience |
| Cost of the disruption | Total impact minus the total impact of a normal period |
| Better off than no response | Total impact with no response minus total impact with the response |
| Back to normal | First of seven consecutive days after the disruption with no late service and no abandonment |

## The goods network model

The original model is included unchanged (`simulate`, `Scenario`, `sample_network`) and available in the
dashboard under **Goods network**. It simulates five suppliers, three warehouses and ten products day by day
under a supplier closure, warehouse outage, transport delay or demand surge, comparing a baseline with backup
sourcing and inventory transfers. Replenishment targets 14 days of demand; unmet demand is split equally
between lost sales and backorders; transfers use stock above a nine-day reserve, arrive next day and cost 0.50
per unit. Its costs cover procurement and transfers only and are not a like-for-like measure of savings.

```python
from network_simulator import Scenario, simulate
result = simulate(Scenario(kind="supplier closure", target="Supplier A", start_day=15, duration=28), strategy="backup")
print(result.metrics)
```

## Limits

- The sample networks are illustrative, and every disruption size and length is an assumption.
- Demand is taken at its expected value each day unless the random-demand check is used; only demand is random
  in that check.
- Staff are interchangeable within a site; skills, licensing across state lines, travel between sites and
  ramp-up of temporary staff beyond one productivity factor are not modelled.
- Temporary staff are billed for every hour deployed, used or not, which makes them look expensive for short events.
- Revenue per request and late penalties are the only customer costs; reputation and repeat business are not counted.

## Tests

```bash
pytest
```

Thirty-five tests: the original goods model's twelve, plus twenty-three covering the US calendar and regions, queue behaviour checked
against a hand-worked example, each disruption and response, the analyses, file round trips and the command line.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): predict and explain supply chain disruptions
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
