# Battery Storage Economics Model

Optimisation model for a home battery business. Calculates optimal revenue across multiple streams (arbitrage, frequency response, capacity market, balancing mechanism) using MILP-based daily dispatch.

There are **two business models**, each with its own self-contained model and HTML report:

| Model | Directory | Command | Description |
|-------|-----------|---------|-------------|
| **Electricity Company** | `qlp_aggressive/` | `python qlp_aggressive/model.py` | Company is a licensed electricity supplier. Sets customer tariff (18–22p/kWh), buys wholesale, pays network charges + levy. Battery serves customer demand first, leftover capacity earns grid services. |
| **VLP + Agile Consumer** | `vlp_agile/` | `python vlp_agile/model.py` | Company is a Virtual Lead Party (no supply licence). Consumer stays on Octopus Agile and gets battery savings from Agile arbitrage. Company earns wholesale arb + FR + CM + BM on remaining capacity. No supply cost, no network charges, no levy. |

## Quick Start

```bash
pip install -r requirements.txt

# Run either or both models:
python qlp_aggressive/model.py    # Electricity Company model
python vlp_agile/model.py         # VLP + Agile Consumer model
```

Each model outputs to its own directory:
- **CSV files** — P&L, cashflow, and loan schedules for every scenario combination
- **report.html** — interactive report with sticky controls (CAC, staff cost, management cost) that recalculate tables and dashboards live

## How the Models Differ

| Aspect | Electricity Company (`qlp_aggressive/`) | VLP + Agile (`vlp_agile/`) |
|--------|----------------------------------------|---------------------------|
| Consumer tariff | Company sets tariff (18–22p/kWh) | Consumer stays on Octopus Agile |
| Consumer saving | Lower tariff vs competitor | Battery arbitrage on Agile rates |
| Company revenue | Customer bill + battery revenue | VLP income only (wholesale arb + FR + CM + BM) |
| Supply cost (COGS) | Wholesale cost of electricity supplied | None (company doesn't supply) |
| Network charges | TNUoS + DUoS + BSUoS + smart metering | None (consumer pays via Agile) |
| Grid levy | £95/yr | £0 |
| Scenario dimensions | 4 tariffs × 3 spreads × 3 consumptions × 4 loans = 144 | 3 spreads × 3 consumptions × 4 loans = 36 |

## Revenue Streams

| Stream | Description |
|--------|-------------|
| **Wholesale Arbitrage** | Buy low, sell high on day-ahead market |
| **Frequency Response (FR)** | Capacity commitment to grid stability (partial day prorated) |
| **Capacity Market (CM)** | Annual de-rated capacity payment (UK T-4/T-1 auctions) |
| **Balancing Mechanism (BM)** | Multiplier on arbitrage when BM bids accepted |

## How the MILP Works

Both models solve 365 separate Mixed Integer Linear Programming problems — one per day — each with 24 hourly decision variables. Each day gets its own optimal strategy based on that day's actual price curve.

The VLP + Agile model uses a **joint MILP** that splits discharge between consumer (saves Agile rate) and grid (earns wholesale rate) hour-by-hour, with FR commitment as a binary choice.

### Key Assumptions

- 15.36 kWh / 6 kW battery, £2,800 installed cost
- 90% round-trip efficiency (sqrt split per leg)
- 2% annual degradation (compound)
- One cycle per day limit
- CM stress events rare enough that battery is usually available
- Spread compression applied year-over-year using capacity forecast data

## Input Data

| File | Description |
|------|-------------|
| `data/inputs/model_input.csv` | All nation parameters (costs, tariffs, FR rates, CM prices, battery specs) |
| `data/inputs/use_profiles.csv` | Hourly kWh consumption profile per nation |
| `data/inputs/capacity_forecasts.csv` | BESS/renewable capacity forecasts (FES System Transformation) |
| `data/prices/United Kingdom.csv` | UK hourly wholesale prices |
| `data/prices/nl_hourly.csv` | Netherlands hourly wholesale prices |
| `data/prices/Germany.csv` | Germany hourly wholesale prices |
| `data/prices/Spain.csv` | Spain hourly wholesale prices |
| `data/agile/agile_rates_2025_cache.csv` | UK Octopus Agile half-hourly rates (used by VLP model) |

If no use profile is available for a country, the UK one is used and noted in output.

## Project Structure

```
elec/
├── qlp_aggressive/            # Electricity Company model
│   ├── model.py                   Self-contained model + report generator
│   ├── report.html                Interactive HTML report
│   ├── pl_*.csv, cashflow_*.csv   P&L and cashflow by scenario
│   └── chart_*.png                Static charts
├── vlp_agile/                 # VLP + Agile Consumer model
│   ├── model.py                   Self-contained model + report generator
│   ├── report.html                Interactive HTML report
│   ├── pl_*.csv, cashflow_*.csv   P&L and cashflow by scenario
│   └── loan_schedule.csv          Amortisation schedules
├── data/
│   ├── inputs/                    Model configuration CSVs
│   ├── prices/                    Wholesale hourly price data
│   └── agile/                     UK Agile tariff cache
├── models/                    # Earlier/experimental model scripts
├── analysis/                  # UK Agile tariff analysis scripts
├── investors/                 # Investor research
├── docs/                      # Project notes
└── requirements.txt
```

## Data Sources

- **Wholesale prices**: Hourly day-ahead prices from ENTSO-E (via CSV exports)
- **Agile rates**: Scraped from Octopus Energy Agile tariff via agilebuddy.uk
- **Capacity forecasts**: BESS capacity (GWh), renewable capacity (GWh) based on market research
- **model_input.csv**: All nation-specific parameters — consumer tariffs, FR rates, CM clearing prices, costs, battery specs
- **use_profiles.csv**: Hourly kWh consumption by nation (UK profile used as fallback)
