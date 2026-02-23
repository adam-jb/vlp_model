# Battery Storage Economics Model

Optimisation model for a home battery Virtual Power Plant (VPP) / Electricity Company business. Calculates optimal revenue across multiple streams (arbitrage, frequency response, capacity market, balancing mechanism) for several European nations.

## Quick Start

```bash
pip install -r requirements.txt
python run_model.py          # Run main model (2025 baseline)
python run_scenarios.py      # Project revenues 2026-2033 under best/base/worst cases
```


## Files which feed into model

  - data/inputs/model_input.csv - All nation parameters (costs, tariffs, FR rates, CM prices, battery specs)
  - data/inputs/use_profiles.csv - Hourly kWh consumption profile per nation
  - data/inputs/uk_market_inputs.csv - UK-specific CM de-rating factors (optional, falls back to defaults)                                                                                                                                           
  - data/inputs/capacity_forecasts.csv - BESS/renewable capacity forecast data from FES System Transformation Scenario
  - data/prices/nl_hourly.csv - Netherlands hourly wholesale prices                                                                                                                                                                                  
  - data/prices/Germany.csv - Germany hourly wholesale prices                                                                                                                                                                                      
  - data/prices/Spain.csv - Spain hourly wholesale prices
  - data/prices/United Kingdom.csv - UK hourly wholesale prices




## How It Works

The model solves 365 separate Mixed Integer Linear Programming (MILP) problems - one per day - each with 24 hourly decision variables. Each day gets its own optimal strategy based on that day's actual wholesale price curve.

### Revenue Streams

| Stream | Description |
|--------|-------------|
| **Wholesale Arbitrage** | Buy low, sell high on day-ahead market |
| **Frequency Response (FR)** | Capacity commitment to grid stability (partial day prorated) |
| **Capacity Market (CM)** | Annual de-rated capacity payment (UK uses T-4/T-1 auctions) |
| **Balancing Mechanism (BM)** | Multiplier on arbitrage when BM bids accepted |

### Two Business Models

1. **VPP (Virtual Power Plant)**: Battery earns from grid services only. Customer keeps their existing supplier.
2. **Electricity Company**: Battery serves customer demand first (margin on retail spread), leftover capacity earns grid services.

### Key Assumptions

- 90% round-trip efficiency (sqrt split per leg)
- SoC limits: 10%-90% for cycling, ~50% during FR commitment
- 8-year battery life for ROI calculations
- CM stress events rare enough that battery is usually available
- Battery size optimised from model_input minimum to ceiling in 2.5 kWh steps

## Running Scenarios

`python run_scenarios.py` projects revenues for 2026-2030 using spread compression research:

```
Spread = Floor + (S0 - Floor) * (C0/C)^alpha * (R/R0)^beta
```

| Scenario | alpha | beta | Rationale |
|----------|-------|------|-----------|
| **Best** | 0.46 | 0.4 | Slower BESS compression (German data), wind widens spreads |
| **Base** | 0.50 | 0.3 | Central estimates |
| **Worst** | 0.65 | 0.0 | Fast compression (CAISO data), no renewable benefit |

Arbitrage revenue scales with the spread ratio; FR and CM are held constant.

## Project Structure

```
elec/
├── data/
│   ├── inputs/              # Model configuration
│   │   ├── model_input.csv      Parameters by nation (costs, rates, tariffs)
│   │   ├── use_profiles.csv     Hourly consumption profiles
│   │   ├── uk_market_inputs.csv UK-specific CM de-rating data
│   │   └── capacity_forecasts.csv BESS/renewable capacity forecasts
│   ├── prices/              # Wholesale hourly price data
│   │   ├── nl_hourly.csv       Netherlands (also used as multi-nation source)
│   │   ├── United Kingdom.csv
│   │   ├── Germany.csv, Spain.csv, France.csv, Italy.csv
│   └── agile/               # UK Agile tariff cache
│       └── agile_rates_2025_cache.csv
├── models/
│   ├── battery_model.py         Main VPP/Elec Company optimiser
│   └── battery_model_agile_consumer.py  Joint consumer + VPP (UK Agile)
├── analysis/                # UK Agile tariff analysis scripts
├── investors/               # IUK investor research
├── docs/                    # Project notes, comparisons, outreach
├── results/                 # Generated output CSVs
├── run_model.py             # Entry point
├── run_scenarios.py         # Scenario projections
└── requirements.txt
```

## Data Sources

- **Wholesale prices**: Hourly day-ahead prices from ENTSO-E (via CSV exports)
- **model_input.csv**: All nation-specific parameters - consumer tariffs, FR rates, CM clearing prices, costs, battery specs
- **use_profiles.csv**: Hourly kWh consumption by nation (UK profile used as fallback if unavailable for a country - noted in output)
- **Agile rates**: Scraped from Octopus Energy Agile tariff via agilebuddy.uk
- **capacity_forecasts.csv**: BESS capacity (GWh), renewable capacity (GWh), and spread projections based on market research

## Nations Modelled

Netherlands, Germany, Spain, UK (configurable via `NATIONS` dict in `battery_model.py`).
France and Italy have price data available but are not yet configured in model_input.csv.
