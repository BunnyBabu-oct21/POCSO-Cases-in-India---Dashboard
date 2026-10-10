# Child Sexual Offence Cases in India: Animated Streamlit Dashboard

An interactive, animated dashboard built **only** from the bundled dataset
`child_sexual_offence_cases_india_verified.csv` (POCSO Act registered-case figures compiled from NCRB
*Crime in India* reports). There is no file-upload option by design.

## Project structure

```
.
├── app.py                                          # Streamlit application
├── child_sexual_offence_cases_india_verified.csv   # Dataset (must sit next to app.py)
├── requirements.txt                                # Python dependencies
└── README.md
```

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

**Requirements:** Python 3.9+ and the packages in `requirements.txt`
(streamlit, pandas, numpy, scipy, matplotlib, seaborn, plotly).

## About the dataset

The CSV has 12 rows and 11 columns. Only these are populated: `year`, `geography`, `offence_category`,
`total_cases_registered`, `source`, `notes`.

| Coverage | Details |
|---|---|
| All-India, all POCSO offences | 2020, 2021, 2022 |
| All-India, Sec 4 & 6 (penetrative) | 2020 (reported only as "more than 28,000", so a lower bound) |
| States, 2021 | Uttar Pradesh, Maharashtra, Madhya Pradesh, Tamil Nadu, Karnataka |
| States, other years | Uttar Pradesh 2022, Karnataka 2023 |
| Victim gender share | One row of **percentages** (girls 97.1-97.6%, boys 2.3-2.8%), 2017-2020 |

The columns `victims_girls`, `victims_boys`, `victims_other_gender`, `cases_solved_chargesheeted` and
`cases_convicted` are entirely empty, so nothing is charted from them.

> **Interpretation note:** the figures are *registered cases*, not victims or convictions. A rise can reflect
> more reporting and registration as well as more offences. With so few data points, treat trends as descriptive,
> not as statistical findings.

## Features

### Sidebar filters
- **Year**, **Geography** and **Offence category** (multi-select)
- **Registered cases** range slider
- **Reset filters** button
- **Frame duration (ms)** slider controlling the speed of all animated charts

### Dashboard tabs

| Tab | Charts |
|---|---|
| **Overview** | Animated **bar** chart (play through years; log-scale option), **donut** (victim gender split), **pie** (composition for a chosen year), **sunburst** (year → component) |
| **Trends** | Animated **line** chart (redraws year by year), **time series** with range slider, YoY change and a scipy linear trend, first-vs-latest change table |
| **Geography** | **Radar** (states × year), **heatmap** (seaborn, state × year) |
| **Distribution & relationships** | **Histogram** with scaled KDE curve and summary stats, animated **scatter** (rank vs cases, log scale), **scatter** of first vs latest reported year |
| **Data & notes** | Filtered table, CSV download of the filtered records, dataset caveats and sources |

### Animation
- Use **▶ Play / ⏸ Pause** or drag the year slider under the animated charts.
- KPI cards count up and charts fade in on load.
- The bar chart opens on the first year; the line chart opens fully drawn and ▶ Play replays it.

## Derived values

Everything comes from the CSV; the values below are calculated from it and labelled as derived in the app.

- **Growth / CAGR:** change between the first and last national year in view.
- **Listed states' share of national:** sum of listed states divided by the national total for the same year.
- **Rest of India:** national total minus the states listed for that year.
- **Other POCSO offences (2020):** all-offences total minus the Sec 4 & 6 value (the Sec 4 & 6 figure is a floor).
- **Gender donut:** midpoint of the reported percentage ranges; the small remainder is shown as "Unspecified / rounding".
- **Trend line:** `scipy.stats.linregress`, fitted only when a series has 3 or more reported years. It is not a forecast.

Years that have state rows but no national total are labelled "listed states only" in the composition charts.

## Behaviour with filters

- If no records match, the app shows a warning and stops; press **Reset filters**.
- Some charts need minimum data and show a message otherwise (radar needs 3 or more states; time series needs
  a series with 2 or more reported years; histogram needs 2 or more records).
- The composition charts (pie, sunburst) decompose the *All POCSO offences* totals, so they are hidden when that
  offence category is deselected.
- The donut shows fixed percentages and is not affected by filters.

## Customising

- **Colours:** edit `PALETTE` and `NAVY` near the top of `app.py`.
- **Data:** replace the CSV with another file of the same column names, keeping the same filename. Coverage notes
  in the *Data & notes* tab are partly hard-coded text and should be updated to match.
- **Animation defaults:** change the default in the `Frame duration (ms)` slider.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` error | Keep the CSV in the same folder as `app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the active environment |
| KPI cards don't animate or look blank | Some browsers or extensions block embedded scripts; the cards fall back to static metrics if the embed fails |
| Charts look stale after editing the CSV | Clear the cache: press `C` in the app, or restart Streamlit |

## Help for a child in danger (India)

CHILDLINE **1098** is the national 24×7 child helpline.

## Credits

Data: NCRB *Crime in India* (2020-2023) figures as compiled in the bundled CSV, via the media reports named in its
`source` column. Built with Streamlit, Plotly, pandas, NumPy, SciPy, Matplotlib and Seaborn.
