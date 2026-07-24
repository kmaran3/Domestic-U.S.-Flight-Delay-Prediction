# Domestic U.S. Flight Delay Prediction

## Project Overview

Collaborated with a team of five to build a flight delay prediction platform that uses machine learning to estimate departure delay times and cancellation probabilities for domestic U.S. flights. The goal was to move beyond static historical averages and apply ensemble learning methods to produce granular, route-level delay predictions — broken down by airline, origin airport, and month of travel — and surface those predictions through an interactive D3.js web visualization.

## The Problem with Existing Delay Information

Airlines and airports report aggregate on-time performance statistics, but these numbers are difficult for individual travelers to act on. The publicly available data suffers from:

- **Over-aggregation** — DOT statistics report airline-level or airport-level on-time percentages, but a traveler flying Southwest out of Denver in July faces a fundamentally different delay profile than someone flying Delta out of Atlanta in March. Aggregate statistics mask route-level variance entirely
- **No forward-looking estimates** — historical on-time percentages tell you what happened last year, not what to expect for your specific upcoming flight. There is no publicly available tool that predicts delay time for a specific airline + airport + month combination
- **No cancellation context** — on-time performance and cancellation risk are reported separately, if at all. A route with low average delay but 5% cancellation probability has a very different risk profile than one with moderate delay and near-zero cancellation
- **No comparative view** — if you know which airport you're flying out of but haven't chosen an airline, there's no easy way to compare which carriers historically perform best from that specific origin

Our platform was built to address all four limitations by training a Random Forest Regressor on 5 years of BTS flight data and delivering predictions through an interactive web tool.

## Data Source & Scale

All data is sourced from the United States Bureau of Transportation Statistics (BTS) Flight Delay and Cancellation Dataset, covering domestic U.S. flights from 2019 through 2023.

**Raw Dataset:**
- Source: [Kaggle — Flight Delay and Cancellation Dataset 2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
- Covers every domestic flight operated by major U.S. carriers across 5 years
- Key fields: `FLIGHT_DATE`, `AIRLINE_CODE`, `ORIGIN_CODE`, `DEP_DELAY` (departure delay in minutes), `CANCELLED` (binary)

**Processed Output:**
- 23,675 unique airline-airport-month combinations with predicted delay times and cancellation probabilities
- 18 airlines spanning major carriers (American, Delta, United, Southwest) and regional operators (Endeavor Air, SkyWest, Republic Airlines, etc.)
- 380 origin airports across the continental U.S., Alaska, and Hawaii
- 2,268 unique airline-airport route pairs
- 63 of the busiest airports mapped with geographic coordinates for the interactive visualization

## Model Architecture — Random Forest Regressor

### Why Random Forest?

Flight delay prediction is a regression problem where the target variable (departure delay in minutes) is continuous, noisy, and influenced by categorical features with high cardinality. Random Forest Regressor was selected because:

- **Handles categorical features naturally** — After one-hot encoding, airline codes (18 categories) and airport codes (380 categories) create a wide, sparse feature matrix. Random Forest handles this efficiently because each tree only considers a random subset of features at each split, preventing any single airport or airline from dominating the model
- **Robust to noise and outliers** — Flight delay data is inherently noisy. A single severe weather event can produce a 300+ minute delay that doesn't reflect the typical experience. Random Forest averages predictions across hundreds of independent trees, which dampens the influence of outlier training examples
- **No feature scaling required** — Unlike SVMs or neural networks, tree-based methods are invariant to feature magnitude. This simplifies the pipeline since one-hot encoded binary features and raw month integers (1-12) can be used without normalization
- **Implicit feature interaction** — A single decision tree can capture interactions like "Delta at ATL in July has different delay patterns than Delta at ATL in January" through sequential splits. The forest ensemble aggregates these interactions across many trees for a more stable estimate
- **Resistant to overfitting on tabular data** — Each tree is trained on a bootstrap sample of the data and considers a random subset of features at each split. This dual source of randomness (bagging + feature subsampling) provides built-in regularization that prevents the model from memorizing training noise

### How Random Forest Regression Works (Mechanically)

Random Forest builds an ensemble of decision trees in parallel, then averages their predictions:

1. **Bootstrap sampling** — For each of the N trees, draw a random sample (with replacement) from the training data. Each tree sees a slightly different version of the dataset
2. **Feature subsampling** — At each split in each tree, only consider a random subset of features (typically sqrt(total_features) for classification, total_features/3 for regression). This decorrelates the trees
3. **Recursive splitting** — Each tree recursively partitions the feature space by choosing the split that minimizes mean squared error (MSE) in the resulting child nodes
4. **Leaf prediction** — When a leaf node is reached, the prediction is the mean of all training samples that landed in that leaf
5. **Ensemble averaging** — The final prediction is the average of all N trees' predictions, which reduces variance compared to any single tree

The key insight is that while individual trees are high-variance (they overfit), averaging many decorrelated trees produces a low-variance estimator. The bootstrap sampling and feature subsampling ensure the trees are sufficiently different from each other.

### Model Configuration

```python
delay_model = RandomForestRegressor().fit(X_train, y_delay_train)
```

Scikit-learn defaults were used:
- `n_estimators=100` (100 trees in the forest)
- `max_depth=None` (trees grow until leaves are pure or contain fewer than `min_samples_split` samples)
- `min_samples_split=2`
- `max_features='auto'` (uses n_features/3 for regression)
- `random_state=42` for reproducibility
- 80/20 train-test split

### Feature Engineering

The model uses 3 input features, each one-hot encoded:

| Feature | Type | Cardinality | Description |
|---------|------|-------------|-------------|
| `AIRLINE_CODE` | Categorical | 18 | Two-letter IATA carrier code (AA, DL, UA, WN, etc.) |
| `ORIGIN_CODE` | Categorical | 380 | Three-letter IATA airport code (ATL, ORD, DEN, etc.) |
| `Month_of_Travel` | Ordinal | 12 | Calendar month (1-12), extracted from `FLIGHT_DATE` |

After one-hot encoding via `pd.get_dummies()`, the feature matrix expands to ~410 columns. The month feature captures seasonal delay patterns (summer thunderstorm season, winter weather, holiday travel surges), while airline and airport codes capture carrier-specific operational tendencies and airport-specific congestion patterns.

### Target Variables

Two prediction targets are generated from the same model pipeline:

1. **Predicted Delay Time** (continuous, minutes) — Trained using `DEP_DELAY` as the target. Predictions are grouped by airline-airport-month and averaged, with negative values (early departures) floored to 0: `max(mean_prediction, 0)`
2. **Cancellation Probability** (continuous, 0-1) — Computed as the empirical cancellation rate per airline-airport-month group: `mean(CANCELLED)` where `CANCELLED` is binary (0 or 1)

### Post-Processing & Aggregation

Raw per-flight predictions are aggregated to produce one prediction per airline-airport-month combination:

```python
grouped_df = df.groupby(['AIRLINE_CODE', 'ORIGIN_CODE', 'Month_of_Travel']).agg({
    'Predicted_Delay_Time': lambda x: max(x.mean(), 0),
    'CANCELLED': 'mean'
}).reset_index()
```

This aggregation serves two purposes:
- **Smoothing** — Individual flight predictions are noisy; grouping and averaging produces more stable estimates that better represent the typical experience
- **Practical utility** — A traveler doesn't need per-flight predictions; they need to know "if I fly American out of Atlanta in March, what delay should I expect on average?"

The output CSV contains 23,675 rows, each representing a unique airline-airport-month combination with a predicted delay time and cancellation probability.

## Model Comparison & Selection

We benchmarked four model configurations to determine which approach best balances accuracy and generalization for delay prediction:

### Classification Models (Delay > 15 Minutes)

These models frame the problem as binary classification: will the flight be delayed more than 15 minutes (1) or not (0)?

| Model | Features | Task | Encoding |
|-------|----------|------|----------|
| Random Forest Classifier (2 features) | `AIRLINE_CODE`, `ORIGIN_CODE` | Binary delay classification | OneHotEncoder |
| Random Forest Classifier (3 features) | `Month`, `AIRLINE_CODE`, `ORIGIN_CODE` | Binary delay classification | OneHotEncoder |
| Gradient Boosting Classifier (2 features) | `AIRLINE_CODE`, `ORIGIN_CODE` | Binary delay classification | OneHotEncoder |
| Gradient Boosting Classifier (3 features) | `Month`, `AIRLINE_CODE`, `ORIGIN_CODE` | Binary delay classification | OneHotEncoder |

These classification models were used for experimentation and testing — they answer "will this flight be delayed?" but not "by how much?" The 15-minute threshold aligns with the DOT's official definition of a delayed flight.

### Regression Model (Continuous Delay Time)

The production model uses Random Forest Regressor with 3 features to predict continuous delay time in minutes, which provides more actionable information than a binary yes/no classification.

**Why regression over classification:**
- A binary prediction ("delayed" vs. "not delayed") loses critical information. A 16-minute delay and a 3-hour delay are both "delayed," but the traveler response is completely different
- Continuous predictions enable ranking — users can compare airlines by expected delay magnitude, not just delay probability
- The visualization tool surfaces the actual predicted delay time (e.g., "10.62 minutes"), which is more useful for planning than a probability of being delayed

**Why Random Forest over Gradient Boosting for production:**
- Gradient Boosting Classifier was tested for the classification variant but builds trees sequentially (each tree corrects the previous ensemble's errors), which makes it slower to train on the full dataset
- Random Forest builds trees in parallel and is more resistant to overfitting when using default hyperparameters without tuning — important since the production model uses scikit-learn defaults
- For the regression task specifically, Random Forest Regressor's averaging behavior produces smoother predictions across airline-airport-month groups, which is desirable for a user-facing tool

## Visualization Architecture

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Visualization | D3.js v5 | Interactive US map, bar charts, tooltips |
| Map projection | d3-geo (AlbersUSA) | Geographic projection for continental US + Alaska/Hawaii |
| Geospatial data | TopoJSON | US state boundaries (`usa_map.json`) |
| Tooltips | d3-tip | Hover/click-activated bar chart tooltips on map |
| Styling | CSS3 | Layout, color theming, responsive positioning |
| Server | Python `http.server` | Local HTTP server for serving static files |

### Why D3.js Over Chart.js, Plotly, or Leaflet

- **D3 provides full control over SVG rendering** — the US map visualization requires custom geographic projections (AlbersUSA), state boundary rendering from TopoJSON data, and precisely positioned circle markers at airport coordinates. Higher-level charting libraries abstract away this control
- **Tooltip interactivity** — hovering over an airport circle triggers a dynamic bar chart showing the top 5 lowest-delay airlines. This requires D3's data-binding pattern (`.data().enter().append()`) to dynamically create SVG elements based on filtered CSV data
- **No build step** — D3 runs directly in the browser from a script tag. The entire application is a single HTML file with CSS and inline JavaScript — no webpack, no npm, no bundler configuration. This makes the tool immediately runnable by anyone with Python installed

### Application Features

**1. Delay Prediction Tool (Left Panel)**

Users select an airline, origin airport, and month of travel from cascading dropdown menus, then click Submit to see:
- **Predicted Delay Time** — the Random Forest Regressor's predicted average departure delay in minutes for that specific combination
- **Cancellation Probability** — the historical cancellation rate for that route and month, displayed as a percentage

The dropdowns cascade dynamically: selecting an airline filters the airport dropdown to only show airports that airline serves, and selecting an airport filters the month dropdown to only show months with available data. This prevents users from selecting invalid combinations that would return no results.

**2. Interactive Airport Map (Right Panel)**

A D3-rendered US map using the AlbersUSA projection displays the 63 busiest domestic airports as interactive circle markers. Each airport is positioned at its actual geographic coordinates using latitude/longitude data from `airports.csv`.

- **Hover interaction** — moving the cursor over an airport circle triggers a tooltip containing a bar chart of the 5 airlines with the lowest expected delay from that airport
- **Click-to-pin** — clicking an airport circle pins the tooltip in place, allowing the user to study the bar chart without holding the cursor steady. Clicking another airport switches the pinned tooltip; clicking outside dismisses it
- **Airline-specific coloring** — each bar in the tooltip chart is colored with the airline's brand color (e.g., American Airlines red, Delta navy, United blue) for quick visual identification

**3. Expected Delay Bar Charts (Tooltip)**

The tooltip bar charts are generated from `expected_delay_by_airline_and_airport.csv`, which contains pre-computed expected delay values for every airline-airport pair. The visualization:
- Filters for the selected airport
- Removes airlines with zero or negative expected delay (on-time or early)
- Sorts ascending by expected delay
- Displays the top 5 lowest-delay airlines as a horizontal bar chart with labeled axes

### Data Flow

```
BTS Raw Data (2019-2023)
    |
    v
viz2_data_clean_train.py
    |-- Extracts AIRLINE_CODE, ORIGIN_CODE, Month_of_Travel, DEP_DELAY, CANCELLED
    |-- One-hot encodes categorical features
    |-- Trains RandomForestRegressor on DEP_DELAY
    |-- Generates per-flight delay predictions
    |-- Aggregates by airline-airport-month (mean delay, mean cancellation)
    |
    v
flightdata_delay_and_cancellation.csv (23,675 rows)
    |-- Used by prediction tool dropdowns
    |
expected_delay_by_airline_and_airport.csv (6,840 rows)
    |-- Used by map tooltip bar charts
    |
airports.csv (63 rows)
    |-- Airport coordinates for map markers
    |
usa_map.json (TopoJSON)
    |-- US state boundaries for map rendering
    |
    v
projectVisual.html + styles.css + D3.js
    |-- Renders interactive web application
    |-- Serves via python -m http.server 8000
```

### Deployment

The application is designed as a fully self-contained static package:
- No backend server required beyond Python's built-in HTTP server
- All data (CSVs), libraries (D3.js, d3-tip, d3-legend, TopoJSON), and assets (GeoJSON map) are bundled in the repository
- No npm install, no pip dependencies (for the visualization), no API keys
- Works offline once the local server is started

## Key Statistics

| Metric | Value |
|--------|-------|
| Training data span | 2019-2023 (5 years) |
| Total prediction rows | 23,675 |
| Airlines covered | 18 |
| Airports in dataset | 380 |
| Airports on map | 63 (busiest by traffic) |
| Unique airline-airport pairs | 2,268 |
| Average predicted delay | 11.78 minutes |
| Maximum predicted delay | 793.15 minutes |
| Routes with nonzero delay | 15,391 (65.0%) |
| Routes with nonzero cancellation | 448 (1.9%) |

## Summary

This project combines Random Forest Regression with D3.js interactive visualization to produce a flight delay prediction tool that gives travelers actionable, route-specific delay estimates. The core insight is that flight delays are not uniformly distributed — they vary significantly by airline, airport, and season — and a model trained on 5 years of granular BTS data can capture these patterns at the airline-airport-month level. The interactive map adds a comparative dimension that no existing public tool provides: instant visual comparison of the best-performing airlines at any of the 63 busiest U.S. airports, helping travelers choose not just when to fly, but who to fly with.
