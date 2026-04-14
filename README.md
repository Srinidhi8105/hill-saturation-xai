# 📈 Hill Saturation Modelling with Explainable AI

> Predicting and explaining corporate R&D investment returns using
> physics-informed machine learning and SHAP-based explainability —
> deployed as a live interactive dashboard.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nidhi-hill-saturation-xai-cdsgves5ycgmczfkz3inai.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-ff6b6b)
![License](https://img.shields.io/badge/License-MIT-22c55e)

---

## The Problem

Every year, companies pour billions into R&D. And every year, the same
question sits uncomfortably in strategy meetings:

> *"Are we past the point where more spending actually helps?"*

Standard ML models give you a number. This system gives you an explanation —
**which features drove the score, at what spend level returns plateau,
and exactly which lever to pull next.**

---

## What is Hill Saturation?

R&D spending follows a law of diminishing returns. Your first $50M opens
enormous territory. The next $50M helps less. By $400M, each additional
dollar adds almost nothing. This is **saturation**.

The Hill function captures this mathematically:

```
innovation_score = (x^n / (K^n + x^n)) × MaxResponse
```

- **x** — R&D spend
- **K** — half-saturation constant: the spend level where you hit 50% of
  your maximum possible output
- **n** — shape parameter: controls curve steepness

If you are spending **below K** → you are in the high-return zone.
If you are spending **above K** → each additional dollar returns almost nothing.

---

## System Architecture

```
Synthetic Dataset (500 companies × 4 industries)
            ↓
    Exploratory Data Analysis
            ↓
  Layer 1: Hill Curve Fitting          SciPy curve_fit per industry
            ↓
  Layer 2: Gradient Boosting           scikit-learn GBM (R² = 0.97)
            ↓
    SHAP Explainability Layer          TreeSHAP — global + local
            ↓
    Streamlit Dashboard                Live inference + explanations
            ↓
    Streamlit Community Cloud          Permanent public URL
```

---

## Dataset

Synthetically generated to encode the Hill saturation effect directly
into the target variable. 500 companies × 8 features × 1 target.

| Feature | Description | Range |
|---|---|---|
| `rd_spend` | Annual R&D expenditure | $1M – $500M |
| `company_size` | Number of employees | 50 – 100,000 |
| `industry_maturity` | How established the industry is | 0.0 – 1.0 |
| `talent_density` | Proportion of skilled workforce | 0.05 – 0.85 |
| `collaboration_index` | External research partnerships | 0.0 – 1.0 |
| `prior_patents` | Patents filed previous year | 0 – 300 |
| `industry_type` | Sector | tech / pharma / manufacturing / energy |
| `innovation_score` | **Target** — composite innovation output | 0 – 100 |

### How the target is built

```
innovation_score =
    Hill(rd_spend, K, n)
    × (1 + 0.4 × talent_density)     talent amplifies returns
    × (1 + 0.2 × collaboration)      collaboration extends saturation
    × (1 - 0.3 × maturity)           mature industries cap ceiling
    × 100
    + 8 × (prior_patents / 300)       momentum baseline
    + noise
```

### Industry Hill parameters

| Industry | K — Half-Saturation ($M) | n — Shape | Maturity |
|---|---|---|---|
| Tech | 180 | 1.8 | Emerging |
| Pharma | 220 | 2.2 | Growing |
| Manufacturing | 120 | 1.5 | Mature |
| Energy | 150 | 1.6 | Mature |

---

## Model

### Two-layer pipeline

**Layer 1 — Hill Curve Fitting per Industry**
Fits the Hill function to each industry independently using SciPy
nonlinear least squares. Recovers K and n. The output (`hill_base_score`)
becomes an engineered feature — giving the GBM direct access to the
saturation transformation.

**Layer 2 — Gradient Boosting Regressor**
Trained on all 8 features. Learns how talent, collaboration, and
maturity modulate the Hill baseline.

```python
GradientBoostingRegressor(
    n_estimators  = 300,
    learning_rate = 0.05,
    max_depth     = 4,
    subsample     = 0.8
)
```

### Performance

| Metric | Score |
|---|---|
| R² | 0.97 |
| RMSE | ~3.8 / 100 |
| Train size | 400 companies |
| Test size | 100 companies |

---

## Explainability (SHAP)

Four visualisations surface the model's reasoning:

### Global Summary (Beeswarm)
Feature importance across all test companies. Confirms `hill_base_score`
and `rd_spend` dominate, with `talent_density` as the strongest modulator.

### Waterfall Plot
Per-company breakdown — how each feature contributed to that exact
prediction, from baseline to final score.

### Dependence Plot — `rd_spend`
SHAP values for `rd_spend` plotted across its full range. The saturation
effect is directly visible: contributions flatten above ~$180–200M.
Coloured by `talent_density` to reveal the interaction.

### Partial Dependence Plots
Marginal effect of each feature in isolation. Confirms the Hill shape
for spend, monotonic amplification for talent, and structural ceiling
for industry maturity.

---

## Dashboard — 4 Pages

| Page | What you see |
|---|---|
| 🏠 Overview | Dataset stats, scatter plots, score distributions |
| 🔮 Predict & Explain | Adjust sliders → live score + SHAP waterfall |
| 🌍 SHAP Global | Beeswarm + feature importance bar chart |
| 📉 Saturation Explorer | Hill curves + marginal return curve for your company |

---

## Key Findings

**1. The model learned the Hill function without being told.**
The PDP for `rd_spend` traces a smooth saturation curve matching
the ground truth K values used to generate the data.

**2. Talent density is the highest-leverage intervention.**
At identical spend levels, high-talent companies consistently
outperform. No ceiling visible across the feature's range.

**3. The saturation point is financially quantifiable.**
Tech companies see diminishing returns above ~$180M.
Manufacturing above ~$100M. This is directly actionable.

**4. Industry maturity is a structural ceiling.**
No combination of spend or talent can fully overcome it.

**5. Collaboration extends the high-return zone.**
External partnerships sustain innovation at spend levels where
purely internal R&D would already be saturated.

---

## Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/hill-saturation-xai.git
cd hill-saturation-xai

# Install
pip install -r requirements.txt

# Launch
streamlit run streamlit_app.py
```

Open `http://localhost:8501`

---

## Repository Structure

```
hill-saturation-xai/
│
├── streamlit_app.py                         Main dashboard
├── requirements.txt                         Dependencies
├── hill_gb_model.pkl                        Trained GBM model
├── hill_params.pkl                          Fitted Hill parameters
├── rd_innovation_dataset_with_features.csv  Dataset
└── README.md                                This file
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python · NumPy · Pandas · SciPy · scikit-learn |
| Explainability | SHAP (TreeSHAP) |
| Visualisation | Matplotlib |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Development | Google Colab · Google Drive |

---

## License

MIT — free to use, modify, and distribute.

---

*Built as part of an Explainable AI coursework project.*
