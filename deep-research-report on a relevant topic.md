# Spatio-Temporal Graph Neural Networks for Hydrological Drought

## 1. Existing SCI/SCIE Research

Recent studies show that **Spatio-Temporal Graph Neural Networks (STGNNs)** effectively model hydrological systems because they capture both:

- **Spatial relationships** (connections between stations, wells, or grid cells), and
- **Temporal dependencies** (seasonal and long-term drought evolution).

Most studies focus on:

- Streamflow forecasting,
- Groundwater prediction,
- Soil moisture forecasting, and
- Drought index prediction.

However, **very few studies directly address hydrological drought prediction using STGNNs**, indicating a promising research opportunity.

---

## 2. Important SCI/SCIE Papers

| Year | Authors | Journal | Application | Model |
|--------|------------|------------|------------------------------|----------------------------|
| 2023 | Yu et al. | *Journal of Hydroinformatics* | Drought category prediction (SPEI) | MSTSN (GNN + GRU + Attention) |
| 2023 | Bai & Tahmasebi | *Journal of Hydrology* | Groundwater forecasting | GNN + Temporal CNN |
| 2024 | Taccari et al. | *Scientific Reports* | Groundwater level prediction | Modified MTGNN |
| 2025 | Akkala et al. | *Hydrology* | Streamflow forecasting | GCN + LSTM |
| 2025 | Akkala et al. | *Hydrology* | Streamflow forecasting with Snow Water Equivalent (SWE) | Multivariate STGNN |
| 2026 | Szatmári et al. | *Journal of Hydroinformatics* | River flow forecasting | Graph Convolutional Recurrent Network (GCRN) |
| 2026 | Wang et al. | *Water* | Soil moisture prediction | CTA-GraphConvLSTM |

---

## 3. Common Methodology

### Data Sources

Common datasets used in STGNN-based hydrological studies include:

- Streamflow monitoring stations (USGS, NWIC, CWC),
- Climate datasets (precipitation, temperature, evapotranspiration),
- Soil moisture products (LandBench, SMAP),
- Groundwater monitoring networks.

### General STGNN Pipeline

```text
Hydro-climatic Data
          ↓
Graph Construction
(nodes = stations/grid cells,
edges = distance or river connectivity)
          ↓
Spatial Learning
(GCN / GAT / ChebConv)
          ↓
Temporal Learning
(LSTM / GRU / Attention)
          ↓
Hydrological Drought Prediction
```

### Evaluation Metrics

Most studies evaluate model performance using:

- RMSE (Root Mean Square Error),
- MAE (Mean Absolute Error),
- R² (Coefficient of Determination),
- NSE (Nash–Sutcliffe Efficiency),
- KGE (Kling–Gupta Efficiency),
- F1-score (for drought classification tasks).

---

## 4. Research Gaps

The current literature reveals several important limitations:

### Lack of direct hydrological drought studies

Most STGNN research focuses on **streamflow forecasting** or **groundwater prediction**, rather than explicitly predicting **hydrological drought indices**.

### Static graph structures

Many studies rely on fixed, distance-based graphs that may not accurately represent changing hydrological relationships over time.

### Limited input variables

Several models exclude important hydro-climatic drivers such as:

- Rainfall,
- Evapotranspiration,
- Soil moisture,
- Reservoir storage,
- Large-scale climate indices.

### Difficulty in predicting extremes

Existing models often struggle to capture **severe drought events**, partly because extreme conditions occur less frequently in historical datasets.

### Limited Indian applications

Very few SCI/SCIE studies investigate STGNN-based hydrological drought prediction in **Indian river basins**, highlighting a significant research opportunity.

---

# Potential BTP/Thesis Direction

> **Develop a Spatio-Temporal Graph Neural Network framework for hydrological drought prediction in Indian river basins using river discharge and atmospheric variables.**

### Possible Input Variables

- River discharge,
- Rainfall,
- Temperature,
- Relative humidity,
- Evapotranspiration,
- Soil moisture (if available).

### Potential Target Variables

- **SDI (Streamflow Drought Index),**
- **SSI (Standardized Streamflow Index),** or
- **Hydrological drought categories.**

---

# Key Takeaway

> **Spatio-Temporal Graph Neural Networks have demonstrated strong performance in hydrological forecasting. However, their application to hydrological drought prediction—particularly within Indian river basins—remains largely unexplored. Addressing this gap offers a promising direction for novel SCI/SCIE research.**