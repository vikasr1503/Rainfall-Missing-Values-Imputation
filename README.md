# STGNN for Hydrological Drought Prediction

## Overview

This project aims to develop a **Spatio-Temporal Graph Neural Network (STGNN)** framework for **Hydrological Drought Prediction** by integrating multiple hydro-meteorological datasets and modeling their spatial and temporal dependencies across river basins.

The framework combines:

- River Discharge Data
- Atmospheric Variables
- Groundwater Data
- Reservoir Storage Data
- River Network / Subbasin Connectivity

to predict future hydrological drought conditions using graph-based deep learning techniques.

---

## Problem Statement

Hydrological drought is characterized by prolonged deficits in water availability, reflected through reduced:

- River Discharge
- Groundwater Levels
- Reservoir Storage

Traditional machine learning models primarily focus on temporal patterns and often fail to capture spatial interactions between hydrologically connected regions.

This project addresses that limitation by utilizing **Spatio-Temporal Graph Neural Networks (STGNNs)**, which can simultaneously learn:

- Spatial relationships between connected subbasins
- Temporal evolution of hydrological conditions

---

## Objectives

### Primary Objective

Develop an STGNN-based framework for hydrological drought prediction using integrated hydro-meteorological datasets.

### Specific Objectives

- Construct a hydrological graph using subbasins and river connectivity.
- Integrate multi-source datasets into a unified spatio-temporal framework.
- Generate drought indicators such as SDI/SSI.
- Develop and train STGNN architectures.
- Compare performance against traditional ML and deep learning models.
- Analyze the contribution of different data sources to drought prediction.

---

## Proposed Framework

```text
Hydrological Data
(Discharge, Groundwater, Reservoirs)
                +
Meteorological Data
(Rainfall, Temperature, Humidity, ET)
                +
Subbasin & River Network Information
                ↓
      Data Integration
                ↓
      Graph Construction
                ↓
    Feature Engineering
                ↓
       Tensor Creation
                ↓
            STGNN
                ↓
 Hydrological Drought Prediction
```

---

## Study Area Representation

### Graph Nodes

Each node represents a:

```text
Subbasin
```

Alternative representations such as monitoring stations may also be explored.

### Graph Edges

Edges represent hydrological connectivity derived from:

```text
River Channel Network
```

Possible alternatives:

- Upstream–Downstream Relationships
- Subbasin Connectivity
- K-Nearest Neighbor Graphs

---

## Input Features

### Hydrological Variables

- River Discharge
- Groundwater Level
- Reservoir Storage

### Atmospheric Variables

- Rainfall
- Temperature
- Relative Humidity
- Evapotranspiration
- Wind Speed (Optional)

---

## Prediction Targets

### Regression Tasks

- Future River Discharge
- Streamflow Drought Index (SDI)
- Standardized Streamflow Index (SSI)

### Classification Tasks

- Normal
- Mild Drought
- Moderate Drought
- Severe Drought

---

## Project Structure

```text
stgnn_hydrological_drought/

│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── graph/
│   └── tensors/
│
├── notebooks/
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── discharge.py
│   │   ├── groundwater.py
│   │   ├── reservoir.py
│   │   └── atmosphere.py
│   │
│   ├── graph/
│   │   ├── build_nodes.py
│   │   ├── build_edges.py
│   │   └── adjacency.py
│   │
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── drought_indices.py
│   │
│   ├── dataset/
│   │   ├── tensor_builder.py
│   │   └── sequence_generator.py
│   │
│   ├── models/
│   │   ├── gcn_gru.py
│   │   ├── gcn_lstm.py
│   │   └── baselines.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── utils/
│
├── configs/
├── results/
├── reports/
└── README.md
```

---

## Data Pipeline

### Step 1: Data Collection

Collect and organize:

- River Discharge Data
- Groundwater Data
- Reservoir Storage Data
- Atmospheric Variables
- Subbasin and River Network Data

---

### Step 2: Data Preprocessing

Tasks include:

- Missing Value Treatment
- Outlier Detection
- Temporal Alignment
- Spatial Mapping
- Data Standardization

---

### Step 3: Feature Integration

For every:

```text
Subbasin × Time Step
```

Create a feature vector:

```python
[
    discharge,
    groundwater,
    reservoir_storage,
    rainfall,
    temperature,
    humidity,
    evapotranspiration
]
```

---

### Step 4: Graph Construction

Construct:

```text
Nodes → Subbasins
Edges → River Connectivity
```

Generate the graph adjacency matrix.

---

### Step 5: Tensor Generation

Convert data into:

```python
[T, N, F]
```

Where:

- `T` = Number of Time Steps
- `N` = Number of Nodes
- `F` = Number of Features

Example:

```python
(300, 130, 7)
```

---

### Step 6: Sequence Generation

Apply a sliding window approach:

```text
Past 12 Months
        ↓
Predict Next Month
```

---

## STGNN Architecture

### Baseline Model

```text
Input Features
        ↓
Graph Convolution Layer
        ↓
Graph Convolution Layer
        ↓
GRU / LSTM
        ↓
Dense Layer
        ↓
Drought Prediction
```

### Candidate Architectures

- GCN + GRU
- GCN + LSTM
- GAT + GRU
- Graph WaveNet
- Graph Transformer

---

## Baseline Models

### Machine Learning

- Random Forest
- XGBoost

### Deep Learning

- LSTM
- GRU

---

## Experimental Design

### Experiment 1

**Input:**

```text
Discharge Only
```

### Experiment 2

**Input:**

```text
Discharge + Atmospheric Variables
```

### Experiment 3

**Input:**

```text
Discharge + Atmospheric Variables + Groundwater
```

### Experiment 4

**Input:**

```text
Discharge + Atmospheric Variables + Groundwater + Reservoir Storage
```

### Research Question

> How much does each additional hydrological variable improve drought prediction performance?

---

## Evaluation Metrics

### Regression Metrics

- RMSE
- MAE
- R² Score
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE)

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Expected Outcomes

- Hydrologically meaningful graph representation of river basins.
- Integrated multi-source hydro-meteorological dataset.
- STGNN-based hydrological drought prediction framework.
- Performance comparison with conventional forecasting models.
- Quantitative analysis of the impact of groundwater, reservoir storage, and atmospheric variables on drought prediction.

---

## Future Scope

- Dynamic Graph Construction
- Graph Attention Networks (GAT)
- Graph Transformers
- Multi-Basin Generalization
- Real-Time Drought Monitoring Systems
- Early Warning and Decision Support Systems

---

## Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Deep Learning

- PyTorch
- PyTorch Geometric

### Geospatial Processing

- GeoPandas
- Shapely
- QGIS

### Visualization

- Matplotlib
- Plotly

---

## Research Theme

**Spatio-Temporal Graph Neural Network for Hydrological Drought Prediction using Integrated Hydro-Meteorological Data and River Basin Connectivity**
