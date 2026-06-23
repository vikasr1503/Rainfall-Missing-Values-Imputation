# Graph Construction and Neural Network Flow

## Overview

The `src/graph/` module is responsible for transforming raw hydrological basin information into a graph structure that can be consumed by a Spatio-Temporal Graph Neural Network (STGNN).

The Graph Neural Network itself does not directly understand:

- Subbasin IDs
- River Networks
- GIS Layers
- CSV Files

Instead, it requires a mathematical graph consisting of:

```text
Nodes
Edges
Edge Attributes
Adjacency Information
```

The purpose of the graph module is to convert real-world hydrological systems into this graph representation.

---

# Complete Flow

```text
Subbasin.csv
        +
River Channel Network
        │
        ▼
build_nodes.py
        │
        ▼
build_edges.py
        │
        ▼
adjacency.py
        │
        ▼
edge_index
edge_attr
node_attributes
dynamic_edge_weights
        │
        ▼
Graph Neural Network
        │
        ▼
Spatial Learning
        │
        ▼
Temporal Learning
        │
        ▼
Drought Prediction
```

---

# 1. build_nodes.py

## Purpose

Defines what constitutes a graph node.

In this project:

```text
Node = Subbasin
```

Each row from:

```text
Subbasin.csv
```

becomes one graph node.

Example:

```csv
PolygonId,Area,Subbasin
1,452.4,1
2,387.1,2
3,498.2,3
```

becomes:

```text
Node 0
Node 1
Node 2
```

---

## Why This Is Required

Graph Neural Networks identify nodes using:

```python
0
1
2
3
...
N-1
```

rather than actual subbasin IDs.

Therefore:

```python
Subbasin 1 → Node 0
Subbasin 2 → Node 1
Subbasin 3 → Node 2
```

---

## Node Attributes

The node table stores static properties.

Example:

```python
Node(
    area_km2,
    centroid_lat,
    centroid_lon,
    elevation,
    slope
)
```

These attributes help describe the physical characteristics of each subbasin.

---

## Output

```text
nodes.csv
```

containing:

```text
node_idx
subbasin_id
area
elevation
coordinates
```

---

# 2. build_edges.py

## Purpose

Defines how nodes are connected.

Without edges:

```text
Node A
Node B
Node C
```

are completely isolated.

The GNN would not know that water can flow between them.

---

## Hydrological Connectivity

River systems naturally form directed networks.

Example:

```text
Subbasin 5
      ↓
Subbasin 12
      ↓
Subbasin 28
```

This becomes:

```python
5 → 12
12 → 28
```

---

## Why Direction Matters

Hydrological influence flows:

```text
Upstream
      ↓
Downstream
```

but not necessarily:

```text
Downstream
      ↑
Upstream
```

Therefore the graph is:

```text
Directed
```

---

## Edge Attributes

Each edge may contain:

```python
[
river_length,
stream_order,
travel_distance,
elevation_drop
]
```

These attributes describe the relationship between connected subbasins.

---

## Dynamic Edge Weights

Unlike social networks, river systems change over time.

During monsoon:

```text
Strong connectivity
```

During drought:

```text
Weak connectivity
```

Edge weights can therefore vary monthly.

Example:

```python
W(A→B,t)
```

where:

```text
t = month
```

This allows the graph structure to reflect real hydrological conditions.

---

## Output

```text
edges.csv
```

containing:

```text
source_node
destination_node
edge_attributes
dynamic_weights
```

---

# 3. build_knn_graph.py

## Purpose

Creates a fallback graph when river network topology is unavailable.

---

## Method

Uses subbasin centroids.

Example:

```text
Subbasin A
Latitude
Longitude
```

For each subbasin:

```python
Find K nearest neighbors
```

typically:

```python
K = 5
```

---

## Example

```text
A connected to:
B
C
D
E
F
```

---

## Result

```text
Distance-Based Graph
```

rather than:

```text
Hydrological Graph
```

---

## Use Case

Useful for:

- Initial testing
- Baseline comparisons
- Missing river topology

---

# 4. build_distance_graph.py

## Purpose

Creates graph connections using a fixed distance threshold.

---

## Method

Example:

```python
Threshold = 100 km
```

Rule:

```python
if distance(A,B) < 100:
    connect(A,B)
```

---

## Weighting

Closer subbasins receive larger weights.

Example:

```python
weight = 1 / distance
```

---

## Example

```text
20 km apart
     ↓
Weight = 0.05

80 km apart
     ↓
Weight = 0.0125
```

---

## Result

A geographically connected graph.

---

# 5. adjacency.py

## Purpose

Converts the graph into tensors that PyTorch Geometric can process.

This is the most important file before model training.

---

# Why Conversion Is Needed

The GNN cannot read:

```csv
source,destination
5,12
12,28
```

It requires:

```python
edge_index
```

---

## edge_index

Represents graph connectivity.

Example:

```python
edge_index =

[[0,1,2],
 [1,2,3]]
```

Meaning:

```text
0 → 1
1 → 2
2 → 3
```

---

## edge_attr

Stores edge information.

Example:

```python
edge_attr =

[
 [45.3,3],
 [22.1,4],
 [18.7,2]
]
```

representing:

```text
river length
stream order
```

for each edge.

---

## Dynamic Edge Weight Matrix

Stores monthly edge strengths.

Example:

```python
shape

[T,E]
```

where:

```text
T = months
E = edges
```

---

## Self-Loops

Adds:

```text
A → A
B → B
C → C
```

This ensures a node retains its own information during message passing.

---

# How The Neural Network Uses This Graph

After adjacency.py finishes, the graph is represented as:

```python
edge_index
edge_attr
node_features
```

These become direct inputs to the Graph Neural Network.

---

# Graph Neural Network Message Passing

Suppose:

```text
Subbasin A
```

has neighbors:

```text
Subbasin B
Subbasin C
Subbasin D
```

The GCN performs:

```text
Information from B
+
Information from C
+
Information from D
+
Information from A
```

to generate:

```text
Updated Representation of A
```

---

# Spatial Learning Stage

The graph layers learn:

```text
How neighboring subbasins influence each other
```

Example:

```text
Low rainfall upstream
        ↓
Reduced discharge
        ↓
Downstream drought propagation
```

---

# Temporal Learning Stage

After spatial aggregation:

```text
Month 1
Month 2
Month 3
...
Month 12
```

are passed into:

```text
GRU
or
LSTM
```

which learns temporal evolution.

---

# Complete STGNN Flow

```text
Subbasin.csv
        ↓
build_nodes.py

River Network
        ↓
build_edges.py

Edges
        ↓
adjacency.py

edge_index
edge_attr
node_features
        ↓
Graph Convolution Layer

Spatial Information Aggregation
        ↓
Graph Convolution Layer

Enhanced Spatial Features
        ↓
GRU / LSTM

Temporal Dependency Learning
        ↓
Dense Layer
        ↓
Hydrological Drought Prediction
```

---

# Final Outcome

The graph module transforms raw hydrological datasets into a structured graph representation where:

```text
Nodes      → Subbasins
Edges      → River Connectivity
Edge Attrs → Hydrological Relationships
Weights    → Dynamic Flow Influence
```

This graph becomes the foundation of the STGNN, enabling the model to learn both spatial drought propagation and temporal drought evolution across the river basin.
