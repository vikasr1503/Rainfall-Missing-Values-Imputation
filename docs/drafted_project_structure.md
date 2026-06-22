# 📂 Project Structure

This repository follows a modular, research-oriented architecture designed to ensure **reproducibility**, **maintainability**, and **scalability**. The project separates data processing, graph construction, model development, training, evaluation, and documentation into independent components, making it easy to extend and experiment with new architectures.

---

## Repository Layout

```text
stgnn-hydrology/
│
├── README.md                       # Project overview and usage instructions
├── LICENSE                         # Open-source license
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment (optional)
├── .gitignore                      # Git ignored files
│
├── docs/                           # Project documentation
│   ├── STGNN_Implementation_Report.md
│   ├── methodology.md
│   ├── architecture.md
│   ├── experiments.md
│   └── references.md
│
├── configs/                        # Experiment configuration files
│   ├── train.yaml
│   ├── model.yaml
│   └── graph.yaml
│
├── data/
│   ├── raw/                        # Original datasets (not tracked by Git)
│   ├── interim/                    # Intermediate cleaned datasets
│   ├── processed/                  # Machine learning-ready datasets
│   ├── tensors/                    # Generated graph tensors
│   └── README.md                   # Dataset documentation
│
├── notebooks/                      # Exploratory notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Graph_Construction.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Result_Analysis.ipynb
│
├── src/                            # Core source code
│   │
│   ├── preprocessing/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   │
│   ├── graph/
│   │   ├── build_knn_graph.py
│   │   ├── build_distance_graph.py
│   │   └── adjacency.py
│   │
│   ├── datasets/
│   │   ├── tensor_builder.py
│   │   ├── sequence_generator.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── gcn_gru.py
│   │   ├── gat_gru.py
│   │   └── graph_wavenet.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   ├── visualization/
│   │   ├── graph_plot.py
│   │   ├── prediction_plot.py
│   │   └── attention_plot.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       └── helpers.py
│
├── scripts/                        # Entry-point scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/                         # Saved model weights
│   ├── checkpoints/
│   └── pretrained/
│
├── outputs/                        # Runtime outputs
│   ├── predictions/
│   ├── logs/
│   └── tensorboard/
│
├── results/                        # Final experimental outputs
│   ├── figures/
│   ├── tables/
│   └── comparison/
│
└── tests/                          # Unit tests
    ├── test_graph.py
    └── test_dataset.py
```

---

# Directory Overview

| Directory      | Purpose                                                                               |
| -------------- | ------------------------------------------------------------------------------------- |
| **docs/**      | Research reports, methodology, architecture details, experiment logs, and references. |
| **configs/**   | YAML configuration files containing model, graph, and training hyperparameters.       |
| **data/**      | Stores datasets throughout the pipeline—from raw files to processed tensors.          |
| **notebooks/** | Interactive notebooks for exploratory analysis, visualization, and experimentation.   |
| **src/**       | Core implementation of the complete STGNN pipeline.                                   |
| **scripts/**   | Command-line entry points for training, evaluation, and inference.                    |
| **models/**    | Saved checkpoints and pretrained model weights.                                       |
| **outputs/**   | Generated predictions, TensorBoard logs, and runtime artifacts.                       |
| **results/**   | Final figures, performance tables, and comparative experimental results.              |
| **tests/**     | Unit tests to verify the correctness of graph construction and dataset generation.    |

---

# Design Philosophy

The repository is organized following a **modular machine learning pipeline**, where each component has a single responsibility.

```text
Raw Data
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Graph Construction
    │
    ▼
Tensor Generation
    │
    ▼
Sequence Generation
    │
    ▼
STGNN Models
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
Visualization & Analysis
```

This design enables:

* Easy experimentation with different graph construction strategies.
* Seamless integration of new STGNN architectures.
* Reproducible experiments through configuration files.
* Clean separation between data engineering and model development.
* Better maintainability and scalability for future research.

---

# Future Extensions

The modular structure allows straightforward integration of additional models and experiments, including:

* Graph Attention Networks (GAT)
* Graph WaveNet
* Temporal Graph Transformers
* Diffusion Convolutional Networks
* Multi-task Learning
* Explainable AI (XAI)
* Hyperparameter Optimization
* Distributed Training

This architecture is intentionally designed to support ongoing research and future publication-quality implementations while maintaining clean software engineering practices.
