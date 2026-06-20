stgnn-hydrology/
│
├── README.md                       # Project overview
├── LICENSE
├── requirements.txt
├── environment.yml                 # Optional (Conda)
├── .gitignore
│
├── docs/
│   ├── STGNN_Implementation_Report.md
│   ├── methodology.md
│   ├── architecture.md
│   ├── experiments.md
│   └── references.md
│
├── configs/
│   ├── train.yaml
│   ├── model.yaml
│   └── graph.yaml
│
├── data/
│   ├── raw/                        # Original datasets (ignored in Git)
│   ├── interim/                    # Cleaned intermediate files
│   ├── processed/                  # Final ML-ready datasets
│   ├── tensors/                    # Saved tensors
│   └── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Graph_Construction.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Result_Analysis.ipynb
│
├── src/
│   │
│   ├── preprocessing/
│   │      load_data.py
│   │      preprocess.py
│   │      feature_engineering.py
│   │
│   ├── graph/
│   │      build_knn_graph.py
│   │      build_distance_graph.py
│   │      adjacency.py
│   │
│   ├── datasets/
│   │      tensor_builder.py
│   │      sequence_generator.py
│   │      dataset.py
│   │
│   ├── models/
│   │      gcn_gru.py
│   │      gat_gru.py
│   │      graph_wavenet.py
│   │
│   ├── training/
│   │      trainer.py
│   │      losses.py
│   │      callbacks.py
│   │
│   ├── evaluation/
│   │      evaluate.py
│   │      metrics.py
│   │
│   ├── visualization/
│   │      graph_plot.py
│   │      prediction_plot.py
│   │      attention_plot.py
│   │
│   └── utils/
│          seed.py
│          logger.py
│          helpers.py
│
├── scripts/
│   train.py
│   evaluate.py
│   predict.py
│
├── models/
│   checkpoints/
│   pretrained/
│
├── outputs/
│   predictions/
│   logs/
│   tensorboard/
│
├── results/
│   figures/
│   tables/
│   comparison/
│
└── tests/
    test_graph.py
    test_dataset.py
