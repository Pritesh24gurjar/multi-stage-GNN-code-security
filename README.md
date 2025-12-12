

# Multi-Stage GNN for Code Vulnerability Detection and CWE Classification

This repository contains the code, experiments, and analysis artifacts for a research
project on **multiclass software vulnerability classification** using **graph neural
networks (GNNs)**.  
The project focuses on **Common Weakness Enumeration (CWE)** classification from C/C++
source code using **graph-based program representations** and **deep message-passing
architectures**.

The implementation and experiments correspond to the paper:

> *Multiclass CWE Classification using Inter Procedural Abstract Graphs and
> Gated Graph Neural Networks*

---

## Overview

Modern vulnerability detection systems increasingly rely on machine learning, yet most
prior work focuses on **binary vulnerability detection**.  
This project addresses the more challenging problem of **multiclass CWE classification**
by:
- Representing source code as **Inter Procedural Abstract Graphs (IPAGs)**
- Training **Gated Graph Neural Networks (GGNNs)** with varying message-passing depth
- Analyzing **class-level performance and disparity** across CWE categories

Experiments are conducted on the **BigVul dataset** and evaluated over the **top-30 most
frequent CWE classes**.

---

## Repository Structure

- **multi-stage-GNN-code-security/**
  - `README.md` — Project documentation
  - `requirements.txt` — Python dependencies
  - `data_gathering.py` — Dataset collection and filtering
  - `exploration.py` — Exploratory data analysis
  - `model.py` — High-level model definitions
  - `GNN_pipeline.ipynb` — End-to-end experiment pipeline
  - **CAG/** — Compact / abstract graph utilities
    - `datautils/` — Dataset and class utilities
    - `utils/` — Graph construction helpers
  - **IPAG-GIN-CFEExplainer/** — Main experimental workspace
    - `multiclass_experiment/` — GGNN multiclass experiments
    - `notebooks/` — Preprocessing and analysis notebooks
    - `outputs/` — Logs and experiment outputs
    - `scripts/` — SLURM jobs and training pipelines
    - `src/ipag_gin/` — Core graph and model implementations
    - `tests/` — Unit tests




---

## Graph Representation

Each C/C++ function is transformed into an **Inter Procedural Abstract Graph
(IPAG)** that captures:
- Abstract Syntax Tree (AST) structure
- Sequential statement ordering
- Control-related dependencies

Nodes are categorized into semantic types:
- `TOKEN`
- `DECLARATION`
- `PROPERTY`

Graphs are constructed using a multi-stage pipeline that includes vocabulary building,
feature extraction, and graph serialization.

---

## Models

The repository implements multiple graph neural architectures, including:
- **GGNN** (primary model used in the paper)
- GCN
- GAT
- GraphSAGE
- GIN

The main focus is on **GGNNs with varying propagation depth**, allowing systematic analysis
of how long-range message passing affects CWE classification performance and robustness.

---

## Experiments

Experiments are organized as **isolated configurations** to ensure fair comparison:
- Baseline GGNN
- Deeper GGNN (more propagation steps)
- Narrow / wide variants
- Strong regularization settings

All large-scale experiments are executed using **SLURM** on GPU-enabled HPC clusters.
Each run logs:
- Configuration files
- Training history
- Validation metrics
- Test-set results
- Class-level disparity analysis

---

## Data

This project uses the **BigVul dataset** (not included in the repository).

You must obtain the dataset separately:
- https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset

Preprocessing scripts assume function-level inputs with associated CWE labels.

---

## Setup

### Requirements
- Python ≥ 3.9
- PyTorch
- PyTorch Geometric
- CUDA (for GPU training)
- SLURM (optional, for cluster execution)

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## Running Experiments

### Data Preprocessing

```bash
python scripts/batch_data_preprocessing_multiclass.py
```

### Training a GGNN Model

```bash
python scripts/trainer_multiclass_cache_clearing.py \
  --config configs/ggnn_top30.json
```

### SLURM Execution

Example:

```bash
sbatch scripts/run_ggnn_cwe_exp.slurm
```

---

## Reproducibility

* All experiments are configuration-driven
* SLURM job files specify exact resource requirements
* Logs, metrics, and outputs are preserved in structured formats
* Unit tests validate key graph construction components

The branch used for the paper:

```
feature-v1
```

---

## Artifact and Research Use

This repository is intended to support:

* Multiclass vulnerability detection research
* Graph-based program analysis
* CWE-specific performance analysis
* Robustness and disparity studies in ML-for-code

Researchers are encouraged to reuse individual components (graph construction, model
definitions, analysis scripts) independently.

---

## Citation

If you use this code or build upon this work, please cite the associated paper:

```bibtex
@article{yourpaper2025,
  title   = {Multiclass CWE Classification using Inter Procedural Abstract Graphs},
  author  = {Sri Immani},
  year    = {2025}
}
```

---

## License

This project is released under the MIT License unless otherwise specified in submodules.

---

## Contact

For questions or issues related to the research code, please open a GitHub issue or
contact the repository maintainer.


