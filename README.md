# Physics-Informed Graph Neural Networks (PIGNN) for Bio-Nanotech

**A "Digital Twin" framework for predicting phytochemical release from Ag-doped TiO₂ nanoparticles using Chemical Topology and Physical Laws.**

---

##  Project Overview
This repository contains the complete computational pipeline for modeling the pH-responsive release of bioactive extracts from *Costus lucanusianus*-mediated nanoparticles. 

Unlike standard "black-box" machine learning, this project utilizes a **Physics-Informed Graph Neural Network (PIGNN)**. The model "reads" the 2D molecular structure of 13 different phytochemicals and ensures all predictions stay within the physical boundaries of the **Korsmeyer-Peppas** diffusion law.

### Key Scientific Innovations
* **Graph-Molecular Encoding:** We represent the plant extract's chemical complexity by converting SMILES strings into molecular graphs (Atoms as nodes, Bonds as edges).
* **Physics-Constraint:** A custom loss function regularizes the model using mass-transport equations, preventing overfitting and ensuring scientific validity.
* **Predictive Digital Twin:** The system can simulate drug release in "unseen" environments, such as a precise pH of 6.3, acting as a virtual laboratory.
---

##  Repository Structure
```text
nanoformulation-ML/
├── data/
│   ├── phytochemical_inventory.csv     # 13 bioactives with SMILES & GC-MS data
│   └── nanoparticle_release_data.csv   # Raw experimental release kinetics
├── notebooks/
│   ├── 01_classical_modeling.ipynb      # Baseline kinetic fits (Higuchi, etc.)
│   ├── 02_smiles_graph_encoding.ipynb   # Molecular visualization & GNN preprocessing
│   └── 03_pignn_predictive_twin.ipynb   # Final PIGNN model & simulation
├── exports/
│   ├── models/                         # Trained PIGNN weights (.pth)
│   └── figures/                        # R² plots, Molecular grids, and Architecture
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Project documentation

