# nanoformulation-ML
ML analysis of pH-responsive drug release (PRIVATE - pre-publication)
 
## 🔒 Status: Work in Progress - Private Repository

**System:** Citral-loaded TiO₂@Ag Nanoparticles  
**Application:** pH-responsive drug delivery for necrotic tissue targeting  
**Analysis:** Machine learning for drug release prediction


##  Project Overview

Machine learning analysis of pH-responsive drug release from citral-loaded nanoparticles. This work compares classical kinetic models with modern ML approaches.

### Key Results
-  **ML Performance:** R² = 0.988 (Gaussian Process)
-  **Classical Models:** R² = 0.165 (average)
-  **pH Selectivity:** 750-fold higher release at pH 6.5 vs pH 7.4
-  **Safety Profile:** 82% less release at physiological pH

---

## Dataset
- **Conditions:** pH 5.5, 6.5, 7.4
- **Time Points:** 0, 6, 12, 24, 48, 96 hours
- **Samples:** n = 90
- **Formulations:** Nanoparticle vs Free drug

**Data source:** `data/Smart_Nano.xlsx`

---

## Analysis Notebooks

### Main Analysis
- **[ML Analysis Notebook](notebooks/citral.ipynb)** - Complete machine learning pipeline
  - Data preprocessing
  - Classical model fitting
  - ML model training and comparison
  - Feature importance analysis
  - Statistical validation

---

##  Analysis Methods

### Classical Models
- Zero-order, First-order, Higuchi, Korsmeyer-Peppas

### Machine Learning
- Gaussian Process Regression (R²=0.988)
- Gradient Boosting (R²=0.959)
- Random Forest (R²=0.790)

### Statistical Analysis
- Feature importance
- AUC comparison
- Interaction effects (pH × Time × Formulation)

---

##  Main Findings

### Model Comparison
| Model Type | Average R² |
|------------|-----------|
| Machine Learning | 0.988 |
| Classical Models | 0.165 |

**Improvement: +499%**

### pH-Dependent Release (AUC₀₋₉₆)
| pH | Nanoparticle | Free Drug | Difference |
|----|--------------|-----------|------------|
| 5.5 | 49.5 | 52.9 | -6.4% |
| 6.5 | 4859.9 | 4715.9 | +3.1% |
| 7.4 | 12.0 | 31.2 | -61.5%* |

*Statistically significant (p=0.02)

---

##  Repository Structure

```
ph-responsive-nanoformulation-ml/
├── README.md                  # This file
├── data/
│   └── Smart_Nano.xlsx       # Raw experimental data
├── notebooks/
│   └── ml_analysis.ipynb     # Main analysis notebook
└── results/
    └── figures/              # Generated plots and visualizations
```

---

##  Tools & Technologies

- **Platform:** Google Colab
- **Language:** Python 3.10+
- **Key Libraries:** scikit-learn, scipy, pandas, matplotlib, seaborn

---

##  How to Run

1. Open the notebook in Google Colab: [ml_analysis.ipynb](notebooks/citral.ipynb)
2. Upload `Smart_Nano.xlsx` when prompted
3. Run all cells sequentially

---

##  Notes

This is ongoing research. Methods and results are subject to change as analysis progresses.

---

##  Confidentiality

Unpublished research - do not share without permission.

---

**Last Updated:** January 2026
