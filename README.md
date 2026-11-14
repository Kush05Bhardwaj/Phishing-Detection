# Intelligent Phishing Website Detection Using Machine Learning and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Capstone-120--Mark-purple.svg)]()

## 📌 Project Overview
This repository contains a complete, production-grade end-to-end Machine Learning pipeline for detecting phishing websites using lexical, architectural, and behavioral features. The solution incorporates:
- **Multi-Model Machine Learning**: Comparative benchmarks across Random Forest, Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP / ANN).
- **Adversarial Cybersecurity Testing**: Systematic feature manipulation to simulate black-box/grey-box evasion attacks, evaluated against defensive adversarial retraining.
- **Explainable AI (XAI)**: Global and local feature attributions using SHAP TreeExplainer, translating complex model attributions into plain-English SOC security alerts.
- **Interactive Demonstration**: A Streamlit web dashboard providing real-time sample scanning, model comparisons, and an adversarial playground.

---

## 🏗️ Repository Layout
```
Phishing-Detection/
│
├── data/
│   ├── raw/               # Raw UCI Phishing Websites dataset
│   └── processed/         # Clean, leak-free train and test splits
│
├── notebooks/             # Exploratory and prototyping notebooks (01 to 05)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_adversarial_testing.ipynb
│   └── 05_explainable_ai.ipynb
│
├── src/                   # Production modular Python source code
│   ├── data/              # Ingestion, validation, and split utilities
│   ├── features/          # Feature engineering and selection transformers
│   ├── models/            # Model training, hyperparameter search, serialization
│   ├── evaluation/        # Metrics computation, ROC/PR curves, confusion matrices
│   ├── adversarial/       # Evasion attack generators and adversarial retraining
│   └── explainability/    # SHAP explainers and SOC security alert generators
│
├── models/                # Serialized model artifacts (.joblib)
├── reports/               # Capstone documentation, figures, and comparison tables
│   ├── figures/
│   └── results/
│
├── app/                   # Interactive Streamlit web application
│   └── streamlit_app.py
│
├── docs/                  # Architecture, methodology, and presentation decks
├── tests/                 # End-to-end automated sanity test suite
├── requirements.txt       # Locked dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Kush05Bhardwaj/Phishing-Detection.git
cd Phishing-Detection
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the Streamlit demonstration application:
```bash
streamlit run app/streamlit_app.py
```
