# Agentic AI for Automated Cybersecurity Threat Investigation and Response

B.Tech Major Project, School of Computer Science & Engineering, IILM University, Greater Noida
Guide: Mr. Yash Garg
Team: Palak Ramtri, Priyam Bhowmik, Anushka Rajput

## Overview
Proposed autonomous system that detects, investigates and responds to cyber threats
using CNN, RNN and Deep Reinforcement Learning, moving security from reactive to proactive.

## Current Status
- [x] Phase-I: problem definition, literature survey, objectives, scope, feasibility (see `docs/`)
- [x] Data preprocessing pipeline (CICIDS2017)
- [x] Baseline classifiers: Logistic Regression and Random Forest (benign vs attack)
- [ ] CNN / RNN detection modules (planned)
- [ ] DRL agent for autonomous threat hunting (planned)
- [ ] Comparison with rule-based baseline (planned)
- [ ] Explainable AI and adversarial robustness (future work)

## Baseline Results
Run the steps below and paste your own table from `results/metrics.md` here.
Results are on a stratified 80/20 random split of CICIDS2017; random splits are
known to be optimistic, so a time/attack-type based split is planned.

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| (fill from results/metrics.md) | | | | | | |

## How to Run
```bash
pip install -r requirements.txt
# 1. download CICIDS2017 CSVs into data/raw/  (see data/README.md)
python src/preprocess.py --max-rows 300000     # drop --max-rows if you have enough RAM
python src/train_baseline.py
```
Outputs are written to `data/processed/` and `results/` (metrics.md, metrics.json,
confusion-matrix images).

## Structure
```
src/preprocess.py       cleaning, binary labels, split, scaling
src/train_baseline.py   baseline models + metrics + confusion matrices
src/make_synthetic.py   FAKE data generator for smoke-testing only
docs/                   Phase-I report, weekly progress reports
results/                generated metrics and plots
```

## Tech Stack
Python, pandas, scikit-learn, matplotlib (PyTorch/TensorFlow planned for CNN/RNN/DRL)
