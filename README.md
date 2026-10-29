# Agentic-AI-for-Automated-Cybersecurity-Threat-Investigation-and-Response
Major Project - ID: PT424, Group ID: 26P_3CSE424, Group Number: 4 

# Agentic AI for Automated Cybersecurity Threat Investigation and Response

B.Tech Major Project, School of Computer Science & Engineering, IILM University, Greater Noida
Guide: Mr. Yash Garg
Team: Palak Ramtri, Priyam Bhowmik, Anushka Rajput, Harsh Aggarwal

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
Stratified 80/20 train-test split of CICIDS2017 (benign vs attack), seed 42.
Random splits are known to give optimistic scores, so a time- or attack-type-based
split is planned.

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |

`results/metrics.md`

## Repository Structure
```
src/preprocess.py       cleaning, binary labels, split, scaling
src/train_baseline.py   baseline models, metrics, confusion matrices
src/make_form_text.py   results summary from real metrics
src/make_synthetic.py   FAKE data generator for smoke-testing only
docs/                   Phase-I report, weekly progress reports
results/                generated metrics and plots
data/                   dataset instructions (raw data not committed)
```

## Roadmap
1. Baseline classifiers on CICIDS2017 (done)
2. CNN and RNN models, compared against the baseline
3. DRL agent in a simulated environment
4. Evaluation against a rule-based baseline, then XAI and adversarial robustness

## Tech Stack
Python, pandas, scikit-learn, matplotlib (PyTorch/TensorFlow planned for CNN/RNN/DRL)
