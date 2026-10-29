"""Generate FAKE CICIDS-like CSVs to smoke-test the pipeline without the real
dataset. Results from this data are meaningless - never report them."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
os.makedirs("data/synthetic", exist_ok=True)
cols = [" Flow Duration", " Total Fwd Packets", " Total Backward Packets",
        "Flow Bytes/s", " Flow Packets/s", " Fwd Packet Length Mean",
        " Bwd Packet Length Mean", " Flow IAT Mean", " Packet Length Variance",
        " SYN Flag Count", " ACK Flag Count", " Average Packet Size"]
for part in range(2):
    n = 3000
    attack = rng.random(n) < 0.3
    X = rng.normal(0, 1, (n, len(cols))) + attack[:, None] * rng.normal(1.2, 0.3, len(cols))
    df = pd.DataFrame(X, columns=cols)
    df[" Fwd Header Length"] = 1
    df[" Fwd Header Length "] = 1          # duplicate-name quirk after stripping
    df.loc[rng.choice(n, 20), "Flow Bytes/s"] = np.inf
    df.loc[rng.choice(n, 20), " Flow Packets/s"] = np.nan
    df[" Label"] = np.where(attack, rng.choice(["DoS Hulk", "PortScan"], n), "BENIGN")
    df.to_csv(f"data/synthetic/synthetic_{part}.csv", index=False)
print("wrote data/synthetic/*.csv (FAKE data for smoke test only)")
