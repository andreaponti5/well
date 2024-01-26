import json
import os
import numpy as np
import pandas as pd
import pickle as pkl

from kbarycenters import KBarycenters
from utils import normalize_pressure_flow, align_pc

# Set the experiment configuration
pd.options.mode.chained_assignment = None
config = {
    "network_name": "Hanoi",
    "algo": "kbary",
    "max_iter": 5
}

# Load and preprocess simulation data
sensors = json.load(open(f"data/network/{config['network_name']}.json"))
trace = pd.read_csv(f"data/simulation/{config['network_name']}/sim_res.csv", sep=";")
trace, normalizer = normalize_pressure_flow(trace[["pipe", "severity", "time"] + sensors])
trace[sensors] = trace[sensors].round(decimals=6)
dataset, supports = align_pc(trace)

# Initialize directory for results
filepath = f"result/{config['network_name']}/{config['algo']}/"
os.makedirs(filepath, exist_ok=True)

for n_cluster in np.arange(5, 56, 10):
    print(f"n_cluster: {n_cluster}")
    config["n_cluster"] = int(n_cluster)
    for trial in range(1):
        print(f"trial: {trial}")
        filename = f"c{n_cluster}_t{trial}"
        config["trial"] = trial
        model = KBarycenters(n_cluster, seed=trial).fit(dataset[sensors], supports, max_iter=config["max_iter"])
        print(f"Execution time: {model.train_time:.4f} sec\n")
        dataset["cluster"] = model._labels
        dataset[["pipe", "severity", "cluster"]].to_csv(f"{filepath}{filename}_cluster.csv", index=False)
        json.dump(config, open(f"{filepath}{filename}_config.json", "w"))
        pkl.dump(model, open(f"{filepath}{filename}_model.pkl", "wb"))
