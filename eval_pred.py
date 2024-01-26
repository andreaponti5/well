import json
import pickle as pkl

import pandas as pd

from utils import normalize_pressure_flow, align_pc

pd.options.mode.chained_assignment = None

network_name = "Hanoi"
algo = "kbary"
sensors = json.load(open(f"data/network/{network_name}.json"))

trace_test = pd.read_csv(f"data/simulation/{network_name}/sim_res_test.csv", sep=";")
trace_test, _ = normalize_pressure_flow(trace_test[["pipe", "severity", "time"] + sensors])
if algo == "kbary":
    trace_test[sensors] = trace_test[sensors].round(decimals=4)
    test_set, supports = align_pc(trace_test)
else:
    test_set = trace_test.groupby(by=["pipe", "severity"]).mean()[sensors]
    test_set = test_set.reset_index()

for c in range(5, 56, 10):
    train_res = pd.read_csv(f"result/{network_name}/{algo}/c{c}_t0_cluster.csv")
    model = pkl.load(open(f"result/{network_name}/{algo}/c{c}_t0_model.pkl", "rb"))

    if algo == "kbary":
        test_set["cluster"] = model.predict(test_set[sensors], supports)
    else:
        test_set["cluster"] = model.predict(test_set[sensors])

    train_clusters = train_res[["pipe", "severity", "cluster"]].groupby("cluster").agg({"pipe": lambda x: list(x),
                                                                                        "severity": lambda x: list(x)})
    n_correct = 0
    for i, cl in train_clusters.iterrows():
        test_cluster = test_set[test_set["cluster"] == i]
        n_correct += test_cluster[test_cluster["pipe"].isin(cl["pipe"])].shape[0]
    print(f"k={c}; acc={n_correct / test_set.shape[0]}")
