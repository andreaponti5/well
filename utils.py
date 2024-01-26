import time

import numpy as np
import pandas as pd


def normalize_pressure_flow(trace_matrix):
    pcol = [col for col in trace_matrix.columns if 'p_' in col]
    fcol = [col for col in trace_matrix.columns if 'f_' in col]
    normalizer = []
    for col in pcol + fcol:
        min_value = trace_matrix[col].min()
        max_value = trace_matrix[col].max()
        if max_value - min_value != 0:
            trace_matrix[col] = (trace_matrix[col] - min_value) / (max_value - min_value)
        normalizer.append((min_value, max_value))
    return trace_matrix, normalizer


def align_pc(trace_matrix):
    pcol = [col for col in trace_matrix.columns if 'p_' in col]
    fcol = [col for col in trace_matrix.columns if 'f_' in col]
    pipe_severity = [[pipe, severity] for pipe in trace_matrix["pipe"].unique() for severity in
                     trace_matrix["severity"].unique()]
    aligned_trace = pd.DataFrame(pipe_severity, columns=["pipe", "severity"])
    supports = {}
    for sensor in pcol + fcol:
        support = np.array(sorted(trace_matrix[sensor].unique()))
        supports[sensor] = support
        aligned_pcs = []
        start = time.perf_counter()
        for pipe in trace_matrix["pipe"].unique():
            for severity in trace_matrix["severity"].unique():
                aligned_pc = np.zeros(len(support))
                values = trace_matrix.loc[(trace_matrix["pipe"] == pipe) &
                                          (trace_matrix["severity"] == severity), sensor].value_counts().sort_index()
                for value, value_count in values.items():
                    idx = np.where(support == value)[0][0]
                    aligned_pc[idx] = value_count
                aligned_pcs.append(aligned_pc)
        print(f"{time.perf_counter() - start} sec")
        aligned_trace[sensor] = aligned_pcs
    return aligned_trace, supports

# Parallel version
# def align_pc(trace_matrix):
#     pcol = [col for col in trace_matrix.columns if 'p_' in col]
#     fcol = [col for col in trace_matrix.columns if 'f_' in col]
#     pipe_severity = [[pipe, severity] for pipe in trace_matrix["pipe"].unique() for severity in
#                      trace_matrix["severity"].unique()]
#     aligned_trace = pd.DataFrame(pipe_severity, columns=["pipe", "severity"])
#     supports = {}
#
#     def _align(pipe, severity):
#         aligned_pc = np.zeros(len(support))
#         values = trace_matrix.loc[(trace_matrix["pipe"] == pipe) &
#                                   (trace_matrix["severity"] == severity), sensor].value_counts()
#         for value, value_count in values.items():
#             idx = np.where(support == value)[0][0]
#             aligned_pc[idx] = value_count
#         return aligned_pc
#
#     for sensor in pcol + fcol:
#         support = np.array(sorted(trace_matrix[sensor].unique()))
#         supports[sensor] = support
#         max_iter = len(trace_matrix["pipe"].unique()) * len(trace_matrix["severity"].unique())
#         aligned_pcs = Parallel(n_jobs=-1)(delayed(_align)(pipe, severity) for pipe, severity in
#                                           tqdm(itertools.product(trace_matrix["pipe"].unique(),
#                                                                  trace_matrix["severity"].unique()), total=max_iter))
#         aligned_trace[sensor] = aligned_pcs
#     return aligned_trace, supports
