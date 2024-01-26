import time

import jax
import jax.numpy as jnp
import numpy as np
import ot
import pandas as pd
from ott.geometry import pointcloud
from ott.problems.linear import barycenter_problem as bp
from ott.solvers.linear import discrete_barycenter as db
from scipy.stats import wasserstein_distance
from tqdm import tqdm


class KBarycenters:
    def __init__(self, n_cluster, seed=None):
        self.n_cluster = n_cluster
        self.seed = seed
        self._labels = None
        self.centroids = []
        self.clusters = []
        self.weights = None
        self.supports = None
        self.train_time = None

    def fit(self, weights, supports, max_iter=10):
        self.weights = weights
        self.supports = supports
        start = time.perf_counter()
        centroids = self._init_medoids()
        dist_mat = self._distance_point_centers(centroids)
        clusters = dist_mat.argmin(axis=1)
        self.centroids.append(centroids)
        self.clusters.append(clusters)
        for i in range(max_iter):
            print(f"\nIter {i}...")
            centroids = []
            for k in tqdm(range(self.n_cluster), desc="Update barycenters"):
                idxs = np.where(clusters == k)[0]
                if idxs.shape[0] > 1:
                    centroids.append(self._barycenter(self.weights.loc[idxs]))
                elif idxs.shape[0] == 1:
                    centroids.append(weights.loc[idxs[0]].to_list())
                else:
                    centroids.extend(self._init_medoids(n=1).to_numpy().tolist())
            dist_mat = self._distance_point_centers(pd.DataFrame(data=centroids, columns=self.weights.columns))
            clusters = dist_mat.argmin(axis=1)
            self.centroids.append(centroids)
            self.clusters.append(clusters)
        self.train_time = time.perf_counter() - start
        self._labels = self.clusters[-1]
        return self

    def predict(self, data, supports):
        centroids = pd.DataFrame(data=self.centroids[-1], columns=self.weights.columns)
        dist_mat = self._distance_point_centers(centroids, data, supports)
        clusters = dist_mat.argmin(axis=1)
        return clusters

    def _init_medoids(self, n=None):
        n = self.n_cluster if n is None else n
        return self.weights.sample(n=n, random_state=self.seed).reset_index(drop=True)

    def _distance_point_centers(self, centroids, data=None, data_support=None):
        if data is None:
            data = self.weights
        if data_support is None:
            data_support = self.supports
        dist_mat = np.zeros((data.shape[0], centroids.shape[0]))
        with tqdm(total=len(data) * len(centroids)) as pbar:
            for i, leak in data.iterrows():
                for j, c in centroids.iterrows():
                    pbar.desc = f"Distances [{i}, {j}]"
                    try:
                        distances = [wasserstein_distance(data_support[sensor], self.supports[sensor], a, b)
                                     for (sensor, a), b in zip(leak.items(), c)]
                        dist_mat[i, j] = np.mean(distances)
                        pass
                    except Exception as e:
                        print(e.__str__())
                    pbar.update(1)
        return dist_mat

    def _barycenter(self, data):
        bary = []
        # with tqdm(total=len(data.columns)) as pbar:
        for col in data.columns:
            # pbar.desc = f"Barycenter [{col}]"
            A = np.vstack(data[col]).T
            support = self.supports[col].reshape((-1, 1))
            M = ot.dist(support, support, metric="sqeuclidean")
            M /= M.max()
            weights = np.full(A.shape[1], 1 / A.shape[1])
            bary.append(ot.bregman.barycenter(A, M, 1e-1, weights, numItermax=10000))
            # pbar.update(1)
        return bary

    def _ott_barycenter(self, data):
        bary = []
        with tqdm(total=len(data.columns)) as pbar:
            for col in data.columns:
                a = jnp.array(data[col].to_list())
                a = a / np.sum(a, axis=1)[:, np.newaxis]
                x = jnp.array([self.supports[col] for _ in range(a.shape[0])])
                geom = pointcloud.PointCloud(x.T)
                solver = jax.jit(db.FixedBarycenter())
                problem = bp.FixedBarycenterProblem(geom, a)
                barycenter = solver(problem)
                bary.append(barycenter.histogram)
                pbar.update(1)
        return bary
