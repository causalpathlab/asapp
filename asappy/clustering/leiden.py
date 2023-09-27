"""
modifiled from welch-lab/pyliger
"""

import numpy as np
import pandas as pd
from annoy import AnnoyIndex


import leidenalg
import igraph as ig
from scipy.sparse import csr_matrix


def run_ann(theta,k, num_trees=None):

    num_observations = theta.shape[0]
    # decide number of trees
    if num_trees is None:
        if num_observations < 100000:
            num_trees = 10
        elif num_observations < 1000000:
            num_trees = 20
        elif num_observations < 5000000:
            num_trees = 50
        else:
            num_trees = 100

    # build knn graph
    t = AnnoyIndex(theta.shape[1], 'angular')
    for i in range(num_observations):
        t.add_item(i, theta[i])
    t.build(num_trees)

    # create knn indices matrices
    theta_knn = np.vstack([t.get_nns_by_vector(theta[i], k) for i in range(num_observations)])
    return theta_knn


def cluster_vote(clusts, theta_knn, k):
    for i in range(theta_knn.shape[0]):
        clust_counts = {}
        for j in range(k):
            if clusts[theta_knn[i, j]] not in clust_counts:
                clust_counts[clusts[theta_knn[i, j]]] = 1
            else:
                clust_counts[clusts[theta_knn[i, j]]] += 1

        max_clust = -1
        max_count = 0
        for key, value in clust_counts.items():
            if value > max_count:
                max_clust = key
                max_count = value
            elif value == max_count:
                if key > max_clust:
                    max_clust = key
                    max_count = value
        clusts[i] = max_clust
    return clusts


def refine_clusts(theta, clusts, k, num_trees=None):
    theta_knn = run_ann(theta, k, num_trees)
    clusts = cluster_vote(clusts, theta_knn, k)
    return clusts

def build_igraph(snn):
    sources, targets = snn.nonzero()
    weights = snn[sources, targets]

    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph()
    g.add_vertices(snn.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights

    return g

def compute_snn(knn, prune):
    """helper function to compute the SNN graph
    
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6280782/
    
    """
    # int for indexing
    knn = knn.astype(np.int32)

    k = knn.shape[1]
    num_cells = knn.shape[0]

    rows = np.repeat(list(range(num_cells)), k)
    columns = knn.flatten()
    data = np.repeat(1, num_cells * k)
    snn = csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

    snn = snn @ snn.transpose()

    rows, columns = snn.nonzero()
    data = snn.data / (k + (k - snn.data))
    data[data < prune] = 0

    return csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

def leiden_cluster(asap_adata,
                   mode = 'corr',
                   resolution=1.0,
                   k=10,
                   prune=1 / 15,
                   random_seed=1,
                   n_iterations=-1,
                   n_starts=10):

    if isinstance(asap_adata,pd.DataFrame):
        knn = run_ann(asap_adata.to_numpy(),k)
    else:
        knn = run_ann(asap_adata.obsm[mode],k)

    snn = compute_snn(knn, prune=prune)

    g = build_igraph(snn)

    np.random.seed(random_seed)
    max_quality = -1
    for i in range(n_starts):  
        seed = np.random.randint(0, 1000)
        kwargs = {'weights': g.es['weight'], 'resolution_parameter': resolution, 'seed': seed}  
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, n_iterations=n_iterations, **kwargs)

        if part.quality() > max_quality:
            cluster = part.membership
            max_quality = part.quality()

    if isinstance(asap_adata,pd.DataFrame):
        return snn, cluster
    else:
        asap_adata.obs['cluster'] = cluster
        asap_adata.obsp['snn'] = snn
