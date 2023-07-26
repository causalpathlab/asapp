
'''
batch correction module modified from 
https://github.com/brianhie/scanorama
'''

from annoy import AnnoyIndex
from intervaltree import IntervalTree
from itertools import cycle, islice
import numpy as np
import operator
import random
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sys
import warnings

# Default parameters.
ALPHA = 0.10
APPROX = True
BATCH_SIZE = 5000
DIMRED = 100
HVG = None
KNN = 20
N_ITER = 500
PERPLEXITY = 1200
SIGMA = 15
VERBOSE = 2

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn=KNN, metric='manhattan', n_trees=10):
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# To reduce memory usage, split bias computation into batches.
def batch_bias(curr_ds, match_ds, bias, batch_size=None, sigma=SIGMA):
    if batch_size is None:
        weights = rbf_kernel(curr_ds, match_ds, gamma=0.5*sigma)
        weights = normalize(weights, axis=1, norm='l1')
        avg_bias = np.dot(weights, bias)
        return avg_bias

    base = 0
    avg_bias = np.zeros(curr_ds.shape)
    denom = np.zeros(curr_ds.shape[0])
    while base < match_ds.shape[0]:
        batch_idx = range(
            base, min(base + batch_size, match_ds.shape[0])
        )
        weights = rbf_kernel(curr_ds, match_ds[batch_idx, :],
                             gamma=0.5*sigma)
        avg_bias += np.dot(weights, bias[batch_idx, :])
        denom += np.sum(weights, axis=1)
        base += batch_size

    denom = handle_zeros_in_scale(denom, copy=False)
    avg_bias /= denom[:, np.newaxis]

    return avg_bias

def handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    Adapted from sklearn.preprocessing.data'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
    return scale

# Compute nonlinear translation vectors between dataset
# and a reference.
def transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=SIGMA, cn=False,
              batch_size=None):
    # Compute the matching.
    match_ds = curr_ds[ds_ind, :]
    match_ref = curr_ref[ref_ind, :]
    bias = match_ref - match_ds
    if cn:
        match_ds = match_ds.toarray()
        curr_ds = curr_ds.toarray()
        bias = bias.toarray()

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            avg_bias = batch_bias(curr_ds, match_ds, bias, sigma=sigma,
                                  batch_size=batch_size)
        except RuntimeWarning:
            sys.stderr.write('WARNING: Oversmoothing detected, refusing to batch '
                             'correct, consider lowering sigma value.\n')
            return csr_matrix(curr_ds.shape, dtype=float)
        except MemoryError:
            if batch_size is None:
                sys.stderr.write('WARNING: Out of memory, consider turning on '
                                 'batched computation with batch_size parameter.\n')
            else:
                sys.stderr.write('WARNING: Out of memory, consider lowering '
                                 'the batch_size parameter.\n')
            return csr_matrix(curr_ds.shape, dtype=float)

    if cn:
        avg_bias = csr_matrix(avg_bias)

    return avg_bias


# Populate a table (in place) that stores mutual nearest neighbors
# between datasets.
def fill_table(table, i, curr_ds, datasets, base_ds=0,
               knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    if approx:
        match = nn_approx(curr_ds, curr_ref, knn=knn)
    # else:
    #     match = nn(curr_ds, curr_ref, knn=knn, metric_p=1)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells

    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)


# Fill table of alignment scores.
def find_alignments_table(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                          prenormalized=False):
    # if not prenormalized:
    #     datasets = [ normalize(ds, axis=1) for ds in datasets ]

    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], knn=knn,
                       approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:],
                       knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji

            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches
    
    # Find the matching pairs of cells between datasets.
def find_alignments(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                    alpha=ALPHA, prenormalized=False,):
    table1, _, matches = find_alignments_table(
        datasets, knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized,
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

def assemble(datasets, verbose=VERBOSE, view_match=False, knn=KNN,
             sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
             ds_names=None, batch_size=None,
             alignments=None, matches=None):
    if len(datasets) == 1:
        return datasets

    if alignments is None and matches is None:
        alignments, matches = find_alignments(
            datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
        )

    ds_assembled = {}
    panoramas = []
    for i, j in alignments:
        if verbose:
            if ds_names is None:
                print('Processing datasets {}'.format((i, j)))
            else:
                print('Processing datasets {} <=> {}'.
                      format(ds_names[i], ds_names[j]))

        # Only consider a dataset a fixed amount of times.
        if not i in ds_assembled:
            ds_assembled[i] = 0
        ds_assembled[i] += 1
        if not j in ds_assembled:
            ds_assembled[j] = 0
        ds_assembled[j] += 1
        if ds_assembled[i] > 3 and ds_assembled[j] > 3:
            continue

        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)

        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            if datasets[i].shape[0] < datasets[j].shape[0]:
                i, j = j, i
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]

        # Map dataset i to panorama j.
        if len(panoramas_i) == 0:
            curr_ds = datasets[i]
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])

            match = []
            base = 0
            for p in panoramas_j[0]:
                if i < p and (i, p) in matches:
                    match.extend([ (a, b + base) for a, b in matches[(i, p)] ])
                elif i > p and (p, i) in matches:
                    match.extend([ (b, a + base) for a, b in matches[(p, i)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
                             batch_size=batch_size)
            datasets[i] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[i]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                 sigma=sigma, cn=True, batch_size=batch_size)
                expr_datasets[i] = curr_ds + bias

            panoramas_j[0].append(i)

        # Map dataset j to panorama i.
        elif len(panoramas_j) == 0:
            curr_ds = datasets[j]
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_i[0] ])

            match = []
            base = 0
            for p in panoramas_i[0]:
                if j < p and (j, p) in matches:
                    match.extend([ (a, b + base) for a, b in matches[(j, p)] ])
                elif j > p and (p, j) in matches:
                    match.extend([ (b, a + base) for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
                             batch_size=batch_size)
            datasets[j] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[j]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_i[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
                                 cn=True, batch_size=batch_size)
                expr_datasets[j] = curr_ds + bias

            panoramas_i[0].append(j)

        # Merge two panoramas together.
        else:
            curr_ds = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])

            # Find base indices into each panorama.
            base_i = 0
            for p in panoramas_i[0]:
                if p == i: break
                base_i += datasets[p].shape[0]
            base_j = 0
            for p in panoramas_j[0]:
                if p == j: break
                base_j += datasets[p].shape[0]

            # Find matching indices.
            match = []
            base = 0
            for p in panoramas_i[0]:
                if p == i and j < p and (j, p) in matches:
                    match.extend([ (b + base, a + base_j)
                                   for a, b in matches[(j, p)] ])
                elif p == i and j > p and (p, j) in matches:
                    match.extend([ (a + base, b + base_j)
                                   for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]
            base = 0
            for p in panoramas_j[0]:
                if p == j and i < p and (i, p) in matches:
                    match.extend([ (a + base_i, b + base)
                                   for a, b in matches[(i, p)] ])
                elif p == j and i > p and (p, i) in matches:
                    match.extend([ (b + base_i, a + base)
                                   for a, b in matches[(p, i)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            # Apply transformation to entire panorama.
            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=sigma,
                             batch_size=batch_size)
            curr_ds += bias
            base = 0
            for p in panoramas_i[0]:
                n_cells = datasets[p].shape[0]
                datasets[p] = curr_ds[base:(base + n_cells), :]
                base += n_cells

            if not expr_datasets is None:
                curr_ds = vstack([ expr_datasets[p]
                                   for p in panoramas_i[0] ])
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                 sigma=sigma, cn=True, batch_size=batch_size)
                curr_ds += bias
                base = 0
                for p in panoramas_i[0]:
                    n_cells = expr_datasets[p].shape[0]
                    expr_datasets[p] = curr_ds[base:(base + n_cells), :]
                    base += n_cells

            # Merge panoramas i and j and delete one.
            if panoramas_i[0] != panoramas_j[0]:
                panoramas_i[0] += panoramas_j[0]
                panoramas.remove(panoramas_j[0])

    return datasets


