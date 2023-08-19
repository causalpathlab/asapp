import pandas as pd
import  numpy as np

def batch_correction_scanorama(mtx,batch_label,alpha,sigma):

    from asappy.util._scanorama import assemble

    batches = list(set(batch_label))
    datasets = []
    datasets_indices = []

    for b in batches:
        indices = [i for i, item in enumerate(batch_label) if item == b]
        datasets_indices = datasets_indices + indices
        datasets.append(mtx[indices,:])

    datasets_bc = assemble(datasets,alpha=alpha,sigma=sigma)
    df = pd.DataFrame(np.vstack(datasets_bc))
    df.index = datasets_indices
    return df.sort_index().to_numpy()

def batch_correction_bbknn(mtx,batch_label,barcodes,genes,preprocess):
    
    from bbknn.matrix import bbknn
    import scanpy as sc

    bbknn_out = bbknn(mtx,batch_label,use_annoy=True)	

    X = np.zeros((mtx.shape[0],len(genes)))
    adata = sc.AnnData(X,
        pd.DataFrame(barcodes),
        pd.DataFrame(genes))

    adata.obs['batch_key'] = batch_label
    key_added = 'neighbors'
    conns_key = 'connectivities'
    dists_key = 'distances'
    adata.uns[key_added] = {}
    adata.uns[key_added]['params'] = bbknn_out[2]
    adata.uns[key_added]['params']['use_rep'] = "X_pca"
    adata.uns[key_added]['params']['bbknn']['batch_key'] = "batch_key"
    adata.obsp[dists_key] = bbknn_out[0]
    adata.obsp[conns_key] = bbknn_out[1]
    adata.uns[key_added]['distances_key'] = dists_key
    adata.uns[key_added]['connectivities_key'] = conns_key
    adata.obsm['X_pca'] = mtx
    if preprocess:
        sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata
