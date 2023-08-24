
def normalize_total_count(mtx):
    import anndata as an
    import scanpy as sc

    adata = an.AnnData(mtx.T)
    sc.pp.filter_cells(adata,min_counts=1e3)
    sc.pp.normalize_total(adata,exclude_highly_expressed=True,target_sum=1e4)
    return adata.X.T
