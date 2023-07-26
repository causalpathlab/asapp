import pandas as pd
import  numpy as np

def batch_correction_scanorama(mtx,batch_label):

    from asap.util._scanorama import assemble

    batches = list(set(batch_label))
    datasets = []
    datasets_indices = []

    for b in batches:
        indices = [i for i, item in enumerate(batch_label) if item == b]
        datasets_indices = datasets_indices + indices
        datasets.append(mtx[indices,:])

    datasets_bc = assemble(datasets)
    df = pd.DataFrame(np.vstack(datasets_bc))
    df.index = datasets_indices
    return df.sort_index().to_numpy()