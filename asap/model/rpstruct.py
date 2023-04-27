import numpy as np
import pandas as pd
from scipy.linalg import qr
import logging
logger = logging.getLogger(__name__)


def get_rpqr_psuedobulk(mtx,rp_mat):

    logger.info('Randomized QR factorized pseudo-bulk')    
    Z = np.dot(rp_mat,mtx).T
    Q, _ = qr(Z,mode='economic')
    Q = (np.sign(Q) + 1)/2

    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    pbulkd = df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']

    pbulk = {}
    for key, value in pbulkd.items():
        pbulk[key] = mtx[:,value].sum(1)

    return pd.DataFrame.from_dict(pbulk,orient='index')


