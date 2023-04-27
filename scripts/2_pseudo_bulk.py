import sys
import numpy as np
from asap.annotation import ASAPNMF
from asap.data.dataloader import DataSet

inpath = sys.argv[1]
outpath = sys.argv[2]

dl = DataSet(inpath,outpath,data_mode='sparse',data_ondisk=False)
dl.initialize_data()
dl.load_data()

asap = ASAPNMF(adata=dl,tree_max_depth=10)
asap.get_pbulk()
np.savez(outpath, pbulk= asap.pbulk_mat)

