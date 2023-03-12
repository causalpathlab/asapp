import sys
import pandas as pd
from asap.data import sim


bulk_data = sys.argv[1]
result_path = sys.argv[2]
size = int(sys.argv[3])

alpha = 10.
beta = 1.
depth = 10000

df = pd.read_csv(bulk_data,compression='zip')

# remove non naive T cells
df = df[['gene', 
'Monocyte, classical',
'Monocyte, non-classical',
'NK cell, CD56dim CD16+',
'B cell, naive', 
'T cell, CD8, naive',
'T cell, CD4, naive',
'T cell, CD8, naive [activated]',
'T cell, CD4, naive [activated]',
'T cell, CD4, memory TREG', 
'T cell, CD4, TH1/17',
]]

    
sim.sim_from_bulk(df,result_path,size,alpha,beta,depth)