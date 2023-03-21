import sys
import pandas as pd
import asap.data.sim as sim


bulk_data = sys.argv[1]
result_path = sys.argv[2]
alpha = sys.argv[3]
rho = sys.argv[4]
depth = sys.argv[5]
size = sys.argv[6]
seed = sys.argv[7]

df = pd.read_csv(bulk_data,compression='zip')


sim.sim_from_bulk(df,result_path,int(size),float(alpha),float(rho),int(depth),int(seed))
# print(alpha,rho,depth,size)