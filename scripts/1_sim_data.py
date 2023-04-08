import sys
import pandas as pd
import asap.data.sim as sim


bulk_data = sys.argv[1]
result_path = sys.argv[2]
rho = sys.argv[3]
size = sys.argv[4]
seed = sys.argv[5]

bulk_data = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/resources/dice/*.csv'

sim.sim_from_bulk(bulk_data,result_path,int(size),float(rho),int(seed))
