import sys
sys.path.append("/home/BCCRC.CA/ssubedi/projects/experiments/asapp/")
import pandas as pd
import asappy.util.sim as sim


# bulk_data = sys.argv[1]
# sim_data_path = sys.argv[2]
# rho = sys.argv[3]
# size = sys.argv[4]
# seed = sys.argv[5]

sim_data_path = 'sim_r_0.9_s_100_sd_1'
rho = 0.9
size = 100
seed = 1


bulk_data = '/data/sishir/database/dice_immune_bulkrna/*.csv'

# sim.simdata_from_bulk_copula(bulk_data,sim_data_path,int(size),float(phi),float(delta),float(rho),int(seed))
depth = 10000
alpha = 1000
sim.sim_from_bulk_gamma(bulk_data,'data/'+sim_data_path,int(size),alpha,float(rho),depth,int(seed))
