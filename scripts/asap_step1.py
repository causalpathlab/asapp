import sys
sys.path.append("/home/BCCRC.CA/ssubedi/projects/experiments/asapp/")
import pandas as pd
import asappy.util.sim as sim


bulk_data = sys.argv[1]
sim_pattern = sys.argv[2]
size = sys.argv[3]
phi = sys.argv[4]
delta = sys.argv[5]
rho = sys.argv[6]
seed = sys.argv[7]

bulk_data = '/data/sishir/database/dice_immune_bulkrna/*.csv'

sim.simdata_from_bulk(bulk_data,sim_pattern,int(size),float(phi),float(delta),float(rho),int(seed))