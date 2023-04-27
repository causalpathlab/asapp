import sys
import pandas as pd
import asap.data.sim as sim


bulk_data = sys.argv[1]
result_path = sys.argv[2]
phi_delta = str(sys.argv[3])
rho = sys.argv[4]
size = sys.argv[5]
seed = sys.argv[6]

phi = phi_delta.split('_')[0]
delta = phi_delta.split('_')[1]

bulk_data = '/home/sishirsubedi/asapp/resources/dice/*.csv'

ct_prop ={'NK':12,
	'THSTAR':2,
	'TH1':2,
	'B_CELL_NAIVE':10,
	'TREG_MEM':2,
	'TH2':4,
	'TREG_NAIVE':2,
	'MONOCYTES':26,
	'M2':2,
	'TFH':4,
	'CD8_NAIVE':12,
	'CD4_NAIVE':16,
	'TH17':6
	}

sim.sim_from_bulk(bulk_data,result_path,int(size),float(phi),float(delta),float(rho),int(seed),use_prop=False,ct_prop=False)
