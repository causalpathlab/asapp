import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim'
input_dir = 'experiments/asapp/figures/fig_3_c/data/'
scripts_dir = 'experiments/asapp/figures/fig_3_c/'

RHO = [0.99] ## FIX NOISE for figure 3
DEPTH = [10000]
SIZE = [15500,31000,46500,62000,77000]
SEED = [1,2,3,4,5]


sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}'
sim_data_pattern = sample + sim_data_pattern


rule all:
    input:
        expand(input_dir + sim_data_pattern+'_B_CELL_NAIVE.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_CD4_NAIVE.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_CD8_NAIVE.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_M2.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_MONOCYTES.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_NK.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TFH.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TH17.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TH1.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TH2.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_THSTAR.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TREG_MEM.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(input_dir + sim_data_pattern+'_TREG_NAIVE.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED)

rule sc_simulated_data:
    input:
        script = scripts_dir + 'step1_data.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        bc = input_dir + sim_data_pattern+'_B_CELL_NAIVE.h5',
        cd4 = input_dir + sim_data_pattern+'_CD4_NAIVE.h5',
        cd8 = input_dir + sim_data_pattern+'_CD8_NAIVE.h5',
        m2 = input_dir + sim_data_pattern+'_M2.h5',
        mono = input_dir + sim_data_pattern+'_MONOCYTES.h5',
        nk = input_dir + sim_data_pattern+'_NK.h5',
        tfh = input_dir + sim_data_pattern+'_TFH.h5',
        th17 = input_dir + sim_data_pattern+'_TH17.h5',
        th1 = input_dir + sim_data_pattern+'_TH1.h5',
        th2 = input_dir + sim_data_pattern+'_TH2.h5',
        ths = input_dir + sim_data_pattern+'_THSTAR.h5',
        treg = input_dir + sim_data_pattern+'_TREG_MEM.h5',
        tregn = input_dir + sim_data_pattern+'_TREG_NAIVE.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

