import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim'
input_dir = 'experiments/asapp/figures/fig_3_a/data/'
scripts_dir = 'experiments/asapp/figures/fig_3_a/'

RHO = [0.99] ## FIX NOISE for figure 3
DEPTH = [10000]
SIZE = [80,155,235,310,385,465,540,620,695,770,  
           1550,2350,3100,3850,4650,5400,6200,6950,7700]
SEED = [4,5,6,7,8,9,10]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}'
sim_data_pattern = sample + sim_data_pattern


rule all:
    input:
        expand(input_dir + sim_data_pattern+'.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED)

rule sc_simulated_data:
    input:
        script = scripts_dir + 'step1_data.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        sim_data = input_dir + sim_data_pattern+'.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

