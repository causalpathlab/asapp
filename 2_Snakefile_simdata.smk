import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']

# RHO = [0.2,0.4,0.6,0.8,1.0] 
RHO = [1.0] 
PHI = [0.0]  
DELTA = [1.0] 
SIZE = [500]
SEED = [1]


sim_data_pattern = '_r_{rho}_p_{phi}_d_{delta}_s_{size}_sd_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(input_dir + sim_data_pattern+'.h5',rho=RHO,phi=PHI,delta=DELTA,size=SIZE,seed=SEED)

rule sc_simulated_data:
    input:
        script = scripts_dir + '2_sim_step1_data.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        sim_data = input_dir + sim_data_pattern+'.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.phi} {wildcards.delta} {wildcards.size} {wildcards.seed}'

