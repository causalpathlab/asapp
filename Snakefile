import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'


configfile: 'config.yaml'

sample = 'sim'
sample_in = config['home'] + config['experiment'] + config['input'] + sample
result_dir = config['home'] + config['experiment'] + config['output']+ sample

scripts_dir = config['home'] + config['experiment']

PHI = [0.8] # data
DELTA = [0.1] # batch
RHO = [0.9] # total cell type effect 
SIZE = [10]
SEED = [1]

sim_data_pattern = sample_in+'_p_{phi}_d_{delta}_r_{rho}_s_{size}_sd_{seed}'
sample_out = result_dir+'_p_{phi}_d_{delta}_r_{rho}_s_{size}_sd_{seed}'

rule all:
    input:
        expand(sim_data_pattern+'.h5',phi=PHI,delta=DELTA,rho=RHO,size=SIZE,seed=SEED)

rule sc_simulated_data:
    input:
        script = scripts_dir + 'sim_step1.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        sim_data = sim_data_pattern+'.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.size} {wildcards.phi} {wildcards.delta} {wildcards.rho} {wildcards.seed}'

