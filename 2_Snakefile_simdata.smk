import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']


# RHO = [0.2,0.4,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0] 
RHO = [0.01] 
DEPTH = [1000,10000]
SIZE = [250,1000]
SEED = [1,2]
TOPIC = [13]
RES =[0.1,0.25,0.5,0.75,1.0]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern


rule all:
    input:
        expand(input_dir + sim_data_pattern+'.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES)

rule sc_simulated_data:
    input:
        script = scripts_dir + '2_sim_step1_data.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        sim_data = input_dir + sim_data_pattern+'.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

