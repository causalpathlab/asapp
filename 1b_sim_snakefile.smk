import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'


configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']

# RHO = [1.0] 
# DEPTH = [10000]
# SIZE = [250]
# SEED = [1]
# TOPIC = [13]
# RES =[0.1]

RHO = [0.01,0.2,0.4,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0] 
DEPTH = [10000]
SIZE = [1000]
SEED = [1,2,3,4,5]
TOPIC = [13]
RES =[0.1,0.25,0.5,0.75,1.0]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'_nmf_eval.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES)

rule run_nmf:
    input:
        script = scripts_dir + '1b_sim_step2_nmf.py'
    output:
        eval = output_dir + sim_data_pattern+'_nmf_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {wildcards.topic} {wildcards.res} '
