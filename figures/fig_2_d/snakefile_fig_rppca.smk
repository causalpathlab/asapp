import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'

sample = 'sim'
input_dir = 'experiments/asapp/figures/fig_2_d/data/'
output_dir = 'experiments/asapp/figures/fig_2_d/results/'
scripts_dir = 'experiments/asapp/figures/fig_2_d/'

RHO = [0.01,0.2,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0] 
DEPTH = [10000]
SIZE = [250] 
SEED = [1,2,3,4,5,6,7,8,9,10]
TOPIC = [13]
RES =[0.1,0.25,0.5,0.75,1.0]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'_rppca_eval.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES)

rule run_rppca:
    input:
        script = scripts_dir + 'step2_rppca.py'
    output:
        eval = output_dir + sim_data_pattern+'_rppca_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {wildcards.topic} {wildcards.res} '
