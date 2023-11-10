import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim'
input_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_3_a/data/'
output_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_3_a/results/'
scripts_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_3_a/'

# RHO = [1.0] 
# DEPTH = [10000]
# SIZE = [250]
# SEED = [1]
# TOPIC = [13]
# RES =[0.1]



RHO = [0.99] ## FIX NOISE for figure 3
DEPTH = [10000]
SIZE = [80,155,235,310,385,465,540,620,695,770]  
SEED = [4,5,6,7,8,9,10]

topic = 13
res = 1.0

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'_result.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED)

rule run_nmf_mem:
    input:
        script = scripts_dir + 'step3_memutil.py',
        evalin = output_dir + sim_data_pattern+'_nmf_eval.csv'
    output:
        eval = output_dir + sim_data_pattern+'_result.csv'
    params:
        sim_data_pattern = sim_data_pattern,
        topic = topic,
        res = res
    resources:
        mem_mb = 32000,  # Memory in megabytes (32 GB)
        cpu_cores = 8   # Number of CPU cores
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {params.topic} {params.res} {input.evalin}'
