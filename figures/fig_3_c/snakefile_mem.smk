import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim'
input_dir = 'experiments/asapp/figures/fig_3_c/data/'
output_dir = 'experiments/asapp/figures/fig_3_c/200k_results/'
scripts_dir = 'experiments/asapp/figures/fig_3_c/'


RHO = [0.99] ## FIX NOISE for figure 3
DEPTH = [10000]
SIZE = [15500,31000,46500,62000,77000]
SEED = [1,2,3,4,5]

topic = 13
res = 1.0

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'_result.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_all_result.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED)

rule run_nmf_mem:
    input:
        script = scripts_dir + 'step3_memutil.py',
        evalin = output_dir + sim_data_pattern+'_nmf_eval.csv'
    output:
        eval = output_dir + sim_data_pattern+'_result.csv'
    params:
        sim_data_pattern = sim_data_pattern,
        topic = topic,
        res = res,
        output = output_dir
    resources:
        mem_mb = 32000,  # Memory in megabytes (32 GB)
        cpu_cores = 8   # Number of CPU cores
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {params.topic} {params.res} {input.evalin} {params.output}'

rule run_nmf_pbdata:
    input:
        script = scripts_dir + 'step4_pbdata.py',
        evalin = rules.run_nmf_mem.output.eval
    output:
        eval = output_dir + sim_data_pattern+'_all_result.csv'
    params:
        sim_data_pattern = sim_data_pattern,
        topic = topic,
        res = res,
        output = output_dir
    resources:
        mem_mb = 32000,  # Memory in megabytes (32 GB)
        cpu_cores = 8   # Number of CPU cores
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {params.topic} {params.res} {input.evalin} {params.output}'
