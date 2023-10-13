import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'


configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']

RHO = [0.2,0.4,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95] 
DEPTH = [100,1000,10000]
SIZE = [250]
SEED = [1,2]
TOPIC = [7]
RES =[0.1,0.3,0.6,0.9]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'.h5asap',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'.h5asap_full',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'_eval.csv',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES)

rule run_asap:
    input:
        script = scripts_dir + '2_sim_step2_asap.py',
        data = input_dir + sim_data_pattern+'.h5'
    output:
        asap_data = output_dir + sim_data_pattern+'.h5',
        asap_outdata = output_dir + sim_data_pattern+'.h5asap',
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script}  {params.sim_data_pattern} {wildcards.topic}'

rule run_asap_full:
    input:
        script = scripts_dir + '2_sim_step3_asapfull.py',
        data = rules.run_asap.output.asap_data
    output:
        asap_outdata = output_dir + sim_data_pattern+'.h5asap_full'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern} {wildcards.topic}'

rule run_nmf_external:
    input:
        script = scripts_dir + '2_sim_step4_external.py',
        data1 = rules.run_asap.output.asap_outdata,
        data2 = rules.run_asap_full.output.asap_outdata
    output:
        eval = output_dir + sim_data_pattern+'_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed} {wildcards.topic} {wildcards.res} '
