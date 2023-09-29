import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'


configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']

RHO = [0.4,0.6,0.8,1.0] # total cell type effect 
SIZE = [10]
SEED = [1,2,3,4,5]

sim_data_pattern = '_r_{rho}_s_{size}_sd_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'.h5',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asapad',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap_full',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc5.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc10.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc50.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_rp5.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_rp10.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_rp50.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_liger.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_baseline.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_eval.csv',rho=RHO,size=SIZE,seed=SEED)

rule run_asap:
    input:
        script = scripts_dir + '2_sim_step2_asap.py',
        data = input_dir + sim_data_pattern+'.h5'
    output:
        asap_data = output_dir + sim_data_pattern+'.h5',
        asap_outdata = output_dir + sim_data_pattern+'.h5asap',
        asap_outdata_p = output_dir + sim_data_pattern+'.h5asapad'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_asap_full:
    input:
        script = scripts_dir + '2_sim_step3_asapfull.py',
        data = rules.run_asap.output.asap_data
    output:
        asap_outdata = output_dir + sim_data_pattern+'.h5asap_full'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_nmf_external:
    input:
        script = scripts_dir + '2_sim_step4_external.py',
        data = rules.run_asap.output.asap_data
    output:
        pc1 = output_dir + sim_data_pattern+'_pc5.csv.gz',
        pc2 = output_dir + sim_data_pattern+'_pc10.csv.gz',
        pc3 = output_dir + sim_data_pattern+'_pc50.csv.gz',
        rp1 = output_dir + sim_data_pattern+'_rp5.csv.gz',
        rp2 = output_dir + sim_data_pattern+'_rp10.csv.gz',
        rp3 = output_dir + sim_data_pattern+'_rp50.csv.gz',
        liger = output_dir + sim_data_pattern+'_liger.csv.gz',
        baseline = output_dir + sim_data_pattern+'_baseline.csv.gz'

    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_nmf_eval:
    input:
        script = scripts_dir + '2_sim_step5_eval.py',
        asap = rules.run_asap.output.asap_outdata,
        asapf = rules.run_asap_full.output.asap_outdata,
        pc1 = rules.run_nmf_external.output.pc1,
        pc2 = rules.run_nmf_external.output.pc2,
        pc3 = rules.run_nmf_external.output.pc3,
        rp1 = rules.run_nmf_external.output.rp1,
        rp2 = rules.run_nmf_external.output.rp2,
        rp3 = rules.run_nmf_external.output.rp3,
        liger = rules.run_nmf_external.output.liger,
        baseline = rules.run_nmf_external.output.baseline
    output:
        eval = output_dir + sim_data_pattern+'_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.size} {wildcards.seed} '



