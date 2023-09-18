import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'


configfile: 'config.yaml'

sample = 'sim'
input_dir = config['home'] + config['experiment'] + config['input'] 
output_dir = config['home'] + config['experiment'] + config['output']

scripts_dir = config['home'] + config['experiment']

RHO = [0.9] # total cell type effect 
SIZE = [1,10]
SEED = [1,2,3]

sim_data_pattern = '_r_{rho}_s_{size}_sd_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        # expand(input_dir + sim_data_pattern+'.h5',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap_full',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc2n10.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc10n100.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_pc50n1000.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_liger.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_scanpy.csv.gz',rho=RHO,size=SIZE,seed=SEED),
        expand(output_dir + sim_data_pattern+'_eval.csv',rho=RHO,size=SIZE,seed=SEED)

# rule sc_simulated_data:
#     input:
#         script = scripts_dir + 'sim_step1_data.py',
#         bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
#     output:
#         sim_data = input_dir + sim_data_pattern+'.h5'
#     params:
#         sim_data_pattern = sim_data_pattern
#     shell:
#         'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.size} {wildcards.seed}'

rule run_asap:
    input:
        script = scripts_dir + 'sim_step2_asap.py',
        data = input_dir + sim_data_pattern+'.h5'
    output:
        asap_data = output_dir + sim_data_pattern+'.h5',
        asap_outdata = output_dir + sim_data_pattern+'.h5asap'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_asap_full:
    input:
        script = scripts_dir + 'sim_step3_asapfull.py',
        data = rules.run_asap.output.asap_data
    output:
        asap_outdata = output_dir + sim_data_pattern+'.h5asap_full'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_nmf_external:
    input:
        script = scripts_dir + 'sim_step4_external.py',
        data = rules.run_asap.output.asap_data
    output:
        pc1 = output_dir + sim_data_pattern+'_pc2n10.csv.gz',
        pc2 = output_dir + sim_data_pattern+'_pc10n100.csv.gz',
        pc3 = output_dir + sim_data_pattern+'_pc50n1000.csv.gz',
        liger = output_dir + sim_data_pattern+'_liger.csv.gz',
        scanpy = output_dir + sim_data_pattern+'_scanpy.csv.gz'

    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}'

rule run_nmf_eval:
    input:
        script = scripts_dir + 'sim_step5_eval.py',
        asap = rules.run_asap.output.asap_outdata,
        asapf = rules.run_asap_full.output.asap_outdata,
        pc1 = rules.run_nmf_external.output.pc1,
        pc2 = rules.run_nmf_external.output.pc2,
        pc3 = rules.run_nmf_external.output.pc3,
        liger = rules.run_nmf_external.output.liger,
        scanpy = rules.run_nmf_external.output.scanpy
    output:
        eval = output_dir + sim_data_pattern+'_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {params.sim_data_pattern}  {wildcards.rho} {wildcards.size} {wildcards.seed} '



