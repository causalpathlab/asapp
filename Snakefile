
import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'

sample_in = config['home'] + config['experiment'] + config['input']+ config['sample_id'] +'/'+config['sample_id']
sample_out = config['home'] + config['experiment'] + config['output']+ config['sample_id'] +'/'
scripts_dir = os.path.join(config['home'] + config['experiment'], "scripts/")


print(sample_in)
print(sample_out)
print(scripts_dir)

ALPHA = [1.,2.]
RHO = [0.9]
DEPTH = [100]
SIZE = [100]

NMF = ['_dcnmf','_altnmf','_fnmf']

sim_data_pattern = sample_in+'_a_{alpha}_r_{rho}_d_{depth}_s_{size}'
sample_out = sample_out+'/a_{alpha}_r_{rho}_d_{depth}_s_{size}/'

print(config['home'] + config['experiment'] + config['resources_dice'])
rule all:
    input:
        expand(sim_data_pattern+'.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE),
        expand(sample_out+'_pbulk.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE),
        expand(sample_out+'_altnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE),
        expand(sample_out+'_dcnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE),
        expand(sample_out+'_fnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE)

rule sc_simulated_data:
    input:
        script = scripts_dir + '1_sim_data.py', 
        bulk_data = config['home'] + config['experiment'] + config['resources_dice']
    output:
        sim_data = sim_data_pattern+'.npz'
    params:
        sim_data_path = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_path} {wildcards.alpha} {wildcards.rho} {wildcards.depth} {wildcards.size}'

rule pseudobulk:
    params:
        sim_data_path = sim_data_pattern,
        pbulk_data_path = sample_out+'_pbulk'
    input:
        script = scripts_dir + '2_pseudo_bulk.py', 
        sim_data = rules.sc_simulated_data.output.sim_data
    output:
        pbulk_data = sample_out+'_pbulk.npz'
    shell:
        'python {input.script} {params.sim_data_path} {params.pbulk_data_path}'

rule nmf:
    params:
        sim_data_path = sim_data_pattern,
        nmf_data_path = sample_out
    input:
        script = scripts_dir + '3_nmf.py', 
        pbulk_data = rules.pseudobulk.output.pbulk_data
    output:
        altnmf_data = sample_out+'_altnmf.npz',
        dcnmf_data = sample_out+'_dcnmf.npz',
        fnmf_data = sample_out+'_fnmf.npz'
    shell:
        'python {input.script} {params.sim_data_path} {params.nmf_data_path} {input.pbulk_data}'

