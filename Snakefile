import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'

sample_in = config['home'] + config['experiment'] + config['input']+ config['sample_id'] +'/'+config['sample_id']
result_dir = config['home'] + config['experiment'] + config['output']+ config['sample_id'] +'/'
scripts_dir = os.path.join(config['home'] + config['experiment'], "scripts/")


print(sample_in)
print(result_dir)
print(scripts_dir)

##### noise level and size
# ALPHA = [1000]
# RHO = [0.0, 0.25, 0.5, 0.75, 1.0]
# DEPTH = [10000]
# SIZE = [100,1000,10000]
# SEED = [1,2,3,4,5,6,7,8,9,10]


##### dispersion(alpha), depth
ALPHA = [1,10,100,1000,10000]
RHO = [0.9]
DEPTH = [10,100,1000,10000]
SIZE = [1000]
SEED = [1,2,3,4,5,6,7,8,9,10]


sim_data_pattern = sample_in+'_a_{alpha}_r_{rho}_d_{depth}_s_{size}_sd_{seed}'
sample_out = result_dir+'a_{alpha}_r_{rho}_d_{depth}_s_{size}_sd_{seed}/'

print(config['home'] + config['experiment'] + config['resources_dice'])
rule all:
    input:
        expand(sim_data_pattern+'.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_pbulk.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_altnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_dcnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_eval.csv', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_fnmf.npz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_scanpy.csv.gz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_pc2n10.csv.gz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_pc10n100.csv.gz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_pc50n1000.csv.gz', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED),
        expand(sample_out+'_eval_ext.csv', alpha=ALPHA,rho=RHO,depth=DEPTH,size=SIZE,seed=SEED)

rule sc_simulated_data:
    input:
        script = scripts_dir + '1_sim_data.py', 
        bulk_data = config['home'] + config['experiment'] + config['resources_dice']
    output:
        sim_data = sim_data_pattern+'.npz'
    params:
        sim_data_path = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_path} {wildcards.alpha} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

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

rule eval:
    params:
        sim_data_path = sim_data_pattern,
        nmf_data_path = sample_out
    input:
        script = scripts_dir + '4_eval.py', 
        altnmf_data = rules.nmf.output.altnmf_data,
        dcnmf_data = rules.nmf.output.dcnmf_data,
        fnmf_data = rules.nmf.output.fnmf_data
    output:
        eval_data = sample_out+'_eval.csv'
    shell:
        'python {input.script} {params.sim_data_path} {params.nmf_data_path} {input.altnmf_data} {input.dcnmf_data} {input.fnmf_data} {wildcards.alpha} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

rule nmf_ext:
    params:
        sim_data_path = sim_data_pattern,
        nmf_data_path = sample_out
    input:
        script = scripts_dir + '6_nmf_ext.py', 
    output:
        scanpy_data = sample_out+'_scanpy.csv.gz',
        pcl_data = sample_out+'_pc2n10.csv.gz',
        pcm_data = sample_out+'_pc10n100.csv.gz',
        pch_data = sample_out+'_pc50n1000.csv.gz'
    shell:
        'python {input.script} {params.sim_data_path} {params.nmf_data_path}'

rule eval_ext:
    params:
        sim_data_path = sim_data_pattern,
        nmf_data_path = sample_out
    input:
        script = scripts_dir + '6_eval_ext.py', 
        scanpy_data = rules.nmf_ext.output.scanpy_data,
        pcl_data = rules.nmf_ext.output.pcl_data,
        pcm_data = rules.nmf_ext.output.pcm_data,
        pch_data = rules.nmf_ext.output.pch_data
    output:
        eval_data = sample_out+'_eval_ext.csv'
    shell:
        'python {input.script} {params.sim_data_path} {params.nmf_data_path} {input.scanpy_data} {input.pcl_data} {input.pcm_data} {input.pch_data} {wildcards.alpha} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

