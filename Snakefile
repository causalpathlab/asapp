from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path

import shutil
onsuccess:
    shutil.rmtree(".snakemake")

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id
scripts_dir = os.path.join(args.home + args.experiment, "scripts/")
EXT = ['rows.csv.gz','cols.csv.gz','data.npy','indices.npy','indptr.npy']
print(sample_in)
print(sample_out)
print(scripts_dir)


rule all:
    input:
        expand(sample_in+'.{ext}',ext=EXT),
        sample_out +'_pbulk.npz',
        sample_out +'_dcnmf.npz',
        sample_out +'_altnmf.npz'


rule sc_simulated_data:
    params:
        datap = sample_in,
        size = 100
    input:
        script = scripts_dir + '1_sim_data.py', 
        dice = args.home + args.experiment + args.resources_dice
    output:
        expand(sample_in+'.{ext}',ext=EXT)
    shell:
        'python {input.script} {input.dice} {params.datap} {params.size}'


rule pseudobulk:
    params:
        datap = sample_in,
        resultp = sample_out
    input:
        script = scripts_dir + '2_pseudo_bulk.py', 
        dataf = expand(sample_in+'.{ext}',ext=EXT)
    output:
        pbulk = sample_out +'_pbulk.npz'
    shell:
        'python {input.script} {params.datap} {params.resultp}'

rule nmf:
    params:
        datap = sample_in,
        resultp = sample_out
    input:
        script = scripts_dir + '3_nmf.py', 
        pbulk = sample_out +'_pbulk.npz'
    output:
        dcnmf = sample_out +'_dcnmf.npz',
        altnmf = sample_out +'_altnmf.npz'
    shell:
        'python {input.script} {params.datap} {params.resultp} {input.pbulk}'

