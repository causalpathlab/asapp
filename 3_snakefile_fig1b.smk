import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'



sample = 'pbmc'
sample_dir = config['home'] + config['example'] + sample + '/' 
input_dir = sample_dir +config['input'] 
output_dir = sample_dir +config['output']
scripts_dir = config['home'] + config['experiment']

TOPIC = [10]  
# SIZE = [3000,10000]
SIZE = [10000]
RES = [0.1,0.25,0.5,0.75,1.0]
SEED = [1,2,3]

sim_data_pattern = '_s_{size}_t_{topic}_r_{res}_s_{seed}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'.h5',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'.h5asap_full',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'_liger.csv.gz',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'_scanpy.csv.gz',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'_baseline.csv.gz',size=SIZE,topic=TOPIC,res=RES,seed=SEED),
        expand(output_dir + sim_data_pattern+'_eval.csv',size=SIZE,topic=TOPIC,res=RES,seed=SEED)

rule run_asap:
    input:
        script = scripts_dir + '3_step1_asap.py',
        data = input_dir + sample+'.h5'
    output:
        asap_data = output_dir + sim_data_pattern+'.h5',
        asap_outdata = output_dir + sim_data_pattern+'.h5asap'
    params:
        sample = sample,
        input_dir = input_dir,
        sim_data_pattern = sim_data_pattern,
        sample_dir = sample_dir
    shell:
        """
        
        #rm {params.input_dir}{params.sim_data_pattern}.h5

        ln -s {params.input_dir}{params.sample}.h5 {params.input_dir}{params.sim_data_pattern}.h5

        python {input.script}  {params.sim_data_pattern} {wildcards.size} {wildcards.topic} {wildcards.res} {wildcards.seed} {params.sample_dir}
        """

rule run_asap_full:
    input:
        script = scripts_dir + '3_step2_asapfull.py',
        data = rules.run_asap.output.asap_data
    output:
        asap_outdata = output_dir + sim_data_pattern+'.h5asap_full'
    params:
        sim_data_pattern = sim_data_pattern,
        sample_dir = sample_dir

    shell:
        'python {input.script} {params.sim_data_pattern} {wildcards.size} {wildcards.topic}  {wildcards.res} {wildcards.seed} {params.sample_dir}'

rule run_nmf_external:
    input:
        script = scripts_dir + '3_step3_external.py',
        data1 = rules.run_asap.output.asap_outdata
    output:
        liger = output_dir + sim_data_pattern+'_liger.csv.gz',
        scanpy = output_dir + sim_data_pattern+'_scanpy.csv.gz',
        baseline = output_dir + sim_data_pattern+'_baseline.csv.gz'    
    params:
        sim_data_pattern = sim_data_pattern,
        sample_dir = sample_dir
    shell:
        'python {input.script} {params.sim_data_pattern} {wildcards.size} {wildcards.topic} {wildcards.res} {wildcards.seed} {params.sample_dir}'

rule run_nmf_eval:
    input:
        script = scripts_dir + '3_step4_eval.py',
        asap = rules.run_asap.output.asap_outdata,
        asapf = rules.run_asap_full.output.asap_outdata,
        liger = rules.run_nmf_external.output.liger,
        scanpy = rules.run_nmf_external.output.scanpy,
        baseline = rules.run_nmf_external.output.baseline
    output:
        eval = output_dir + sim_data_pattern+'_eval.csv'
    params:
        sim_data_pattern = sim_data_pattern,
        sample_dir = sample_dir

    shell:
        'python {input.script} {params.sim_data_pattern} {wildcards.size} {wildcards.topic} {wildcards.res} {wildcards.seed} {params.sample_dir}'