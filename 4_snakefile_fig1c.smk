import shutil
onsuccess:
    shutil.rmtree(".snakemake")

configfile: 'config.yaml'



sample = 'lung'
sample_dir = config['home'] + config['example'] + sample + '/' 
input_dir = sample_dir +config['input'] 
output_dir = sample_dir +config['output']
scripts_dir = config['home'] + config['experiment']

TOPIC = [10]  ### 6-aml/pbmc, 13-sim, 10-lung/brca/pancreas/kidney
RES = [0.3]

sim_data_pattern = '_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern

rule all:
    input:
        expand(output_dir + sim_data_pattern+'.h5',topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'.h5asap',topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'_liger.csv.gz',topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'_scanpy.csv.gz',topic=TOPIC,res=RES),
        expand(output_dir + sim_data_pattern+'_eval.csv',topic=TOPIC,res=RES)

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

        python {input.script}  {params.sim_data_pattern} {wildcards.topic} {wildcards.res} {params.sample_dir}
        """

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
        'python {input.script} {params.sim_data_pattern} {wildcards.topic} {wildcards.res} {params.sample_dir}'

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
        'python {input.script} {params.sim_data_pattern} {wildcards.topic} {wildcards.res} {params.sample_dir}'