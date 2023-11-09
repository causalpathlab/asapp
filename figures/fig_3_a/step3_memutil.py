import sys
import numpy as np
import pandas as pd

def save_eval(df_res):
    df = pd.read_csv(result_file1)
    df_res = pd.concat([df,df_res],ignore_index=True)
    df_res.to_csv(result_file2,index=False)

def construct_res(model_list,res_list,method,res):
    for model,r in zip(model_list,res_list):
        res.append([method,model,rho,depth,size,seed,topic,cluster_resolution,r])
    return res



# sample = 'sim_r_0.99_d_10000_s_770_s_1'
# rho = 0.99
# depth = 10000
# size = 770
# seed = 1
# topic = 13
# cluster_resolution = 1.0

sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = sys.argv[4]
seed = int(sys.argv[5])
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])
infile = sys.argv[8]

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_3_a/'
result_file1 = wdir+'results/'+sample+'_nmf_eval.csv'
result_file2 = wdir+'results/'+sample+'_result.csv'
print(sample)

model_list = ['asap','asapf','liger','scanpy','nmf']

sample_file=wdir+'results/meta/'+sample+'_memory_profiler_output.txt'

lines_with_numbers = []

with open(sample_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line and line[0].isdigit():
            if len(line)>5:
                if 'MiB' in line:
                    lines_with_numbers.append(line.split())

df = pd.DataFrame({'Lines': lines_with_numbers})
df2 = pd.DataFrame(df['Lines'].tolist())
df2 = df2.iloc[:,:8]

df2[0] = df2[0].astype(int)
df2[3] = df2[3].astype(float)

def get_mem(si,ei):
    memsum = df2.loc[(df2[0] >= si) & (df2[0] <= ei), 3].cumsum().max() 
    startmem = df2.iloc[0,3]
    memu = memsum - startmem
    return memu

asap_mem = get_mem(49,78)
asapf_mem = get_mem(80,144)
scanpy_mem = get_mem(205,241)
baseline_mem = get_mem(243,264)
liger_mem = get_mem(146,203)

res_list5 = [asap_mem,asapf_mem,liger_mem,scanpy_mem,baseline_mem]

res = []
res = construct_res(model_list,res_list5,'Memory',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
