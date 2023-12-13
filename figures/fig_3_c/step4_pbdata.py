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



# sample = 'sim_r_0.99_d_10000_s_77000_s_1'
# rho = 0.99
# depth = 10000
# size = 77000
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
wdir = sys.argv[9]

result_file1 = wdir+sample+'_result.csv'
result_file2 = wdir+sample+'_all_result.csv'
print(sample)

model_list = ['asap']

file_path=wdir+sample+'_model.log'


lines_with_numbers = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line and line[0].isdigit():
            if len(line)>5:
                if 'pseudobulk shape...(' in line:
                    lines_with_numbers.append(line.split())

df1 = pd.DataFrame({'Lines': lines_with_numbers})
df = pd.DataFrame(df1['Lines'].tolist())
df = df.iloc[:,4:6]
pbvals = np.array([int(x.replace(')','')) for x in df[5]])


lines_with_numbers = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line and line[0].isdigit():
            if len(line)>5:
                if 'Pseudo-bulk size :(' in line:
                    lines_with_numbers.append(line.split())

df1 = pd.DataFrame({'Lines': lines_with_numbers})
df3 = pd.DataFrame(df1['Lines'].tolist())
df3 = df3.iloc[:,4:7]

pbtotal = int([x.replace(')','') for x in df3[6]][0])
pbsize_mean = pbvals.mean()
pbsize_std = pbvals.std()

model_list = ['asap']
res_list1 = [pbtotal]
res_list2 = [pbsize_mean]
res_list3 = [pbsize_std]

res = []
res = construct_res(model_list,res_list1,'pbtotal',res)
res = construct_res(model_list,res_list2,'pbsize',res)
# res = construct_res(model_list,res_list3,'pbsize_std',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
