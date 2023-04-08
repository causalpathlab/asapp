import pandas as pd

import glob, os
os.chdir(".")
files = []
for file in glob.glob("*.csv"):
    files.append(file)

dfall = pd.DataFrame()

for f in files:
    df = pd.read_csv(f)
    df[['gene','type']] = df['Additional_annotations'].str.split(';',1,expand=True)
    df = df[df['type']=='protein_coding']
    df = df.drop(columns=['gene','type'])

    ct = f.split('.')[0]
    df[ct] = (df.iloc[:,3:].to_numpy() * df['Transcript_Length(bp)'].to_numpy().reshape(df.shape[0],1)).mean(1)
    df = df[['Feature_name',ct]].T

    df.columns = df.iloc[0]
    df = df[1:]

    df = df.rename_axis(None,axis=1) 

    dfall = pd.concat([dfall,df],axis=0)

dfall = dfall.T
dfall.to_csv('raw_merged.csv.gz',compression='gzip')

