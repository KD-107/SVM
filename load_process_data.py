import numpy as np
import pandas as pd
from icecream import ic
from collections import Counter

def Continuous(data,line):
    counter = Counter(data[:, line])
    counter_key = counter.keys()
    tag = 0
    counter_tabel = {}
    for key in counter_key:
        counter_tabel[key] = tag
        tag = tag + 1
    for i in range(len(data[:, line])):
        data[i, line] = counter_tabel[data[i, line]]
    return data

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def load_process_data():

    pica_data = []
    # ic(pica_data.shape)
    with open("./data/pica2015.csv",'r') as f:
        lines = f.readlines()
        for i in range(1,len(lines)):
            # ic(lines[i])
            line = lines[i]
            line = line.split(",")
            pica_data.append(line)

        # Using my methods
        # 离散数据连续化
        np_pica_data = np.array(pica_data)
        np_pica_data = Continuous(np_pica_data,3)
        np_pica_data = Continuous(np_pica_data,14)
        np_pica_data = Continuous(np_pica_data,15)

        # Using pandas

        df = pd.DataFrame(np_pica_data[:,:429])
        label = pd.DataFrame(np_pica_data[:,429])
        df.columns = lines[0].split(",")[:429]
        label.columns = ['REPEAT']

        # 以向前填充来填充缺失值
        df.fillna(method='bfill')
        df.fillna(method='ffill')
        label.fillna(method='bfill')
        label.fillna(method='ffill')
        # Object to float
        for key in df.columns:
            df[key] = pd.to_numeric(df[key], errors='coerce')
        label['REPEAT'] = pd.to_numeric(label['REPEAT'],errors='coerce')
        # 归一化
        df_mean_norm = mean_norm(df)
        ic(df_mean_norm.isnull())
        df_mean_norm.to_csv("./data/processed_data.csv",index=False)
        label.to_csv("./data/label.csv",index=False)
        return df_mean_norm,label

load_process_data()