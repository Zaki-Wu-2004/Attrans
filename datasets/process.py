import os
import pandas as pd
import torch

def csv2tensor(file_path):
    #input: a file that contains the EEG data in csv format(multiple columns, each column represents a site)
    #output: a dictionary of tensor converted from csv file, representing different sites for EEG
    df = pd.read_csv(file_path)
    tensor_columns = {col: torch.tensor(df[col].values, dtype=torch.float32) for col in df.columns}
    return tensor_columns

folder_path = os.getcwd()
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    tensor_columns = csv2tensor(file_path)
    #test
    print(tensor_columns)