import os
import pandas as pd
import torch

def csv2tensor(file_path):
    #input: a file that contains the EEG data in csv format(multiple columns, each column represents a site)
    #output: a dictionary of tensor converted from csv file, representing different sites for EEG
    df = pd.read_csv(file_path)
    tensor_columns = {col: torch.tensor(df[col].values, dtype=torch.float32) for col in df.columns}
    tensor_list=list(tensor_columns.values())
    # for i in tensor_list:
    #     print(i.shape)
    res=torch.stack(tensor_list, dim=0)
    return res

def dataLoader(file_path):
    #input: a file that contains the EEG data in csv format(multiple columns, each column represents a site)
    #output: a dictionary of tensor converted from csv file, representing different sites for EEG
    df = pd.read_csv(file_path)
    raw_tensor = csv2tensor(file_path)
    C, T = raw_tensor.shape
    segment_length = 250
    batch_size = T // segment_length

    if "concentrating" in file_path:
        label = torch.tensor([2], dtype=torch.float32)
        label = label.expand(batch_size,)
    elif "relaxed" in file_path:
        label = torch.tensor([0], dtype=torch.float32)
        label = label.expand(batch_size,)
    elif "neutral" in file_path:
        label = torch.tensor([1], dtype=torch.float32)
        label = label.expand(batch_size,)
    
    res = raw_tensor[:, :batch_size * segment_length].reshape(batch_size, C, segment_length)

    # print(res.shape)
    # print(label)
    # print(res)
    return res, label

def assemble_data():
    folder_path = os.getcwd()
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    data = []
    labels = []
    for file in files:
        tensor, label = dataLoader(file)
        data.append(tensor)
        labels.append(label)
    res=torch.cat(data, dim=0)
    labels=torch.cat(labels, dim=0)
    return res, labels
