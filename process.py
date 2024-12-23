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

def dataLoader(file_path, seg=250, mode='train'):
    #input: a file that contains the EEG data in csv format(multiple columns, each column represents a site)
    #output: a dictionary of tensor converted from csv file, representing different sites for EEG
    df = pd.read_csv(file_path)
    shift = 10
    raw_tensor = csv2tensor(file_path)
    C, T = raw_tensor.shape
    segment_length = seg
    batch_size = T // segment_length

    res = raw_tensor[:, 3*shift:(3*shift+batch_size * segment_length-250)].reshape(-1, C, segment_length)

    if mode == 'train':
        data = []
        for i in [0,1,2,4,5]:
            res = raw_tensor[:, i*shift:(i*shift+batch_size * segment_length-250)].reshape(-1, C, segment_length)
            data.append(res)

        res = torch.cat(data, dim=0)

    length = len(res)

    if "concentrating" in file_path:
        label = torch.tensor([2], dtype=torch.int64)
        label = label.expand(length,)
    elif "relaxed" in file_path:
        label = torch.tensor([0], dtype=torch.int64)
        label = label.expand(length,)
    elif "neutral" in file_path:
        label = torch.tensor([1], dtype=torch.int64)
        label = label.expand(length,)

    # print(res.shape)
    # print(label)
    # print(res)
    return res, label

def assemble_data(folder_path, mode='finetune', mode2='train'):
    if not folder_path:
        folder_path = os.getcwd()
    files = [folder_path + f for f in os.listdir(folder_path) if f.endswith('.csv')]

    data = []
    labels = []
    seg = 250 if mode=='finetune' else 1000
    for file in files:
        tensor, label = dataLoader(file, seg, mode2)
        data.append(tensor)
        labels.append(label)
    res=torch.cat(data, dim=0)
    labels=torch.cat(labels, dim=0)
    return res, labels


if __name__ == "__main__":

    folder_path = './datasets/pretrain_data/' #os.getcwd()
    
    data, label = assemble_data(folder_path, 'pretrain')
    print(data.shape, label.shape)
