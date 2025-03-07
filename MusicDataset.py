from datasets import Dataset
import os
import torch
import numpy as np
from scipy.stats import expon


def load_music_dataset():
    dict_ds = {'input_ids': []}
    test_min = []
    test_max = []
    for file in os.listdir('./OurWavTokenizer/Tokens'):
        d = torch.squeeze(torch.load(f'./OurWavTokenizer/Tokens/{file}'))
        dict_ds['input_ids'].append(d)
        test_min.append(torch.min(d).item())
        test_max.append(torch.max(d).item())
    return Dataset.from_dict(dict_ds)


# def load_music_dataset():
#     dict_ds = {'input_ids': []}
#     test_min = []
#     test_max = []
#     directory = "./token-new"
#     error = 0
#     # List all files and folders
#     path_folders = os.listdir(directory)
#     for folder in path_folders:
#         folder = os.path.join(directory, folder)
#         for file in os.listdir(folder):
#             try:
#                 d = torch.squeeze(torch.load(os.path.join(folder, file)))
#
#                 dict_ds['input_ids'].append(d)
#                 test_min.append(torch.min(d).item())
#                 test_max.append(torch.max(d).item())
#             except:
#                 #print(f"Error in file {os.path.join(folder, file)}")
#                 error+=1
#     return Dataset.from_dict(dict_ds)


#ds = load_music_dataset()
