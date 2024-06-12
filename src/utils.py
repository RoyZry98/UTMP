import torch
import os
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model

def torch_mul(input_a:torch.tensor,input_b:torch.tensor):
    dim = len(input_a.shape)
    if dim == 1:
        return torch.mul(input_a,input_b)
    if dim == 2:
        return torch.matmul(input_a,input_b)
    elif dim == 3: 
        a,b,c = input_a.size()
        input_a = input_a.view(a, b)
        # a,b = input_b.size()
        # input_b = input_b.view(a, b)
        return torch.matmul(input_a, input_b).reshape((a,b,c))
