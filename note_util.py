import torch
from collections import OrderedDict

def merge_channels(seq, total_size=None):
    if total_size is None:
        total_size = sum([value.shape[-1] for key, value in seq.items()])
    batch_size, n_timesteps,_ = next(iter(seq.items()))[1].shape
    seq_device = next(iter(seq.items()))[1].device
    merged_seq = torch.zeros((batch_size,n_timesteps,total_size),device=seq_device)
    offset=0
 
    for key, value in seq.items():
        merged_seq[...,offset:offset+value.shape[-1]] = seq[key]
        offset+=value.shape[-1]
    return merged_seq

def split_channels(seq, token_sections):
    split_seq = OrderedDict()
    offset=0
    for token_section in token_sections:
        split_seq[token_section["label"]] = seq[...,offset:offset+token_section["n_channels"]]
        offset+=token_section["n_channels"]
    return split_seq