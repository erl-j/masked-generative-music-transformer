#%%
import torch
from collections import OrderedDict
from note_train import special_loss
#%%
target = OrderedDict()
target["type"]=torch.tensor([[[1, 0],[1,  0],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)
target["pitch"]=torch.tensor([[[0,1],[0,  1],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)

logits = OrderedDict()
logits["type"]=torch.tensor([[[1,0],[0,  1],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)
logits["pitch"]=torch.tensor([[[0,1],[0,  1],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)

mask = OrderedDict()
mask["type"]=torch.tensor([[[0,0],[0,0],[1, 1],[1, 1],[1, 1]]]).to(torch.float32)
mask["pitch"]=torch.tensor([[[1,1],[0,0],[1,1],[1,1],[0, 0]]]).to(torch.float32)

loss = special_loss(logits, target, mask, debug=True)

# %%
