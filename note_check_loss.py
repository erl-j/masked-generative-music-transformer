#%%
import torch
from collections import OrderedDict
from note_loss import special_loss_slow, special_loss_parallel, special_loss

target = OrderedDict()
target["type"]=torch.tensor([[[1, 0],[1,  0],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)
target["pitch"]=torch.tensor([[[0,1],[0,  1],[0, 1],[0, 1],[0, 1]]]).to(torch.float32)

logits = OrderedDict()
logits["type"]=(target["type"].clone()+0.5)*10
logits["pitch"]=(target["pitch"].clone()+0.5)*10

mask = OrderedDict()
mask["type"]=torch.tensor([[[0,0],[0,0],[1, 1],[1, 1],[1, 1]]]).to(torch.float32)
mask["pitch"]=torch.tensor([[[1,1],[0,0],[1,1],[1,1],[0, 0]]]).to(torch.float32)

slow_loss = special_loss_slow(logits, target, mask, debug=False)
print(slow_loss)

loss = special_loss_parallel(logits, target, mask, debug=False)
print(loss)

loss = special_loss(logits, target, mask, debug=False)
print(loss)

# shuffle the time order of the target mask and logits
perm=torch.randperm(target["type"].shape[1])
for key in target:
    target[key]  = target[key][:,perm,:]
    logits[key]  = logits[key][:,perm,:]
    mask[key]  = mask[key][:,perm,:]

slow_loss = special_loss_slow(logits, target, mask, debug=False)
print(slow_loss)

loss = special_loss_parallel(logits, target, mask, debug=False)
print(loss)

loss = special_loss(logits, target, mask, debug=False)
print(loss)

# append new random time steps to the target mask and logits that are unmasked
for key in target:
    target[key] = torch.cat([target[key],target[key]],dim=1)
    logits[key] = torch.cat([logits[key],logits[key]],dim=1)
    mask[key] = torch.cat([mask[key],torch.zeros_like(mask[key])],dim=1)


slow_loss = special_loss_slow(logits, target, mask, debug=False)
print(slow_loss)

loss = special_loss_parallel(logits, target, mask, debug=False)
print(loss)

loss = special_loss(logits, target, mask, debug=False)
print(loss)

# %%

# append new random time steps to the target mask and logits that are unmasked
for key in target:
    target[key] = torch.cat([target[key],target[key]],dim=1)
    logits[key] = torch.cat([logits[key],logits[key]],dim=1)
    mask[key] = torch.cat([mask[key],torch.ones_like(mask[key])],dim=1)


slow_loss = special_loss_slow(logits, target, mask, debug=False)
print(slow_loss)

loss = special_loss_parallel(logits, target, mask, debug=False)
print(loss)

loss = special_loss(logits, target, mask, debug=False)
print(loss)



# %%
