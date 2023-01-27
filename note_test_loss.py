from note_train import special_loss
import torch

target = torch.zeros((4,3,2))
target[:,:,0]=1

logits = torch.zeros_like(target)
logits[:,:,1]=999999999

token_sections =[ {"label":"type","n_channels":2}]
mask = torch.ones_like(target)

loss = special_loss(logits,target,mask,token_sections)
