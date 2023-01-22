import einops
import torch

def note_loss(target,logits,mask, channel_vocab_sizes=[]):
    # target: (batch token channel), logits: (batch token channel), mask: (batch token channel)
    for batch in range(target.shape[0]):
        for predicted_token_index in range(target.shape[1]):
            for target_token_index in range(target.shape[1]):
                # check if target is a candidate
                # if mask matches and unmasked channel are equal
                target_freqs = []
                if (mask[batch,target_token_index] == mask[batch,predicted_token_index]).all() and (((1-mask)*target)[batch,target_token_index] == (1-mask)*target[batch,predicted_token_index]).all():
                    # 
                    target_freqs.append(target[batch,target_token_index]*mask[batch,target_token_index])
                
                # get average target freqs
                target_freqs = torch.stack(target_freqs).mean(dim=0)
                # compute cross entropy

             
            
           

  






    # binary cross entropy
    x = einops.rearrange(x,'b p t c -> (b p t) c')
    y = einops.rearrange(y,'b p t c -> (b p t) c')
    mask = einops.rearrange(mask,'b p t c -> (b p t) c')
    loss = F.cross_entropy(y,x,reduction='none')
    # mask loss
    loss = loss[...,None] * mask
    # compute mean
    loss = loss.mean()
    return loss
