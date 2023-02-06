#%%
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from note_util import merge_channels, split_channels

def special_loss_slow(logits,target,mask,debug=False):

    merged_target = merge_channels(target)
    merged_mask = merge_channels(mask)
    merged_logits = merge_channels(logits)

    loss=0

    batch_size,n_timesteps, _ = merged_logits.shape

    masked_target = merged_target*(1-merged_mask)-merged_mask

    masked_target_is_equal = (masked_target.unsqueeze(1)==masked_target.unsqueeze(2)).all(dim=-1)

    for pred_timestep in range(n_timesteps):
        target_mask_is_equal = masked_target_is_equal[:,pred_timestep,:]
        target_is_note = target["type"][:,:,0]==1

        for channel in logits.keys():

            if mask[channel][:,pred_timestep].sum()>0:
                step_logit = logits[channel][:,pred_timestep][:,None,:].repeat(1,n_timesteps,1)

                cross_entropy = F.cross_entropy(step_logit,target[channel],reduction="none")

                if channel == "type":
                    loss_mask = target_mask_is_equal[:,:,None]
                else:
                    loss_mask = target_is_note[:,:,None] * target_mask_is_equal[:,:,None]

                cross_entropy  = cross_entropy * loss_mask

                if cross_entropy.sum()>0:
                    loss += cross_entropy.sum()/(loss_mask.sum()*n_timesteps)
    return loss

# parralelize this so every timestep is done at once

def special_loss_parallel(logits,target,mask,debug=False):
    merged_target = merge_channels(target)
    merged_mask = merge_channels(mask)
    merged_logits = merge_channels(logits)

    loss=0

    batch_size,n_timesteps, _ = merged_logits.shape

    masked_target = merged_target*(1-merged_mask)-merged_mask

    masked_target_is_equal = (masked_target.unsqueeze(1)==masked_target.unsqueeze(2)).all(dim=-1)

    target_is_note = target["type"][:,:,0]==1

    for channel in logits.keys():

        # compute the cross entropy between every logit and every target
        cross_entropy = F.cross_entropy(
            logits[channel][:,:,None,:].repeat(1,1,n_timesteps,1).reshape(batch_size*n_timesteps*n_timesteps,-1),
            target[channel][:,None,:,:].repeat(1,n_timesteps,1,1).reshape(batch_size*n_timesteps*n_timesteps,-1),
            reduction="none").reshape(batch_size,n_timesteps,n_timesteps,-1)

        is_masked = mask[channel][:,:,None,:].repeat(1,1,n_timesteps,1)

        # compute the loss mask
        if channel == "type":
            loss_mask = masked_target_is_equal[...,None]*is_masked
        else:
            loss_mask = target_is_note[...,None,None] * masked_target_is_equal[...,None]*is_masked
        
        # apply the loss mask
        cross_entropy  = cross_entropy * loss_mask

        # sum the loss
        if cross_entropy.sum()>0:
            loss += cross_entropy.sum()/(loss_mask.sum())
        
    return loss

def special_loss(logits,target,mask,debug=False):

    merged_target = merge_channels(target)
    merged_mask = merge_channels(mask)
    merged_logits = merge_channels(logits)

    loss=0

    merged_masked_target = merged_target*(1-merged_mask)-merged_mask

    masked_target_is_equal = (merged_masked_target.unsqueeze(1)==merged_masked_target.unsqueeze(2)).all(dim=-1)

    is_note = target["type"][:,:,0]==1
    batch_size,n_timesteps, _ = merged_logits.shape
    
    if debug:
        for i in range(n_timesteps):
            plt.title("merged_target")
            plt.imshow(merged_target[0].detach().cpu().numpy().T, cmap="gray", interpolation="none",aspect="auto", extent=(0,n_timesteps,0,merged_logits.shape[-1]))
            plt.show()
        
            plt.title("merged_mask")
            plt.imshow(merged_mask[0].detach().cpu().numpy().T, cmap="gray", interpolation="none",aspect="auto", extent=(0,n_timesteps,0,merged_logits.shape[-1]))
            plt.show()

            plt.title("merged_masked_target")
            plt.imshow(merged_masked_target[0].detach().cpu().numpy().T, cmap="gray", interpolation="none",aspect="auto", extent=(0,n_timesteps,0,merged_logits.shape[-1]))
            # show grid every integer
            plt.xticks(np.arange(0,n_timesteps+1,1.0))
            plt.yticks(np.arange(0,merged_logits.shape[-1]+1,1.0))
            plt.grid(True)
            # make ticks integers
            # add a green arrow to show the current timestep
            plt.arrow(i+0.25,-1,0,0.5,head_width=0.25, head_length=0.5, color="green")

            # add green arrow to every timestep where the masked target is equal
            for j in range(n_timesteps):
                if masked_target_is_equal[0,i,j]:
                    plt.arrow(j+0.75,-1,0,0.5,head_width=0.25, head_length=0.5, color="blue")

            # remove the ticks on the y axis and x axis
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                left=False,         # ticks along the left edge are off
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False)
                
            plt.show()

    for channel in logits.keys():
        if channel == "type":
            loss_mask = masked_target_is_equal
        else:
            loss_mask = masked_target_is_equal*is_note[:,:,None]

        expanded_target = target[channel][:,None,:,:].expand(-1,n_timesteps,-1,-1).reshape(batch_size*n_timesteps*n_timesteps,-1)
        expanded_logits = logits[channel][:,:,None,:].expand(-1,-1,n_timesteps,-1).reshape(batch_size*n_timesteps*n_timesteps,-1)
        channel_loss = F.cross_entropy(expanded_logits, expanded_target, reduction="none")

        # reshape the loss to have shape (batch_size, pred_timesteps, target_timesteps)
        channel_loss = channel_loss.reshape(batch_size,n_timesteps,n_timesteps)
       
        is_masked = mask[channel].sum(dim=-1)==mask[channel].shape[-1]

        loss+=torch.sum(channel_loss*loss_mask.float()*is_masked[:,:,None].float())/torch.sum(loss_mask.float()*is_masked[:,:,None].float())
    return loss