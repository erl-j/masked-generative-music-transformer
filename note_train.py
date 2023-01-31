#%%
from note_dataset import NoteDataset

import numpy as np
import torch
import matplotlib.pyplot as plt
import einops
import torchtext
from torch import nn
import os
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from note_model import TransformerModel
import glob
import datetime
from collections import OrderedDict

# make generic so that it accepts a graph with dependencies between the various events/sections
# recursive perhaps?

def special_loss(logits,target,mask,debug=False):
    merged_target = merge_channels(target)
    merged_mask = merge_channels(mask)
    merged_logits = merge_channels(logits)
    loss=0
    merged_masked_target = merged_target*(1-merged_mask)+(merged_mask*-1)
    # matrix of shape (batch_size, pred_timesteps, target_timesteps) 
    # where each element is 1 if the masked target is equal and 0 otherwise
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

        # expand the target and logits to have shape (batch_size, pred_timesteps, target_timesteps, n_channels)
        
        # calculate cross entropy for every combination of target and prediction
        expanded_target = target[channel][:,:,None,:].expand(-1,-1,n_timesteps,-1).reshape(batch_size*n_timesteps*n_timesteps,-1)
        expanded_logits = logits[channel][:,None,:,:].expand(-1,n_timesteps,-1,-1).reshape(batch_size*n_timesteps*n_timesteps,-1)
        channel_loss = F.cross_entropy(expanded_logits, expanded_target, reduction="none")

        # reshape the loss to have shape (batch_size, pred_timesteps, target_timesteps)
        channel_loss = channel_loss.reshape(batch_size,n_timesteps,n_timesteps)
       
        is_masked = mask[channel].sum(dim=-1)==mask[channel].shape[-1]

        loss+=torch.mean(channel_loss*loss_mask.float()*is_masked[:,:,None].float())
    return loss

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

class Model(pl.LightningModule):
    def __init__(self, token_sections, n_layers=None,n_hidden_size=None, masking_mode="channel"):
        super().__init__()

        self.token_sections = token_sections
        self.total_size = sum([section["n_channels"] for section in self.token_sections])

        self.model = TransformerModel(n_channels=self.total_size,n_layers=n_layers,n_hidden_size=n_hidden_size)

        self.masking_mode = masking_mode

    def forward(self,x,mask):
        y = self.model(x,mask)
        return y
    
    def compute_loss(self, logits, targets, mask):
        return special_loss(logits,targets,mask)

    def training_step(self, batch, batch_idx):
        # get data
        seq = batch["seq"]

        # batch size
        batch_size, sequence_length, _  = next(iter(seq.items()))[1].shape
        device = next(iter(seq.items()))[1].device

        n_sections = len(seq)

        mask_ratio = self.schedule(torch.rand(batch_size,device=device))

        if self.masking_mode == "channel":
            # compute mask ratio
            mask_ratio = einops.repeat(mask_ratio,'b -> b sequence sections',sequence=sequence_length,sections=n_sections)
            # compute mask
            section_mask = (torch.rand(mask_ratio.shape,device=mask_ratio.device) < mask_ratio).float()

            mask = OrderedDict()
            for i, key, value in zip(range(n_sections),seq.keys(),seq.values()): 
                mask[key]=section_mask[...,i].unsqueeze(-1).expand(value.shape)
        elif self.masking_mode == "full":
            mask_ratio = einops.repeat(mask_ratio,'b -> b sequence full',sequence=sequence_length,full=self.total_size)
            section_mask = (torch.rand(mask_ratio.shape,device=mask_ratio.device) < mask_ratio).float()
            mask = split_channels(section_mask,self.token_sections)

        merged_seq = merge_channels(seq,self.total_size)
        merged_mask = merge_channels(mask,self.total_size)
        # compute loss
        
        merged_logits = self(merged_seq,merged_mask)
        # time iteration
        timea = datetime.datetime.now()

        logits = split_channels(merged_logits,self.token_sections)

        loss = self.compute_loss(logits,seq,mask)
        timeb = datetime.datetime.now()
        # print(timeb-timea)
        # log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def schedule(self,progress_ratio):
        if not isinstance(progress_ratio,torch.Tensor):
            progress_ratio = torch.tensor(progress_ratio,device=self.device)
        return torch.cos((np.pi/2.0)*progress_ratio)

    def inverse_schedule(self,mask_ratio):
        if not isinstance(mask_ratio,torch.Tensor):
            mask_ratio = torch.tensor(mask_ratio,device=self.device)
        return torch.acos(mask_ratio)/(np.pi/2.0)

    def generate_with_channel_mode(self,x,section_mask,temperature,n_sampling_steps,plot):
        if x is None:
            n_timesteps = 128
        else:
            n_timesteps = next(iter(x.items()))[1].shape[1]
        if x is None:
            x = torch.zeros((1,n_timesteps,self.total_size),device=self.device)
            x = split_channels(x,self.token_sections)
        if section_mask is None:
            section_mask = torch.ones((1,n_timesteps,len(x)),device=self.device)

        n_cells=n_timesteps*len(x)

        if n_sampling_steps==-1:
            is_autoregressive=True
            step=0
            n_sampling_steps= n_cells
        else:
            is_autoregressive=False
            mask_ratio = torch.mean(section_mask,dim=[1,2])
            step = int(self.inverse_schedule(mask_ratio)*n_sampling_steps)
        
        batch_size,n_timesteps, _= next(iter(x.items()))[1].shape

        with torch.no_grad():
            for step in tqdm(range(step,n_sampling_steps)):
                
                n_masked = int(torch.sum(section_mask).item())
                if is_autoregressive:
                    n_masked_next = n_masked-1
                else:
                    n_masked_next = int(np.ceil(self.schedule((step+1)/n_sampling_steps).cpu()*n_cells))

                # print n_masked_next, n_masked
                # print(f"step {step}/{n_sampling_steps} - {n_masked_next-n_masked} cells to mask")
                # print(f"n_masked_next {n_masked_next} - n_masked {n_masked}")

                mask = OrderedDict()
                for i, key, value in zip(range(len(x)),x.keys(),x.values()): 
                    mask[key]=section_mask[...,i].unsqueeze(-1).expand(value.shape)
                merged_mask = merge_channels(mask,self.total_size)
                merged_x = merge_channels(x,self.total_size)

                merged_logits = self(merged_x,merged_mask)

                logits= split_channels(merged_logits,self.token_sections)

                y_probs = OrderedDict()
                for key, value in logits.items():
                    y_probs[key] = torch.softmax(value/temperature,dim=-1)
                
                # get list of indices of section mask == 1
                masked_indices = section_mask[0].nonzero()


                #index_to_unmask = masked_indices[np.random.randint(len(masked_indices))]
                indices_to_unmask = masked_indices[np.random.choice(len(masked_indices),n_masked-n_masked_next,replace=False)]

                for index_to_unmask in indices_to_unmask:
                    timestep = index_to_unmask[0]
                    key = list(x.keys())[index_to_unmask[1]]

                    # sample from distribution
                    probs = y_probs[key][:,timestep]

                    sampled_index = torch.distributions.Categorical(probs=probs).sample([batch_size])
                    one_hot = torch.zeros_like(probs)
                    one_hot[:,sampled_index] = 1

                    x[key][:,timestep] = one_hot
                    section_mask[:,index_to_unmask[0],index_to_unmask[1]] = 0
                
                if plot:
                    plt.imshow(section_mask[0].cpu().numpy(),aspect="auto",cmap="gray",interpolation="none")
                    plt.show()

        return x

    def generate(self,x=None, section_mask=None, temperature=1.0, n_sampling_steps=None,mode=None,plot=False):
        if mode=="channel":
            return self.generate_with_channel_mode(x,section_mask,temperature,n_sampling_steps,plot)
            
    def configure_optimizers(self):
        return self.model.configure_optimizers()

class DemoCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)


    ds = NoteDataset(prepared_data_path="data/prepared_vast+gamer_noteseq_data.pt", crop_size=36)
    #ds = NoteDataset(prepared_data_path="data/prepared_gamer_noteseq_data_1000.pt", crop_size=36)

    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers=20)

    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[],log_every_n_steps=1,gpus=[1],max_epochs=-1)
    model = Model(token_sections=ds.get_token_sections(),n_layers=4,n_hidden_size=512)


    trainer.fit(model, dl)

        






    
# # %%

# %%
#
