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

def special_loss(logits,target,mask):
    merged_target = merge_channels(target)
    merged_mask = merge_channels(mask)
    merged_logits = merge_channels(logits)

    loss=0
    merged_masked_target = merged_target*(1-merged_mask)-merged_mask

    # matrix of shape (batch_size, pred_timesteps, target_timesteps) 
    # where each element is 1 if the masked target is equal and 0 otherwise
    masked_target_is_equal = (merged_masked_target.unsqueeze(1)==merged_masked_target.unsqueeze(2)).all(dim=-1)
    batch_size ,n_timesteps, _ = merged_logits.shape
    for channel in logits.keys():
        expanded_target = target[channel][:,None,...]*masked_target_is_equal[...,None]
        target_mean = torch.mean(expanded_target,dim=2)
        channel_loss = torch.nn.functional.cross_entropy(
            logits[channel].reshape((batch_size*n_timesteps,-1)),
            target_mean.reshape((batch_size*n_timesteps,-1)),
            reduction="none",
        ).reshape((batch_size,n_timesteps))
        is_masked = mask[channel].sum(dim=-1)==mask[channel].shape[-1]
        loss+=torch.mean(channel_loss*is_masked[:,None])
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

    def generate_with_channel_mode(self,x,section_mask,temperature,plot):
        if x is None:
            n_timesteps = 128
        else:
            n_timesteps = next(iter(x.items()))[1].shape[1]
        if x is None:
            x = torch.zeros((1,n_timesteps,self.total_size),device=self.device)
            x = split_channels(x,self.token_sections)
        if section_mask is None:
            section_mask = torch.ones((1,n_timesteps,len(x)),device=self.device)

        batch_size,n_timesteps, _= next(iter(x.items()))[1].shape

        n_masked =  int(torch.sum(section_mask).item())

        n_cells = n_timesteps*len(x)

        with torch.no_grad():
            for step in range(n_cells-n_masked,n_cells):
                
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
                index_to_unmask = masked_indices[0]

                timestep = index_to_unmask[0]
                key = list(x.keys())[index_to_unmask[1]]

                # sample from distribution
                probs = y_probs[key][:,timestep]

                sampled_index = torch.distributions.Categorical(probs=probs).sample([batch_size])
                one_hot = torch.zeros_like(probs)
                one_hot[:,sampled_index] = 1

                x[key][0,timestep] = one_hot
                section_mask[:,index_to_unmask[0],index_to_unmask[1]] = 0

                # plt.imshow(section_mask[0].cpu().numpy(),aspect="auto")
                # plt.show()
        return x

    def generate(self,x=None, section_mask=None, temperature=1.0, mode=None,plot=False):
        if mode=="channel":
            return self.generate_with_channel_mode(x,section_mask,temperature,plot)
            
    def configure_optimizers(self):
        return self.model.configure_optimizers()

class DemoCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)


    ds = NoteDataset(prepared_data_path="data/prepared_gamer_noteseq_data.pt", crop_size=36)

    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers=20)

    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[],log_every_n_steps=1,gpus=[1])
    model = Model(token_sections=ds.get_token_sections(),n_layers=4,n_hidden_size=512)


    trainer.fit(model, dl)

        






    
# # %%

# %%
#
