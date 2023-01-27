#%%
from midi_dataset import NoteSeqDataset

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
# make generic so that it accepts a graph with dependencies between the various events/sections
# also rewrite with better use of tensor operations.
# recursive perhaps?

def special_loss(logits,target,mask,token_sections=[{"label":"null","n_channels":1}]):
    loss=0
    masked_target = target*(1-mask)-mask
    # matrix of shape (batch_size, pred_timesteps, target_timesteps) 
    # where each element is 1 if the masked target is equal and 0 otherwise
    masked_target_is_equal = (masked_target.unsqueeze(1)==masked_target.unsqueeze(2)).all(dim=-1)
    target_sum = target[:,None,:, :]*masked_target_is_equal[...,None]
    target_mean = torch.mean(target_sum,axis=1)

    channel_offset=0

    logits = einops.rearrange(logits, "batch sequence channel -> (batch sequence) channel")
    target_mean=einops.rearrange(target_mean, "batch sequence channel -> (batch sequence) channel")
    mask = einops.rearrange(mask, "batch sequence channel -> (batch sequence) channel")

    for i in range(len(token_sections)):
        section_loss = torch.nn.functional.cross_entropy(
            logits[...,channel_offset:token_sections[i]["n_channels"]+channel_offset],
        target_mean[...,channel_offset:token_sections[i]["n_channels"]+channel_offset],
       reduction="none")
        section_loss = section_loss * (torch.sum(mask[...,channel_offset:channel_offset+token_sections[i]["n_channels"]],axis=-1)==token_sections[i]["n_channels"])
        loss+=torch.mean(section_loss)
        channel_offset += token_sections[i]["n_channels"]
    return loss
      

class Model(pl.LightningModule):
    def __init__(self, token_sections, n_layers=None,n_hidden_size=None):
        super().__init__()

        self.token_sections = token_sections
        self.n_channels = sum([section["n_channels"] for section in self.token_sections])

        self.model = TransformerModel(n_channels=self.n_channels,n_layers=n_layers,n_hidden_size=n_hidden_size)

    def forward(self,x,mask):
        y = self.model(x,mask)
        return y
    
    def compute_loss(self, logits, targets, mask):
        return special_loss(logits,targets,mask, self.token_sections)

    def training_step(self, batch, batch_idx):

        # get data
        seq = batch["seq"]
        # batch size
        batch_size = seq.shape[0]

        n_sections = len(self.token_sections)

        # compute mask ratio
        mask_ratio = self.schedule(torch.rand(batch_size,device=seq.device))
        mask_ratio = einops.repeat(mask_ratio,'b -> b sequence sections',sequence=seq.shape[1],sections=n_sections)
        # compute mask
        section_mask = (torch.rand(mask_ratio.shape,device=mask_ratio.device) < mask_ratio).float()

        channel_offset=0
        mask = []
        for i in range(n_sections):
            mask.append(section_mask[:,:,i].unsqueeze(-1).repeat(1,1,self.token_sections[i]["n_channels"]))
        mask = torch.concat(mask,axis=-1)

        assert mask.shape == seq.shape 
        
        # compute loss
        y = self(seq,mask)

        # time iteration
        timea = datetime.datetime.now()

        loss = self.compute_loss(y, seq,mask)

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

    # def generate(self, n_sampling_steps, temperature=1.0, x= None,mask=None, activity_bias=0, use_confidence_sampling=True, plot=False):
    #     if x is None:
    #         x = torch.zeros((1,self.n_pitches, self.n_timesteps,self.vocab_size),device=self.device)
    #         x[:,:,:,1] = 1
    #     else:
    #         x  = piano_roll_to_model_format(x)
    
    #     if mask is None:
    #         mask = torch.ones((1,self.n_pitches, self.n_timesteps,1),device=self.device)
    #     else:
    #         mask = (mask)[...,None]

    #     mask_ratio = torch.mean(mask,dim=[1,2])
    #     step = int(self.inverse_schedule(mask_ratio)*n_sampling_steps)

    #     # Inference algo from https://arxiv.org/pdf/2202.04200.pdf
    #     with torch.no_grad():

    #         for step in range(step,n_sampling_steps):

    #             y,_ = self(x,mask)

    #             y = einops.rearrange(y,'b p t c -> b (p t) c')

    #             y[:,:,0] += activity_bias

    #             probs = torch.softmax(y/temperature,dim=-1)

    #             # n tokens to mask
    #             n_mask = int(np.floor(self.schedule((step+1)/n_sampling_steps).cpu()*self.n_pitches * self.n_timesteps))

    #             # sample from probs
    #             sampled_indices = torch.distributions.Categorical(probs=probs).sample()

    #             # turn indices into one-hot
    #             sample = torch.zeros_like(probs)
    #             sample.scatter_(-1,sampled_indices[...,None],1)

    #             assert torch.all(torch.sum(sample,dim=-1) == 1)

    #             # confidence of indices (unmasked indices have confidence 1)
    #             confidences = torch.sum(sample * probs,axis=-1)

    #             flat_mask = einops.rearrange(mask,'b p t c -> b (p t) c')

    #             confidences = confidences * flat_mask[...,0] + (1-flat_mask[...,0])
                
    #             # get confidence of n_mask:th lowest confidence
    #             confidence_threshold = torch.sort(confidences,dim=-1)[0][:,n_mask]

    #             flat_x = einops.rearrange(x,'b p t c -> b (p t) c')

    #             if plot == True:

    #                 os.makedirs("artefacts/gif",exist_ok=True)
    #                 # globally set colormap to viridis
    #                 plt.rcParams['image.cmap'] = 'inferno'

    #                 # hide axes

    #                 # global title for all subplots
    #                 plt.suptitle(f"step {step}")
                    
    #                 fig,ax = plt.subplots(1,5,figsize=(15,5))
    #                 ax[0].imshow(flat_x[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #                 ax[0].set_title("piano roll")

    #                 ax[1].imshow(flat_mask[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #                 ax[1].set_title("mask")

    #                 ax[2].imshow(probs[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #                 ax[2].set_title(f"probability of note")

    #                 ax[3].imshow(sample[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #                 ax[3].set_title(f"sampled outcome")

    #                 ax[4].imshow(confidences[0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #                 ax[4].set_title(f"confidence of sample")

    #                 # hide axes
    #                 for a in ax:
    #                     a.axis('off')

    #                 # global title for entire figure
    #                 fig.suptitle(f"step {step+1}")

    #                 plt.savefig(f"artefacts/gif/plot_{step}.png")
    #                 plt.show()


    #             # get sample 
    #             sample = flat_mask * sample + (1-flat_mask) * flat_x

    #             if use_confidence_sampling:
    #                 new_mask = (confidences <  confidence_threshold[...,None])[...,None].float()
    #             else:
    #                 new_mask = flat_mask

    #                 # get number of masked tokens
    #                 n_current_mask = int(torch.sum(new_mask))

    #                 # get indices that are currently masked
    #                 masked_indices = torch.where(flat_mask[...,0] == 1)[1]

    #                 n_to_unmask = n_current_mask-n_mask

    #                 # get n_mask random indices
    #                 random_indices = torch.randperm(masked_indices.shape[0])[:n_to_unmask]

    #                 # get indices to unmask
    #                 unmask_indices = masked_indices[random_indices]

    #                 # unmask
    #                 new_mask[:,unmask_indices,:] = 0


                    
    #             flat_x = flat_x*new_mask + (1-new_mask) * sample

    #             assert torch.all(torch.sum(x,axis=-1) == 1)
                
    #             flat_mask = new_mask
    #             mask = einops.rearrange(flat_mask,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)
    #             x = einops.rearrange(flat_x,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)

    #     if plot == True:

    #         # globally set colormap to viridis
        
            
    #         fig,ax = plt.subplots(1,5,figsize=(15,5))
    #         ax[0].imshow(flat_x[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #         ax[0].set_title("piano roll")

    #         ax[1].imshow(flat_mask[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #         ax[1].set_title("mask")

    #         ax[2].imshow(probs[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #         ax[2].set_title(f"probability of note")

    #         ax[3].imshow(sample[0,:,0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #         ax[3].set_title(f"sampled outcome")

    #         ax[4].imshow(confidences[0].cpu().reshape(self.n_pitches,self.n_timesteps),vmin=0,vmax=1)
    #         ax[4].set_title(f"confidence of sample")

    #         # global title for entire figure
    #         fig.suptitle(f"step {step+1}")

    #         # hide axes
    #         for a in ax:
    #             a.axis('off')

    #         plt.savefig(f"artefacts/gif/plot_{step+1}.png")
    #         plt.show()



    #     assert torch.all(mask) == 0

    #     assert torch.all(torch.sum(x,axis=-1) == 1)
    #     return model_format_to_piano_roll(x)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()

class DemoCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)


    ds = NoteSeqDataset(prepared_data_path="data/prepared_gamer_noteseq_data.pt", crop_size=36)

    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers=20)

    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[],log_every_n_steps=1,gpus=[0])
    model = Model(token_sections=ds.get_token_sections(),n_layers=4,n_hidden_size=512)


    trainer.fit(model, dl)

        






    
# # %%

# %%
#
