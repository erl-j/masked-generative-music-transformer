import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from unet import UNet2D

def get_fixed_embeddings(n_pitches,n_timesteps,padding_channels=0):
    # pitch
    chroma_embedding = torch.arange(0,n_pitches)%12 / 12
    octave_embedding = torch.floor(torch.arange(0,n_pitches)//12)/(n_pitches//12)
    pitch = torch.arange(0,n_pitches)/n_pitches
    even = torch.arange(0,n_pitches)%2

    chroma = einops.repeat(chroma_embedding,'p -> p t',t=n_timesteps)
    octave = einops.repeat(octave_embedding,'p -> p t',t=n_timesteps)
    pitch = einops.repeat(pitch, 'p -> p t',t=n_timesteps)
    even = einops.repeat(even, 'p -> p t',t=n_timesteps)

    fixed_embeddings = {
        "chroma":chroma,
        "octave":octave,
        "pitch":pitch,
        "even":even
    }
    periods =[2,4,6,8,16,32]
    if n_timesteps > 32:
        periods.append(64)

    for period in periods:
        time_embedding = (torch.arange(0,n_timesteps)%period)/period
        time = einops.repeat(time_embedding,'t -> p t',p=n_pitches)
        fixed_embeddings[f"time_{period}"] = time

    for key, value in fixed_embeddings.items():
        plt.imshow(value,aspect='auto')
        plt.title(key)
        os.makedirs("artefacts/plots/fixed_embeddings",exist_ok=True)
        plt.savefig(f"artefacts/plots/fixed_embeddings/{key}.png")

    fixed_embeddings  = torch.stack([v for k,v in fixed_embeddings.items()],dim=-1)

    return fixed_embeddings

def flatten_roll(pr):
    if len(pr.shape) == 3:
        flat_pr = einops.rearrange(pr,'b p t -> b (p t)')
    elif len(pr.shape) == 2:
        flat_pr = einops.rearrange(pr,'p t -> (p t)')
    return flat_pr

def unflatten_piano_roll(flat_pr, n_pitches, n_timesteps):
    if len(flat_pr.shape) == 2:
        pr = einops.rearrange(flat_pr,'b (p t) -> b p t',p=n_pitches,t=n_timesteps)
    elif len(flat_pr.shape) == 1:
        pr = einops.rearrange(flat_pr,'(p t) -> p t',p=n_pitches,t=n_timesteps)
    return pr


class UnetModel(torch.nn.Module):
    def __init__(self, n_pitches, n_timesteps, n_hidden_size, conv_depths):
        super().__init__()
        self.n_pitches = n_pitches
        self.n_timesteps = n_timesteps
        self.fixed_embeddings = get_fixed_embeddings(n_pitches=n_pitches,n_timesteps=n_timesteps, padding_channels=0)
        self.n_total_embedding_channels = n_hidden_size
        self.n_vocab_channels=2
        self.n_mask_channels=1
        self.n_additional_embedding_channels = self.n_total_embedding_channels-self.n_vocab_channels-self.n_mask_channels-self.fixed_embeddings.shape[-1]
        self.n_heads = 1
      
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.n_total_embedding_channels-self.n_additional_embedding_channels, self.n_additional_embedding_channels),
            nn.Sigmoid(),
        )
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_total_embedding_channels, dim_feedforward=self.n_total_embedding_channels, nhead=self.n_heads, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.unet_encoder = UNet2D(in_channels=self.n_total_embedding_channels,out_channels=self.n_total_embedding_channels,conv_depths=conv_depths)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_total_embedding_channels, self.n_vocab_channels),
        )
    
    def forward(self,x,mask):

        batch_size = x.shape[0]

        x = einops.rearrange(x,'b p t c -> b (p t) c')
        mask = einops.rearrange(mask,'b p t c -> b (p t) c')

        x = torch.cat([x*(1-mask),mask],dim=-1)

        fixed_embeddings=einops.repeat(self.fixed_embeddings, "p t c-> b (p t) c", b=batch_size).to(x.device)
    
        x = torch.concat([x,fixed_embeddings],dim=-1)
        embed = self.embedding_layer(x)
        
        x = torch.concat([x,embed],dim=-1)

        x = einops.rearrange(x,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)

        x = self.unet_encoder(x)
        y = self.output_layer(x)

        y_prob = F.softmax(y,dim=-1)

        # y = einops.rearrange(y,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)
        # y_prob = einops.rearrange(y_prob,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)

        return y, y_prob
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

class TransformerModel(torch.nn.Module):
    def __init__(self, n_pitches,n_timesteps,n_layers,n_hidden_size):
        super().__init__()
        self.n_pitches = n_pitches
        self.n_timesteps = n_timesteps
        self.fixed_embeddings = get_fixed_embeddings(n_pitches=n_pitches,n_timesteps=n_timesteps, padding_channels=0)
        self.n_total_embedding_channels = n_hidden_size
        self.n_vocab_channels=2
        self.n_mask_channels=1
        self.n_additional_embedding_channels = self.n_total_embedding_channels-self.n_vocab_channels-self.n_mask_channels-self.fixed_embeddings.shape[-1]
        self.n_heads = 1
        self.n_layers= n_layers
      
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.n_total_embedding_channels-self.n_additional_embedding_channels, self.n_additional_embedding_channels),
            nn.Sigmoid(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_total_embedding_channels, dim_feedforward=self.n_total_embedding_channels, nhead=self.n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_total_embedding_channels, self.n_vocab_channels),
        )
       
    def forward(self,x,mask):

        batch_size = x.shape[0]

        x = einops.rearrange(x,'b p t c -> b (p t) c')
        mask = einops.rearrange(mask,'b p t c -> b (p t) c')

        attention_mask_mode ="mask"
        if attention_mask_mode == "off notes":
            attention_mask = x[:,:,1]
            attention_mask = einops.repeat(attention_mask, 'b t1 -> b t2 t1',t2=x.shape[1],b=batch_size).bool()
            attention_mask = attention_mask &  (1- torch.eye(attention_mask.shape[1],attention_mask.shape[2],device=x.device)).bool()[None,...]
        elif attention_mask_mode == "mask":
            m = mask[:,:,0]
            attention_mask = einops.repeat(m, 'b t1 -> b t2 t1',t2=x.shape[1],b=batch_size).bool()
            attention_mask = attention_mask & (1- torch.eye(attention_mask.shape[1],attention_mask.shape[2],device=x.device)).bool()[None,...]
        elif attention_mask_mode =="none":
            attention_mask = torch.zeros((batch_size,x.shape[1],x.shape[1]),device=x.device).bool()
        elif attention_mask_mode == "off notes & mask":
            m = mask[:,:,0]
            mask_attention_mask = einops.repeat(m, 'b t1 -> b t2 t1',t2=x.shape[1],b=batch_size).bool()
            attention_mask = x[:,:,1]
            off_attention_mask = einops.repeat(attention_mask, 'b t1 -> b t2 t1',t2=x.shape[1],b=batch_size).bool()
            attention_mask = mask_attention_mask | off_attention_mask
            attention_mask = attention_mask & (1- torch.eye(attention_mask.shape[1],attention_mask.shape[2],device=x.device)).bool()[None,...]

        x = torch.cat([x*(1-mask),mask],dim=-1)

        fixed_embeddings=einops.repeat(self.fixed_embeddings, "p t c-> b (p t) c", b=batch_size).to(x.device)
    
        x = torch.concat([x,fixed_embeddings],dim=-1)
        embed = self.embedding_layer(x)
        x = torch.concat([x,embed],dim=-1)

        x = self.transformer_encoder(x, attention_mask)
        y = self.output_layer(x)

        y_prob = F.softmax(y,dim=-1)

        y = einops.rearrange(y,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)
        y_prob = einops.rearrange(y_prob,'b (p t) c -> b p t c',p=self.n_pitches,t=self.n_timesteps)

        return y, y_prob

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

class MLPModel(torch.nn.Module):

    def __init__(self,n_pitches,n_timesteps) -> None:
        super().__init__()
        self.fixed_embeddings = get_fixed_embeddings(n_pitches=n_pitches,n_timesteps=n_timesteps, padding_channels=0)

        self.n_fixed_embeddings_channels = self.fixed_embeddings.shape[-1]
        self.n_vocab_channels=2
        self.n_mask_channels=1
        self.n_total_embedding_channels = 512
        self.n_additional_embedding_channels=self.n_total_embedding_channels-(self.n_vocab_channels+self.n_mask_channels+self.n_fixed_embeddings_channels)
        self.n_layers=3
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.n_vocab_channels+self.n_mask_channels+self.n_fixed_embeddings_channels, self.n_additional_embedding_channels),
            nn.ReLU(),
        )

        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(self.n_total_embedding_channels,self.n_total_embedding_channels),nn.ReLU(),nn.LayerNorm(self.n_total_embedding_channels))
            for _ in range(self.n_layers)])

        self.output_layer = nn.Sequential(
            nn.Linear(self.n_total_embedding_channels+ self.n_fixed_embeddings_channels, self.n_vocab_channels),
        )

    def forward(self, x, mask):
        batch_size = x.shape[0]
        masked_x = torch.cat([x*(1-mask),mask],dim=-1)

        fixed_embeddings=einops.repeat(self.fixed_embeddings, "p t c-> b p t c", b=batch_size).to(x.device)

        masked_x_w_fixed_embeddings = torch.cat([masked_x,fixed_embeddings],dim=-1)

        x_embed = self.embedding_layer(masked_x_w_fixed_embeddings)

        x = torch.cat([masked_x_w_fixed_embeddings,x_embed],dim=-1)

        for linear in self.linears:
            xout = linear(x)
            global_context = torch.mean(xout*(1-mask),dim=[1,2],keepdim=True)
            x = xout + global_context + x

        x = torch.cat([x,fixed_embeddings],dim=-1)
        y = self.output_layer(x)

        y_prob = F.softmax(y,dim=-1)

        return y, y_prob

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer