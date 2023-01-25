import os

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(torch.nn.Module):
    def __init__(self, n_channels,n_layers,n_hidden_size):
        super().__init__()

        self.n_total_embedding_channels = n_hidden_size
        self.n_channels=n_channels
        self.n_additional_embedding_channels = self.n_total_embedding_channels-self.n_channels
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
            nn.Linear(self.n_total_embedding_channels, self.n_channels),
        )
       
    def forward(self,x,mask):
        batch_size = x.shape[0]

        # set masked to -1
        masked_x = (1-mask)*x + mask*-1

        embed = self.embedding_layer(masked_x)

        x = torch.cat([x,embed],dim=-1)

        x = self.transformer_encoder(x)
        y = self.output_layer(x)

        y_prob = torch.zeros_like(y)

        offset=0
        # for section in self.token_sections:
        #     y_prob=F.softmax(y[:,:,section[offset:offset+section.n_channels]],dim=-1)

        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

# def token_loss(target_indices,logits,section_mask,token_sections=[{"null",2},{"pitch",36},{"onset_time",32},{"duration",32}]):
#     # target: (batch token channel_dist), logits: (batch token channel_dist), mask: (batch token channel)
#     loss=0

#     expanded_mask = []
#     for ch_idx, channel_vocab_size in token_sections:
#         expanded_mask.append(mask[:,:,ch_idx].unsqueeze(-1).expand(-1,-1,channel_vocab_size))
#     expanded_mask = torch.cat(expanded_mask,dim=-1)

#     for batch in range(target_indices.shape[0]):
#         # filter predicted tokens.
#         # filter away non masked tokens and maksked tokens in null tokens

#         indices=[]
#         for index in target_indices.shape[1]:
#             # filter away tokens that are not masked
#             if (section_mask[batch,predicted_token_index] == 0).all():
#                     continue
#             # filter away tokens where null is true and unmasked
#             if (target_indices[batch,predicted_token_index,0] == 0) and (section_mask[batch,predicted_token_index,0] == 0):
#                 continue
#             indices.append(index)

#         for predicted_token_index in indices:
#             # check if there is a mask
#                 for target_token_index in indices:
#                     # check if target is a candidate
#                     # if unmasked channels are equal then it is a candidate
#                     candidate_list = []
#                     if (section_mask[batch,target_token_index] == section_mask[batch,predicted_token_index]).all():
#                         if((target_indices[batch,target_token_index]+1)*(1-section_mask) == (target_indices[batch,predicted_token_index]+1)*(1-section_mask)).all():
#                             candidate_list.append(target_token_index)
            
#                     for channel_idx, channel_vocab_size in token_sections:
#                         if section_mask[batch,target_token_index,channel_idx] == 1:
#                             one_hots=[]
#                             for candidate in candidate_list:
#                                 # check if target is a match
#                                 # if target is a match then add to candidate list
#                                 one_hot = torch.zeros(channel_vocab_size, device=target_indices.device)
#                                 one_hot[target_indices[batch,target_token_index,channel_idx]] = 1
#                                 one_hots.append(one_hot)

#                             one_hot_mean = torch.mean(torch.stack(one_hots),dim=0)
#                             loss += torch.nn.functional.cross_entropy(logits[batch,predicted_token_index,channel_idx],one_hot_mean)
#     return loss
    
    # if target is a match then add to candidate list
                        


                    
