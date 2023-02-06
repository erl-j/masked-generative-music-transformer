import torch
import torch.nn as nn

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
        masked_x = (1-mask)*x - mask

        embed = self.embedding_layer(masked_x)

        x = torch.cat([x,embed],dim=-1)

        x = self.transformer_encoder(x)
        y = self.output_layer(x)

        return y

   
    