import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from src.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
    

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

#def get_torch_trans(heads=8, layers=1, channels=64):
#    encoder_layer = TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64)
#    return TransformerEncoder(encoder_layer, num_layers=layers)



def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # Weight initialization
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    # t_embedding(t). The embedding dimension is 128 in total for every time step t.
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_MECSDI(nn.Module):
    def __init__(self, config, inputdim=2, num_steps=0, matched_state=0):
        super().__init__()
        self.config = config
        self.channels = config["channels"]
        if matched_state > 1:
            num_steps = matched_state
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=config["diffusion_embedding_dim"],
        )
        
        #self.diffusion_embedding_me = DiffusionEmbedding(num_seps, embedding_dim=config["diffusion_embedding_dim"])
        self.token_emb_dim = 1
        inputdim = 2 * self.token_emb_dim

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.token_emb_dim, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K = x.shape
        
        x = x.reshape(B, inputdim, K)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K)
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K)

        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
    
    def forward_feature(self, y, base_shape):
        B, channel, K = base_shape
        y = y.reshape(B, channel, K).permute(0, 1, 2).reshape(B, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, channel, K).permute(0, 1, 2).reshape(B, channel, K)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K)
        
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
       
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip



