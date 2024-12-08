import torch
import torch.nn as nn
import typing as ty
import torch.nn.init as nn_init
from torch import Tensor
import math
import numpy as np


class Tokenizer(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
        phase1_num: ty.Optional[ty.List[int]],
        phase2_num: ty.Optional[ty.List[int]], 
        phase1_cat: ty.Optional[ty.List[int]],
        phase2_cat: ty.Optional[ty.List[int]], 
    ) -> None:
        super().__init__()

        d_bias = d_numerical + len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.d_token = d_token
        self.register_buffer("category_offsets", category_offsets)
        
        self.category_embeddings = nn.Embedding(sum(categories) + 1, self.d_token)
        self.category_embeddings.weight.requires_grad = False
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        
        
        for col in range(len(phase1_cat)):
            ind1 = np.cumsum(categories)[phase1_cat[col]] - 1
            ind2 = np.cumsum(categories)[phase2_cat[col]] - 1
            self.category_embeddings.weight[ind1:(ind1 + categories[phase1_cat[col]]), :] = self.category_embeddings.weight[ind2:(ind2 + categories[phase2_cat[col]]), :]
        
        self.weight = nn.Parameter(Tensor(d_numerical, self.d_token)) #init weights for numercic data
        self.weight.requires_grad = False

        self.bias = nn.Parameter(Tensor(d_bias, self.d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #Problem! Some features may actually represents the same thing, but different weights will mitigate their identity.
        for col in range(len(phase1_num)):
            self.weight[phase1_num[col], :] = self.weight[phase2_num[col], :]
            
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            self.bias.requires_grad = False

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        x_cat = x_cat.type(torch.int32)

        assert x_some is not None
        #for all the numeric features
        x = self.weight.T * x_num 

        if x_cat is not None:
            x = x[:, np.newaxis, :, :]
            x = x.permute(0, 1, 3, 2)
            #concat numeric + categorical embedding together
            x = torch.cat([x, self.category_embeddings(x_cat + self.category_offsets[None])], dim=2)
            
        if self.bias is not None:
            x = x + self.bias[None]

        return x

    def recover(self, Batch, d_numerical):
        B, L, K = Batch.shape
        K_new = int(K / self.d_token)
        Batch = Batch.reshape(B, K_new, self.d_token)
        #Batch = Batch - self.bias

        Batch_numerical = Batch[:, :d_numerical, :]
        Batch_numerical = Batch_numerical / self.weight
        Batch_numerical = torch.mean(Batch_numerical, 2, keepdim=False)

        Batch_cat = Batch[:, d_numerical:, :]
        new_Batch_cat = torch.zeros([Batch_cat.shape[0], Batch_cat.shape[1]])
        for i in range(Batch_cat.shape[1]):
            token_start = self.category_offsets[i] + 1
            if i == Batch_cat.shape[1] - 1:
                token_end = self.category_embeddings.weight.shape[0] - 1
            else:
                token_end = self.category_offsets[i + 1]
            emb_vec = self.category_embeddings.weight[token_start : token_end + 1, :]
            for j in range(Batch_cat.shape[0]):
                distance = torch.norm(emb_vec - Batch_cat[j, i, :], dim=1)
                nearest = torch.argmin(distance)
                new_Batch_cat[j, i] = nearest + 1
            new_Batch_cat = new_Batch_cat.to(Batch_numerical.device)
        return torch.cat([Batch_numerical, new_Batch_cat], dim=1)