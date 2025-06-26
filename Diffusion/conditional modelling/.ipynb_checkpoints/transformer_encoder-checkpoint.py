import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model = 128
        self.num_heads = num_heads
        assert d_model % num_heads == 0,
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, context):
        """
        query: (batch, num_target, d_model) (noisy X tokens)
        context: (batch, num_context, d_model) -> from encoder (Y,A)

        output: attended representation for each query token
        """
        B, num_query, d_model = query.size()
        _, num_context, _ = context.size()
        d_k = d_model // num_heads

        # Project inputs to Q,K,V
        Q = self.q_proj(query).view(batch_size, num_unknown, num_heads, d_k).transpose(1,2)  
        K = self.k_proj(context).view(batch_size, num_known, num_heads, d_k).transpose(1,2) 
        V = self.v_proj(context).view(batch_size, num_known, num_heads, d_k).transpose(1,2)

        # Attention weights: Q @ K.T
        attn_weights = torch.matmul(Q, K.transpose(-2,-1)) / (d_k**0.5)                    
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attend to values
        attended = torch.matmul(attn_weights, V)                     

        # Combine heads & output projection
        attended = attn_weights @ V                                        
        attended = attended.transpose(1,2).contiguous().view(batch_size, num_unknown, d_model)
        output = self.out_proj(attended)                                                
        return output
    
    


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: "TransformerEncoderLayer", num_layers: int,) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, src_1: Tensor, src_2: Tensor) -> Tensor:
        output_1 = src_1
        first_layer = self.layers[0]

        batch_first = first_layer.self_attn.batch_first
        
        seq_len = self._get_seq_len(src_1, batch_first)

        for mod in self.layers:
            output_1 = mod(output_1, src_2)

        return output_1
    
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def _get_seq_len(self, src: Tensor, batch_first: bool) -> Optional[int]:

        if src.is_nested:
            return None
        else:
            src_size = src.size()
            if len(src_size) == 2:
                # unbatched: S, E
                return src_size[0]
            else:
                # batched: B, S, E if batch_first else S, B, E
                seq_len_pos = 1 if batch_first else 0
                return src_size[seq_len_pos]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=True, batch_first=False,
                                            **factory_kwargs)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, bias=True, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=True, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src_1: Tensor, src_2: Tensor) -> Tensor:
        src_1 = self.norm1(src_1 + self._sa_block(src_1, src_2))
        src_1 = self.norm2(src_1 + self._ff_block(src_1))

        return src_1


    # self-attention block
    def _sa_block(self, x: Tensor, a: Tensor) -> Tensor:
        x = self.self_attn(x, a, a, need_weights=False)[0]
        return self.dropout1(x) 

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)