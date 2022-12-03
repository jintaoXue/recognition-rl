import numpy as np
import torch
from torch import Tensor
from typing import Optional,Tuple
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import torch.nn.functional as F
import math
import warnings



#################################################
# scaled_dot_product_attention###################
#################################################

def scaled_dot_product_attention_2(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scaled_term, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scaled_term = scaled_term
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k) # [B, n_head, T, T]
        attn = attn / self.scaled_term

        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e10)

        attn = self.softmax(attn) # [B, n_head, T, T] 这一步引入了非线性
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v) # [B, n_head, T, H//n_head] # Sum

        return output, attn

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


#################################################
# multi_head_attention###########################
#################################################

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, out_dim,num_heads, dropout=0.) -> None:
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, out_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    
        # compute in-projection
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        
        # reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, None , self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

if __name__ == '__main__':
    d_model = 8
    seq_len = 3
    batch_size = 6
    num_heads = 2
    # mask = None
    mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal = 0).unsqueeze(0)
    
    input = torch.rand(batch_size, seq_len, d_model)
    multi_attn = MultiheadAttention(num_heads = num_heads, d_model = d_model, dropout = 0.1)
    out = multi_attn(query = input, key = input, value = input, mask = mask)
    print(out.shape)