import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        # dim_in, dim_k, dim_v are the input, key, and value dimensions respectively
        super().__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be divisible by num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads

        # 把输入的x (dim_in) 映射到 q, k, v (dim_k, dim_v) dim_k和dim_v是所有头总维度
        self.linear_q = nn.Linear(dim_in, dim_k)
        self.linear_k = nn.Linear(dim_in, dim_k)
        self.linear_v = nn.Linear(dim_in, dim_v)
        
        self._norm_factor = 1.0 / (dim_k // num_heads) ** 0.5  # 缩放因子，防止点积过大

        self.linear_out = nn.Linear(dim_v, dim_in)  # 最后的线性变换
    
    def forward(self, x, mask=None):
        bs, seq_len, dim_in = x.shape
        num_h = self.num_heads
        # 计算单头的 d_k 和 d_v
        d_k = self.dim_k // num_h
        d_v = self.dim_v // num_h

        q = self.linear_q(x).reshape(bs, seq_len, num_h, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        k = self.linear_k(x).reshape(bs, seq_len, num_h, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        v = self.linear_v(x).reshape(bs, seq_len, num_h, d_v).transpose(1, 2)  # (bs, num_heads, seq_len, d_v)

        attn_score = torch.matmul(q, k.transpose(2, 3)) * self._norm_factor  # (bs, num_heads, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, seq_len)
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_weight = F.softmax(attn_score, dim=-1)  # (bs, num_heads, seq_len, seq_len)

        output = torch.matmul(attn_weight, v)  # (bs, num_heads, seq_len, d_v)
        output = output.transpose(1, 2).contiguous().reshape(bs, seq_len, self.dim_v)  # (bs, seq_len, dim_v)
        output = self.linear_out(output)  # (bs, seq_len, dim_in)
        return output, attn_weight
        

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.): # qkv_bias 设为True更常见
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x 的形状: (B, L, Dim)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x