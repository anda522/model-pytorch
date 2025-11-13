import math
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
    
    def forward(self, q, k, v, mask=None):
        bs, seq_len, dim_in = q.shape
        num_h = self.num_heads
        # 计算单头的 d_k 和 d_v
        d_k = self.dim_k // num_h
        d_v = self.dim_v // num_h

        q = self.linear_q(q).reshape(bs, seq_len, num_h, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        k = self.linear_k(k).reshape(bs, seq_len, num_h, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        v = self.linear_v(v).reshape(bs, seq_len, num_h, d_v).transpose(1, 2)  # (bs, num_heads, seq_len, d_v)

        attn_score = torch.matmul(q, k.transpose(2, 3)) * self._norm_factor  # (bs, num_heads, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, seq_len)
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_weight = F.softmax(attn_score, dim=-1)  # (bs, num_heads, seq_len, seq_len)

        output = torch.matmul(attn_weight, v)  # (bs, num_heads, seq_len, d_v)
        output = output.transpose(1, 2).contiguous().reshape(bs, seq_len, self.dim_v)  # (bs, seq_len, dim_v)
        output = self.linear_out(output)  # (bs, seq_len, dim_in)
        return output, attn_weight
        
class MultiQueryAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super().__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be divisible by num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads

        d_k = dim_k // num_heads
        d_v = dim_v // num_heads

        # Q 仍然按多头映射；K/V 只映射出单头维度（共享给所有 Q 头）
        self.linear_q = nn.Linear(dim_in, dim_k)
        self.linear_k = nn.Linear(dim_in, d_k) # 1个kv头
        self.linear_v = nn.Linear(dim_in, d_v)

        self._norm_factor = 1.0 / d_k ** 0.5

        self.linear_out = nn.Linear(d_v, dim_in)

    def forward(self, q, k, v, mask=None):
        bs, seq_len, _ = q.shape
        h = self.num_heads
        d_k = self.dim_k // h
        d_v = self.dim_v // h

        q = self.linear_q(q).reshape(bs, seq_len, h, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        k = self.linear_k(k).reshape(bs, seq_len, 1, d_k).transpose(1, 2)  # (bs, num_heads, seq_len, d_k)
        v = self.linear_v(v).reshape(bs, seq_len, 1, d_v).transpose(1, 2)  # (bs, num_heads, seq_len, d_v)

        # 将单个 KV 头广播到 h 个 Q 头
        k = k.expand(bs, h, seq_len, d_k)
        v = v.expand(bs, h, seq_len, d_v)

        attn_score = torch.matmul(q, k.transpose(2, 3)) * self._norm_factor  # (bs, num_heads, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, seq_len)
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_weight = F.softmax(attn_score, dim=-1)  # (bs, num_heads, seq_len, seq_len)

        output = torch.matmul(attn_weight, v)  # (bs, num_heads, seq_len, d_v)
        output = output.transpose(1, 2).contiguous().reshape(bs, seq_len, self.dim_v)  # (bs, seq_len, dim_v)
        output = self.linear_out(output)  # (bs, seq_len, dim_in)
        return output, attn_weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8, kv_heads=2):
        super().__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0
        assert num_heads % kv_heads == 0, "num_heads 必须能被 kv_heads 整除"
        self.dim_in, self.dim_k, self.dim_v = dim_in, dim_k, dim_v
        self.num_heads = num_heads
        self.kv_heads = kv_heads

        d_k = dim_k // num_heads
        d_v = dim_v // num_heads

        # Q: 仍按 num_heads 输出；K/V: 仅输出 kv_heads 个头
        self.linear_q = nn.Linear(dim_in, dim_k)
        self.linear_k = nn.Linear(dim_in, d_k * kv_heads)
        self.linear_v = nn.Linear(dim_in, d_v * kv_heads)

        self.scale = d_k ** -0.5
        self.linear_out = nn.Linear(dim_v, dim_in)

    def forward(self, q, k, v, mask=None):
        bs, T, _ = q.shape
        h, kh = self.num_heads, self.kv_heads
        d_k = self.dim_k // h
        d_v = self.dim_v // h
        group = h // kh  # 每个 KV 头服务的 Q 头数

        # Q: (bs,h,T,d_k)
        q = self.linear_q(q).reshape(bs, T, h, d_k).transpose(1, 2)

        # K: (bs,kh,T,d_k)  -> 重复到 (bs,h,T,d_k)
        k = self.linear_k(k).reshape(bs, T, kh, d_k).transpose(1, 2)
        k = k.repeat_interleave(group, dim=1)

        # V: (bs,kh,T,d_v)  -> 重复到 (bs,h,T,d_v)
        v = self.linear_v(v).reshape(bs, T, kh, d_v).transpose(1, 2)
        v = v.repeat_interleave(group, dim=1)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (bs,h,T,T)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weight = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weight, v) # (bs,h,T,d_v)
        out = out.transpose(1, 2).contiguous().reshape(bs, T, self.dim_v)  # (bs,T,dim_v)
        out = self.linear_out(out) # (bs,T,dim_in)
        return out, attn_weight

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

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        # pe: (max_len, dim)
        pe = torch.zeros(max_len, dim)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # 变成 (1, L, D)，方便和 (B, L, D) 相加
        pe = pe.unsqueeze(0)  # (1, max_len, dim)

        # register_buffer 表示这是模型的一部分，但不是参数（不参与反向更新）
        self.register_buffer('pe', pe)  # pe: (1, max_len, dim)

    def forward(self, x):
        """
        x: (B, L, D)
        返回: x + 位置编码
        """
        B, L, D = x.size()
        # 只取前 L 个位置，并广播到 batch 维度
        x = x + self.pe[:, :L, :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., max_len=5000):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(dim, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B, L, D)
        x = self.pos_encoding(x)  # 就在这里加位置编码，一次就够

        for blk in self.layers:
            x = blk(x)
        return x
