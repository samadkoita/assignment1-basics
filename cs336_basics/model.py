from typing import Any, Mapping
import math
import einops
import torch
from torch import nn
from typing import Optional
from jaxtyping import Float, Int
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None =None, dtype: torch.dtype | None =None):
        super().__init__()
        init_weights = torch.empty(out_features, in_features, dtype=dtype, device=device)
        self.weight: nn.Parameter = nn.Parameter(data=init_weights)
        std = math.sqrt(2/(in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[torch.Tensor, "... d_in"]):
        return einops.einsum(
            x, self.weight, "... d_in, d_out d_in -> ... d_out"
        )

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(data=torch.empty(size=(num_embeddings, embedding_dim)))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


    def forward(self, x: Float[torch.Tensor, "... seqlen"]) -> Int[torch.Tensor, "... seqlen embedding_dim"]:
        return self.weight[x]

def silu(x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: Float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model: int = d_model
        self.eps = torch.tensor(eps, dtype=dtype)
        self.weight = nn.Parameter(
            data=torch.ones(size=(d_model,), dtype=dtype, device=device)
        )

    def forward(self, x: Float[torch.Tensor, "... d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rmse = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)
        x = x * self.weight / rmse
        return x.to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device=device
        positions = torch.arange(max_seq_len, device=device)
        thetas = theta**(-2 * torch.arange(d_k//2, device=device) / d_k)
        angles_seq_dk = einops.einsum(positions, thetas, "seqlen, d_k -> seqlen d_k")
        cos_angles = torch.cos(angles_seq_dk)
        sin_angles = torch.sin(angles_seq_dk)
        self.register_buffer("sin_embedding", sin_angles, persistent=False)
        self.register_buffer("cos_embedding", cos_angles, persistent=False)

    def forward(self, x: Float[torch.Tensor, "... seqlen d_k"], token_positions: Int[torch.Tensor, "... seqlen"]) -> Float[torch.Tensor, "... seqlen d_k"]:
        cos_vals_seq_dk = self.cos_embedding[token_positions]
        sin_vals_seq_dk = self.sin_embedding[token_positions]
        out = torch.empty(size=x.shape)
        even_x = x[..., ::2]
        odd_x = x[..., 1::2]
        out[..., ::2] = cos_vals_seq_dk*even_x - sin_vals_seq_dk*odd_x
        out[..., 1::2] = sin_vals_seq_dk*even_x + cos_vals_seq_dk*odd_x
        return out

def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    in_features -= torch.max(in_features, dim=dim, keepdim=True)[0]
    in_features = torch.exp(in_features)
    in_features /= torch.sum(in_features, dim=dim, keepdim=True)
    return in_features 

def scaled_dot_product_attention(q: Float[torch.Tensor, "... seq1 d_k"],
                                 k: Float[torch.Tensor, "... seq2 d_k"],
                                 v: Float[torch.Tensor, "... seq2 d_v"],
                                 mask: Optional[Float[torch.Tensor, "... seq1 seq2"]] = None):
    d_k = q.shape[-1]
    qk = einops.einsum(q, k, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2") * (d_k**(-0.5))
    if mask is not None:
        qk.masked_fill_(mask == 0, -torch.inf)
    qk = softmax(qk, dim=-1)
    final = einops.einsum(qk, v, "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v")
    return final

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
    
    def forward(self, x: Float[torch.Tensor, "... seq d_model"], rope: Optional[RotaryPositionalEmbedding] = None, token_positions: Optional[Int[torch.Tensor, "... seq"]] = None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        to_mh = lambda t: einops.rearrange(t, "... seq (num_heads head_dim) -> ... num_heads seq head_dim", num_heads=self.num_heads)
        q, k, v = to_mh(q), to_mh(k), to_mh(v)
        seqlen = x.shape[-2]
        if rope:
            if token_positions is None:
                raise ValueError("token_positions is required when using RoPE")
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        mask = torch.tril(torch.ones(size=(seqlen, seqlen)))
        res = scaled_dot_product_attention(q, k, v, mask)
        res = einops.rearrange(res, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        return self.output_proj(res)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: Float[torch.Tensor, "... seq d_model"], rope: Optional[RotaryPositionalEmbedding] = None, token_positions: Optional[Int[torch.Tensor, "... seq"]] = None):
        x += self.attn(self.ln1(x), rope, token_positions)
        x += self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size :int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        head_dim = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=head_dim, max_seq_len=context_length)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        pass

    def forward(self, x: Int[torch.Tensor, "... seq"], token_positions: Int[torch.Tensor, "... seq"]):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, self.rope, token_positions)
        x = self.lm_head(self.ln_final(x))
        return x

if __name__ == "__main__":
    model = RotaryPositionalEmbedding(
        theta=10, d_k=4, max_seq_len=10
    )
    model(torch.rand((5, 3, 4)), [0, 1, 2])
