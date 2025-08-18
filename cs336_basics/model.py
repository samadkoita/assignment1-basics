from typing import Any, Mapping
import math
import einops
import torch
from torch import nn
from jaxtyping import Float, Int
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None =None, dtype: torch.dtype | None =None):
        super().__init__()
        init_weights = torch.empty(out_features, in_features, dtype=dtype, device=device)
        self.w: nn.Parameter = nn.Parameter(data=init_weights)
        std = math.sqrt(2/(in_features + out_features))
        torch.nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]):
        return einops.einsum(
            x, self.w, "... d_in, d_out d_in -> ... d_out"
        )

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(data=torch.empty(size=(num_embeddings, embedding_dim)))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)


    def forward(self, x: Float[torch.Tensor, "... seqlen"]) -> Int[torch.Tensor, "... seqlen embedding_dim"]:
        return self.embeddings[x]

if __name__ == "__main__":
    model = Embedding(3, 4)

