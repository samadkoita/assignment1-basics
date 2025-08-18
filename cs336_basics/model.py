from typing import Any, Mapping
import einops
import torch
from torch import nn
from jaxtyping import Float
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None =None, dtype: torch.dtype | None =None):
        super().__init__()
        self.w: nn.Parameter = nn.Parameter(data=torch.empty(out_features, in_features, dtype=dtype, device=device))
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]):
        return einops.einsum(
            x, self.w, "... d_in, d_out d_in -> ... d_out"
        )


if __name__ == "__main__":
    model = Linear(3, 4)
    sd = model.state_dict()
    print(sd)
    state_dict = {
        "w": torch.zeros((4, 3))
    }
    model.load_state_dict(sd)
    res = torch.ones(5, 3)
    print(model(res))
    model.load_state_dict(state_dict)
    print(model(res))
    print("HI")