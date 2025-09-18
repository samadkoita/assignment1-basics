import typing
import torch
import torch.nn as nn
import os

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    result = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration
    )
    
    torch.save(result, out)
    return 

def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer):
    result = torch.load(src)
    model.load_state_dict(result["model"])
    optimizer.load_state_dict(result["optimizer"])
    return result["iteration"]
