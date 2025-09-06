import math
import torch
from typing import Optional, Callable
from termcolor import colored

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t')
                grad = p.grad.data
                p.data = p.data - grad*lr*((t+1)**(-0.5))
                state['t'] = t+1
        return loss
        
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.999), eps: float=1e-8):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)
                m = (1 - beta1)*grad + beta1*m
                v = (1 - beta2)*(grad**2) + beta2*v
                
                lr_eff = lr * ((1 - beta2**t)**0.5) / (1 - beta1**t)
                p.data = p.data.mul_(1 - lr*weight_decay)
                p.data = p.data - (lr_eff*m/ (v**0.5 + eps))
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int):
    if t < t_w:
        return alpha_max*t/t_w
    if t > t_c:
        return alpha_min
    return alpha_min + (alpha_max - alpha_min)*(1 + math.cos(math.pi*(t - t_w)/(t_c - t_w)))/2
                

if __name__ == "__main__":
    torch.optim.AdamW
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(4, 3, bias=False),
    #      torch.nn.Linear(3, 3, bias=False),
    # )
         
    # optimiser = SGD(model.parameters(), lr=1e-3)
    # criterion = torch.nn.CrossEntropyLoss()
    # for _ in range(10):
    #     model.train()
    #     optimiser.zero_grad()
    #     x = torch.rand(size=(5,4))
    #     y_hat = torch.randint(0, 2, size=(5,))
    #     y = model(x)
    #     loss = criterion(y, y_hat)
    #     loss.backward()
    #     optimiser.step()

    for lr in [1, 1e1, 1e2, 1e3]:
        print(colored(f"\nLearning rate: {lr}", "red"))
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = AdamW([weights], lr=lr)
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            
            loss.backward() # Run backward pass, which computes gradients.
            grad_norm = torch.norm(weights.grad).item()
            # Log gradient and value for the same few random weights each time to monitor trends
            if 'sampled_indices' not in locals():
                num_samples = min(2, weights.numel())  # Sample up to 5 weights
                sampled_indices = torch.randperm(weights.numel())[:num_samples]
            
            flat_weights = weights.view(-1)
            flat_grads = weights.grad.view(-1)
            
            # for i, idx in enumerate(sampled_indices):
            #     weight_val = flat_weights[idx].item()
            #     grad_val = flat_grads[idx].item()
            #     print(f"  Weight[{idx}]: value = {weight_val:.6f}, grad = {grad_val:.6f}")
            opt.step() # Run optimizer step.
            
            print(f"Step {t}: Loss = {loss.item():.6f}, Grad norm = {grad_norm:.6f}")
