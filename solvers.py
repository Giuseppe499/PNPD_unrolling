import torch
from typing import Callable
from dataclasses import dataclass


@dataclass
class PNPD_parameters:
    alpha: float
    beta: float
    k_max: int = 1


@dataclass
class PNPD_functions:
    P_inv: Callable
    W: Callable
    W_T: Callable
    grad_f: Callable
    prox_h_star: Callable[[torch.Tensor, float], torch.Tensor]


def PNPD_step(
    u: torch.Tensor, v: torch.Tensor, params: PNPD_parameters, funs: PNPD_functions
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = params.alpha
    beta = params.beta
    k_max = params.k_max

    # Gradient descent step
    u = u - alpha * funs.P_inv(funs.grad_f(u))

    # Proximal operator approximation via nested iteration
    u_sum = torch.zeros_like(u)
    u = u - alpha * funs.W_T(v)
    for _ in range(k_max):
        v = funs.prox_h_star(v + beta / alpha * funs.W(u), beta / alpha)
        u = u - alpha * funs.W_T(v)
        u_sum += u
    u = u_sum / k_max

    return u, v
