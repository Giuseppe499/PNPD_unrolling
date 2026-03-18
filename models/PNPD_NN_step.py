import torch
from torch import nn
from typing import Callable
from solvers import PNPD_step, PNPD_parameters, PNPD_functions

class PNPD_NN_step(nn.Module):
    def __init__(self,
            alpha_Net: nn.Module,
            beta_Net: nn.Module,
            nu_Net: nn.Module,
            lam_Net: nn.Module,
            params: PNPD_parameters,
            funs: PNPD_functions,
            P_inv: Callable[[float, torch.Tensor], torch.Tensor],
            prox_h_star: Callable[[float, torch.Tensor, float], torch.Tensor],
            grad_f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ):
        super(PNPD_NN_step, self).__init__()
        self.alpha_Net = alpha_Net
        self.beta_Net = beta_Net
        self.nu_Net = nu_Net
        self.lam_Net = lam_Net

        self.params = params
        self.funs = funs
        self.P_inv = P_inv
        self.prox_h_star = prox_h_star
        self.grad_f = grad_f

    def forward(
            self,
            observed: torch.Tensor,
            u: torch.Tensor,
            v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        W_T_v = self.funs.W_T(v)
        # net_input =torch.cat((u, observed, W_T_v), dim=1)  # Stack u, observed and W_T_v along the channel dimension
        net_input =torch.cat((u, observed), dim=1) # Stack u and observed along the channel dimension
        # net_input = u

        # Update stepsizes
        self.params.alpha = self.alpha_Net(net_input) + .5
        # self.params.alpha = torch.ones(u.shape[0], 1).to(u.device) * .98  # Use a constant value for alpha
        self.last_alpha = self.params.alpha
        self.params.alpha = self.params.alpha[:, None, None]  # Add dimensions to match PNPD_step input
        self.params.beta = self.beta_Net(net_input) + .5/8

        # self.params.beta = torch.ones(u.shape[0], 1).to(u.device) / 8. * .98  # Use a constant value for beta
        self.last_beta = self.params.beta
        self.params.beta = self.params.beta[:, None, None]

        # Update regularization parameter
        lam_regularization = self.lam_Net(net_input) + 1e-4
        self.last_lam = lam_regularization
        lam_regularization = lam_regularization[:, None, None]  # Add dimensions to match lam_prox input
        self.funs.prox_h_star = lambda x, lam_prox: self.prox_h_star(lam_regularization, x, lam_prox)

        # Update preconditioner parameter
        nu = self.nu_Net(net_input)
        self.last_nu = nu
        nu = nu[:, None, None]  # Add dimensions to match P_inv input
        self.funs.P_inv = lambda x: self.P_inv(nu, x)

        # Set the functions for the gradient of f
        observed_rfft = torch.fft.rfft2(observed)
        self.funs.grad_f = lambda u: self.grad_f(u, observed_rfft)

        return PNPD_step(u, v, self.params, self.funs)