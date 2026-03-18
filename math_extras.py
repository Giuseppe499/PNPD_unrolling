import torch
from torch.fft import rfft2, irfft2


def least_squares_data_fidelity(x, b, A_fun=torch.nn.Identity()):
    return torch.sum((A_fun(x) - b) ** 2, dim=(-2, -1)) / 2.0


def grad_least_squares_blur(x, b_rfft2, psf_rfft2, psf_rfft2_conj):
    xFFT = rfft2(x)
    return irfft2(psf_rfft2_conj * (psf_rfft2 * xFFT - b_rfft2))


def total_variation(x, eps=0.0):
    dx = torch.zeros_like(x)
    dx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    dy = torch.zeros_like(x)
    dy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.sum(torch.sqrt(dx**2 + dy**2 + eps), dim=(-2, -1))


def LS_TV_loss(x, b, lambda_tv, A_fun=torch.nn.Identity(), eps=0.0):
    fidelity = least_squares_data_fidelity(x, b, A_fun)
    tv = total_variation(x, eps)
    return fidelity + lambda_tv * tv


def P_inv(x: torch.Tensor, nu: float, psfAbsSq: torch.Tensor) -> torch.Tensor:
    return irfft2(rfft2(x) / ((1 - nu) * psfAbsSq + nu))


def grad2D(m: torch.tensor):
    dx = torch.roll(m, -1, dims=-2) - m
    dy = torch.roll(m, -1, dims=-1) - m
    # Comment for periodic boundary conditions
    dx[..., -1, :] = 0
    dy[..., :, -1] = 0
    return torch.stack((dx, dy))


def div2D(dxdy: torch.tensor):
    dx = dxdy[0, ...]
    dy = dxdy[1, ...]
    fx = torch.roll(dx, 1, dims=-2) - dx
    fy = torch.roll(dy, 1, dims=-1) - dy
    fx[..., 0, :] = -dx[..., 0, :]
    fx[..., -1, :] = dx[..., -2, :]
    fy[..., :, 0] = -dy[..., :, 0]
    fy[..., :, -1] = dy[..., :, -2]
    return fx + fy


def prox_h_star_TV(lam: float, dxdy: torch.tensor):
    dx = dxdy[0, ...]
    dy = dxdy[1, ...]
    factor = torch.sqrt(
        dx * dx + dy * dy + 1e-8 * lam
    )  # FIXME 1e-8 to avoid nan gradient. I should find a better way to fix this.
    factor = torch.clamp(factor / lam, min=1)
    factor = torch.stack((factor, factor))
    return dxdy / factor


def half_split(n):
    half = n // 2
    return half, n - half


def gaussian_psf(sigma, size):
    half_width_l, half_width_r = half_split(size[0])
    half_height_l, half_height_r = half_split(size[1])
    x = torch.arange(-half_width_l, half_width_r, dtype=torch.float32)
    y = torch.arange(-half_height_l, half_height_r, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    psf = psf.roll(-half_width_l, dims=0)
    psf = psf.roll(-half_height_l, dims=1)
    return psf


def blur_image_rfft(image_rfft2, psf_rfft2):
    return image_rfft2 * psf_rfft2


def blur_image(image, psf_rfft2):
    return irfft2(blur_image_rfft(rfft2(image), psf_rfft2))
