import torch
from math_extras import (
    grad2D,
    div2D,
    least_squares_data_fidelity,
    total_variation,
    LS_TV_loss,
)


def test_grad2D_and_div2D_transpose():
    w = 3
    h = 4
    canonical_basis_X = torch.zeros((w, h, w, h))

    for i in range(w):
        for j in range(h):
            canonical_basis_X[i, j, i, j] = 1

    A = grad2D(canonical_basis_X)
    assert A.shape == (2, w, h, w, h), "A has incorrect shape"

    canonical_basis_Y = torch.zeros((2, 2, w, h, w, h))
    for i in range(w):
        for j in range(h):
            for k in range(2):
                canonical_basis_Y[k, k, i, j, i, j] = 1

    A_T = div2D(canonical_basis_Y)
    A_T_T = A_T.permute(0, 3, 4, 1, 2)
    assert A_T.shape == (2, w, h, w, h), "A_T has incorrect shape"
    assert torch.allclose(A, A_T_T), "A and A_T_T are not equal"


def test_least_squares_data_fidelity_basic():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = least_squares_data_fidelity(x, b)
    assert torch.allclose(result, torch.tensor(0.0))


def test_least_squares_data_fidelity_nonzero():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    result = least_squares_data_fidelity(x, b)
    expected = torch.sum(x**2) / 2.0
    assert torch.allclose(result, expected)


def test_total_variation_constant():
    x = torch.ones((1, 4, 4))
    result = total_variation(x, eps=0.0)
    assert torch.allclose(result, torch.tensor(0.0))


def test_total_variation_constant_with_eps():
    x = torch.ones((1, 4, 4))
    tol = 1e-2
    result = total_variation(x, eps=tol)
    print(result)
    assert torch.allclose(result, torch.tensor(tol**0.5 * x.numel()))


def test_total_variation_single_jump():
    x = torch.zeros((1, 4, 4))
    x[..., 2, 2] = 1.0
    result = total_variation(x, eps=0.0)
    print(result)
    assert torch.allclose(result, 2 + torch.sqrt(torch.tensor(2.0)))


def test_LS_TV_loss_sum():
    x = torch.ones((1, 4, 4))
    b = torch.ones((1, 4, 4))
    lambda_tv = 2.0
    fidelity = least_squares_data_fidelity(x, b)
    tv = total_variation(x)
    loss = LS_TV_loss(x, b, lambda_tv)
    assert torch.allclose(loss, fidelity + lambda_tv * tv)
