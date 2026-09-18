"""Differentiable world-space Gaussian orientation updates."""

import math

import torch
from torch import Tensor


def rotate_gaussian_quaternions(base: Tensor, rotation_vectors: Tensor) -> Tensor:
    """Left-compose [G,3] world rotation vectors with static [G,4] wxyz quaternions."""
    angles = torch.linalg.vector_norm(rotation_vectors, dim=-1, keepdim=True)
    scalar = torch.cos(angles / 2)
    vector = 0.5 * torch.sinc(angles / (2 * math.pi)) * rotation_vectors
    real, imaginary = base[:, :1], base[:, 1:]
    rotated = torch.cat((scalar * real - (vector * imaginary).sum(dim=-1, keepdim=True),
                         scalar * imaginary + real * vector
                         + torch.linalg.cross(vector, imaginary, dim=-1)), dim=-1)
    rotated = torch.nn.functional.normalize(rotated, dim=-1)
    # Preserve the exact static value at zero while retaining the rotation derivative.
    unchanged = base.detach() + (rotated - rotated.detach())
    return torch.where(angles == 0, unchanged, rotated)
