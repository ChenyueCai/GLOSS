# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import scipy as sp
import torch
import igl
from collections import namedtuple
from gloss.utils.misc import to_numpy, to_tensor


def normalize(tensor:torch.Tensor) -> torch.Tensor:
    """
    

    Args:
        tensor (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    range = torch.amax(tensor) - torch.amin(tensor)
    eps = 0.000001
    try:
        assert range > eps 
        return (tensor - torch.amin(tensor)) / (torch.amax(tensor) - torch.amin(tensor))
    except AssertionError:
        print("range is near zero, cannot normalize")
        return tensor


# Local Features
def curvature(v:torch.Tensor, f:torch.Tensor):
    """generate a named tuple class for curvature related values, curvature values are smoothed

    Args:
        v (torch.Tensor): _description_
        f (torch.Tensor): _description_

    Returns:
        _type_: NamedTuple that extracts curvature properties
    """
    v, f = to_numpy(v), to_numpy(f)
    p1, p2, pv1, pv2 = igl.principal_curvature(v, f)
    g = igl.gaussian_curvature(v, f)
    pv1, pv2 = to_tensor(pv1), to_tensor(pv2)
    Curvature = namedtuple('Curvature', ['p1', 'p2', 'pv1', 'pv2', 'g', 'm'])
    c = Curvature(p1=to_tensor(p1), p2=to_tensor(p2), pv1=to_tensor(pv1), pv2=to_tensor(pv2), 
                  g=to_tensor(g), m=to_tensor(p1 + p2)/2)
    return c

# Deals with Duplicate Vertices
# Multiple vertices appears at the same position
# face topology watertight when making all the vertices the same one

def merge_mapping(v:torch.Tensor) -> torch.Tensor:
    vertices, mapping, counts = torch.unique(v, return_inverse=True, return_counts=True, dim=0) 
    # mapping : OG -> New no duplicate indices # inverse: no duplicate OG
    return vertices, mapping, counts


def merge_duplicate(f:torch.Tensor, mapping:torch.Tensor) -> tuple[torch.Tensor]:
    faces = mapping[f]
    return faces


# A = [-Y, X, 0]
# B = [-X*Z, -Y*Z, X*X+Y*Y]
# [ X,Y,Z]·[-Y,X,0] = -X*Y+Y*X = 0
# [ X,Y,Z]·[-X*Z,-Y*Z,X*X+Y*Y] = -X*X*Z-Y*Y*Z+Z*(X*X+Y*Y) = 0
# [-Y,X,0]·[-X*Z,-Y*Z,X*X+Y*Y] = Y*X*Z+X*Y*Z = 0
# This is called the nullspace of your vector.
# If X=0 and Y=0 then A=[1,0,0], B=[0,1,0]
def get_nullspace(vectors):
    """
    Gets the null-space (2 orthonormal vectors) of each vector in the batch.

    Args:
        vectors: torch vector of directions N x 3

    Returns:
        null spaces of all vectors of shape N x 2 x 3 (i.e. 2 remaining vectors for orthonormal basis)

    """
    eps = 1e-6

    vectors = vectors / torch.linalg.norm(vectors, dim=-1).unsqueeze(-1)
    nullspace = torch.zeros((vectors.shape[0], 2, 3), dtype=vectors.dtype, device=vectors.device)
    nullspace[:, 0, 0] = vectors[:, 1] * -1.0
    nullspace[:, 0, 1] = vectors[:, 0]
    nullspace[:, 1, 0] = vectors[:, 0] * vectors[:, 2] * -1.0
    nullspace[:, 1, 1] = vectors[:, 1] * vectors[:, 2] * -1.0
    nullspace[:, 1, 2] = vectors[:, 0] * vectors[:, 0] + vectors[:, 1] * vectors[:, 1]
    degenerate = torch.logical_and(torch.abs(vectors[:, 0]) < eps, torch.abs(vectors[:, 2]) < eps)
    nullspace[degenerate, ...] = 0
    nullspace[degenerate, 0, 0] = 1.0
    nullspace[degenerate, 1, 1] = 1.0

    nullspace = nullspace / torch.linalg.norm(nullspace, dim=-1).unsqueeze(-1)
    return nullspace
