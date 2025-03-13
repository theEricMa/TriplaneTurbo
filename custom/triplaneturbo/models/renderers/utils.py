import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.typing import *
from collections import defaultdict


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='sphere', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)



    if type == 'sphere_general':

        # Normalize the direction of the rays
        rays_d = F.normalize(rays_d, dim=-1)

        # Calculate the dot product of the direction of the ray and the origin of the ray
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)

        # Calculate the discriminant (b^2 - 4ac)
        discriminant = b**2 - 4.0 * (radius**2 - bound**2)

        # If the discriminant is less than 0, the ray does not intersect the sphere
        mask = discriminant >= 0
        discriminant = torch.where(mask, discriminant, torch.zeros_like(discriminant))

        # Calculate the near and far intersection distances
        near = 0.5 * (-b - torch.sqrt(discriminant))
        far = 0.5 * (-b + torch.sqrt(discriminant))

        # Ensure that 'near' is not closer than 'min_near'
        near = torch.max(near, min_near * torch.ones_like(near))

    elif type == 'sphere':

        near = radius - bound
        far = radius + bound

    return near, far




def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    # a little modification of chunk_batch from th eresstudio/utils/ops.py
    # by split the input in the second dimension instead of the first dimension
    B = None
    for arg in list(args) + list(kwargs.values()):
        # the following line is 1/2 of the modification
        if isinstance(arg, torch.Tensor) and arg.ndim >= 2:
            B = arg.shape[1]
            break
        #############################
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    for i in range(0, max(1, B), chunk_size):
        # the following line is 2/2 of the modification
        out_chunk = func(
            *[
                arg[:, i : i + chunk_size] if isinstance(arg, torch.Tensor) and arg.ndim >= 2 and arg.shape[1] == B else arg
                for arg in args
            ],
            **{
                k: arg[:, i : i + chunk_size] if isinstance(arg, torch.Tensor) and arg.ndim >= 2 and arg.shape[1] == B else arg
                for k, arg in kwargs.items()
            },
        )
        #############################
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0) # TODO: check if this is correct
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged

@torch.no_grad()
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples