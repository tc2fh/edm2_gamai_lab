# Shared target/conditioning encodings for the ViViT-conditioned EDM2 pipeline.
#
# Implements contract C4 in docs/vivit_pipeline_contracts.md exactly. Pure
# numpy in / numpy out; `decode_target` also works unmodified on torch
# tensors since it is just a threshold comparison.
#
# Consumers: prepare_data_vivit_pairs.py (target/cond encoding at dataset-
# build or load time) and the PyTorch PairDataset / training loop (which
# import these functions directly rather than reimplementing the math).

import numpy as np
from scipy.ndimage import distance_transform_edt

#----------------------------------------------------------------------------
# Target encodings.

def encode_target(mask_uint8, encoding, spacing=(1, 1, 2)):
    """Encode a {0,1} tumor mask into the diffusion target representation.

    Args:
        mask_uint8: array-like, any shape, values in {0,1} (uint8 or bool).
        encoding:   'binary' or 'sdf'.
        spacing:    voxel spacing in mm, used only by 'sdf'.

    Returns:
        float32 array of the same shape, range [-1, 1].
    """
    mask = np.asarray(mask_uint8).astype(bool)

    if encoding == 'binary':
        return mask.astype(np.float32) * 2.0 - 1.0

    if encoding == 'sdf':
        # Degenerate cases: distance_transform_edt is only well-defined when
        # both the foreground and background are non-empty (with no zero
        # elements to measure to, scipy's output is implementation-defined,
        # not a documented +-8mm saturation), so these are handled explicitly.
        if not mask.any():
            # Nothing inside: every voxel is (at least) 8mm outside -> +1.
            return np.ones(mask.shape, dtype=np.float32)
        if mask.all():
            # Nothing outside: every voxel is (at least) 8mm inside -> -1.
            return -np.ones(mask.shape, dtype=np.float32)

        outside_dist = distance_transform_edt(~mask, sampling=spacing)
        inside_dist = distance_transform_edt(mask, sampling=spacing)
        sdf = outside_dist - inside_dist  # negative inside, positive outside
        sdf = np.clip(sdf, -8.0, 8.0) / 8.0
        return sdf.astype(np.float32)

    raise ValueError(f'unknown target encoding: {encoding!r}')


def is_degenerate_target(mask_uint8):
    """True if a mask has no foreground or no background voxels.

    Callers (the pair builder) use this to warn and count targets whose SDF
    encoding takes the special-cased +-1-everywhere branch of encode_target.
    """
    mask = np.asarray(mask_uint8).astype(bool)
    return bool((not mask.any()) or mask.all())


def decode_target(x, encoding):
    """Inverse of encode_target's sign convention. Returns a bool mask.

    Works unchanged on numpy arrays or torch tensors (both support `>` and
    `<=` against a python scalar and return a same-shaped bool result).
    """
    if encoding == 'binary':
        return x > 0
    if encoding == 'sdf':
        return x <= 0
    raise ValueError(f'unknown target encoding: {encoding!r}')

#----------------------------------------------------------------------------
# Conditioning-image encoding.

def encode_cond_image(cond_mask, cond_image, mode):
    """Build the extra input-concatenation channels for the UNet.

    Args:
        cond_mask:  {0,1} array-like (uint8/bool), shape (D,H,W), or None.
        cond_image: z-scored float array-like, shape (D,H,W), or None.
        mode:       'none', 'mask', or 'mask+image'.

    Returns:
        float32 array (C, D, H, W) with C in {0, 1, 2} per contract C4.
    """
    if mode == 'none':
        ref = cond_mask if cond_mask is not None else cond_image
        if ref is None:
            raise ValueError("mode='none' still needs cond_mask or cond_image to infer shape")
        shape = tuple(np.asarray(ref).shape)
        return np.zeros((0,) + shape, dtype=np.float32)

    if cond_mask is None:
        raise ValueError(f"mode={mode!r} requires cond_mask")
    mask_ch = np.asarray(cond_mask).astype(np.float32) * 2.0 - 1.0

    if mode == 'mask':
        return mask_ch[None, ...]

    if mode == 'mask+image':
        if cond_image is None:
            raise ValueError("mode='mask+image' requires cond_image")
        img_ch = np.clip(np.asarray(cond_image).astype(np.float32), -5.0, 5.0)
        return np.stack([mask_ch, img_ch], axis=0)

    raise ValueError(f'unknown cond_image mode: {mode!r}')

#----------------------------------------------------------------------------
# Time-gap feature.

def time_gap_feature(days):
    """log1p(days / 360), used both for delta_days and per-history scan age.

    Accepts a python scalar, numpy array, or torch tensor; returns the same
    kind of object (relies on log1p/asarray-free ops so torch tensors pass
    through torch.log1p via __array_function__... to keep this dependency-
    free for numpy, plain numpy is used here and callers pass numpy/scalars;
    torch callers should use torch.log1p(days / 360) directly, which is the
    same formula).
    """
    return np.log1p(np.asarray(days, dtype=np.float64) / 360.0)

#----------------------------------------------------------------------------
