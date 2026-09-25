# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming (history, target) pairs from a dataset built by
prepare_data_vivit_pairs.py (contract C3 in docs/vivit_pipeline_contracts.md),
with ViViT context tokens loaded from the token store (contract C2) at run
time. Returns the batch dict of contract C5."""

import os
import json
import numpy as np
import torch

from training.encodings import encode_target, encode_cond_image

#----------------------------------------------------------------------------
# Dataset of (history, target) pairs for the ViViT-conditioned EDM2 pipeline.

class PairDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                       # Root directory produced by prepare_data_vivit_pairs.py.
        tokens_dir      = None,     # C2 token store root. None = read from dataset.json.
        split           = 'train',  # 'train', 'val', or 'test'.
        target_encoding = 'binary', # 'binary' or 'sdf' (contract C4).
        cond_image      = 'mask',   # 'none', 'mask', or 'mask+image' (contract C4).
        max_history     = None,     # Pad/truncate history to this many scans. None = dataset.json's max_history.
        token_key       = 'tokens', # Which C2 token array to read ('tokens', 'tokens_l3', 'tokens_l6', 'tokens_l9').
        flip_axes       = False,    # Random per-axis flips, applied identically to image and cond_image.
        cache_tokens    = True,     # Cache loaded token arrays in process memory (they are read-only and shared).
        sigma_data      = None,     # Override sigma_data. None = read from dataset.json's train stats (default).
        use_context     = True,     # False = never read tokens; items omit 'context'/'context_mask'/'context_ages'.
        spacing         = None,     # Override voxel spacing (mm), e.g. for reconstruction from C7 metadata. None = read from dataset.json.
    ):
        if target_encoding not in ('binary', 'sdf'):
            raise ValueError(f'unknown target_encoding: {target_encoding!r}')
        if cond_image not in ('none', 'mask', 'mask+image'):
            raise ValueError(f'unknown cond_image mode: {cond_image!r}')

        manifest_path = os.path.join(path, 'dataset.json')
        if not os.path.isfile(manifest_path):
            raise IOError(f'no dataset.json found under {path!r}')
        with open(manifest_path) as f:
            manifest = json.load(f)
        if split not in manifest['splits']:
            raise IOError(f'split {split!r} not found in {manifest_path!r}')

        self._path = path
        self._split = split
        self._split_dir = os.path.join(path, split)
        self._target_encoding = target_encoding
        self._cond_image_mode = cond_image
        self._token_key = token_key
        self._flip_axes = bool(flip_axes) and split == 'train'
        self._cache_tokens = cache_tokens
        self._token_cache = {}
        self._use_context = bool(use_context)

        self._tokens_dir = tokens_dir if tokens_dir is not None else manifest['tokens_dir']
        self._samples = manifest['splits'][split]['samples']
        if len(self._samples) == 0:
            raise IOError(f'split {split!r} has no samples in {manifest_path!r}')
        self._resolution = tuple(int(s) for s in manifest['shape'])
        # Physical voxel spacing (mm), for SDF encoding (contract C4/C0).
        # C0 (2026-09-12): the re-extracted training-frame token store no
        # longer has isotropic-ish (1,1,2) spacing -- every consumer must
        # read it from the data, never hard-code it. `spacing` may be given
        # explicitly (e.g. reconstructing from a checkpoint's C7 metadata,
        # which already carries it), mirroring the sigma_data override below.
        if spacing is not None:
            self._spacing = tuple(float(s) for s in spacing)
        else:
            if 'spacing' not in manifest:
                raise IOError(f"dataset.json is missing 'spacing' in {manifest_path!r} (contract C0/C4)")
            self._spacing = tuple(float(s) for s in manifest['spacing'])

        manifest_max_history = manifest.get('max_history')
        self._max_history = int(max_history) if max_history is not None else int(manifest_max_history)

        if sigma_data is not None:
            # Explicit override: e.g. generation/evaluation reconstructing a
            # dataset from a checkpoint's C7 metadata, which already knows
            # net.sigma_data and does not need (and may not have) a
            # dataset.json stats block at all -- a test-only pair dir built
            # from another repo's manifest can legitimately lack train stats.
            self._sigma_data = float(sigma_data)
        else:
            # Default: sigma_data (and any other normalization stat) always
            # comes from the TRAIN split's stats, regardless of which split
            # this dataset serves (contract C3: dataset.json only ever has
            # train-split stats -- val and test must never define their own
            # normalization).
            if 'train' not in manifest.get('stats', {}):
                raise IOError(f"dataset.json is missing 'stats.train' in {manifest_path!r}")
            stats = manifest['stats']['train']
            rms_key = f'target_rms_{target_encoding}'
            if rms_key not in stats:
                raise IOError(f'{rms_key!r} missing from dataset.json train stats')
            self._sigma_data = float(stats[rms_key])

        self._cond_channels = {'none': 0, 'mask': 1, 'mask+image': 2}[cond_image]

    def __len__(self):
        return len(self._samples)

    def _load_tokens(self, scan_id):
        key = scan_id
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        fpath = os.path.join(self._tokens_dir, self._split, f'{scan_id}.npz')
        with np.load(fpath) as z:
            tok = np.asarray(z[self._token_key], dtype=np.float32)
        if self._cache_tokens:
            self._token_cache[key] = tok
        return tok

    def __getitem__(self, idx):
        entry = self._samples[idx]
        fpath = os.path.join(self._split_dir, f"{entry['idx']:08d}.npz")
        with np.load(fpath) as z:
            target_mask = z['target_mask']
            cond_mask = z['cond_mask']
            cond_image_arr = z['cond_image'] if 'cond_image' in z.files else None
            delta_days = float(np.asarray(z['delta_days']))
            if self._use_context:
                history_scan_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in z['history_scan_ids']]
                history_ages = np.asarray(z['history_ages_days'], dtype=np.float32)

        image = encode_target(target_mask, self._target_encoding, spacing=self._spacing)[np.newaxis, ...] # (1,D,H,W)
        cond_image = encode_cond_image(cond_mask, cond_image_arr, self._cond_image_mode) # (C,D,H,W)

        if self._flip_axes:
            flip_dims = [1 + ax for ax in range(3) if np.random.rand() < 0.5] # +1: skip channel dim
            if flip_dims:
                image = np.flip(image, axis=flip_dims).copy()
                if cond_image.shape[0] > 0:
                    cond_image = np.flip(cond_image, axis=flip_dims).copy()

        item = dict(
            image=image.astype(np.float32),
            cond_image=cond_image.astype(np.float32),
            delta_days=np.float32(delta_days),
            idx=np.int64(entry['idx']),
        )

        if self._use_context:
            # Zero-padded to `max_history`, oldest-first, most recent
            # `max_history` scans kept if the sample's own history is longer.
            # Skipped entirely when use_context is False, to avoid the K
            # per-scan token-file reads (the whole point of --no-context).
            K = self._max_history
            n_hist = len(history_scan_ids)
            keep_ids = history_scan_ids[-K:] if n_hist > K else history_scan_ids
            keep_ages = history_ages[-K:] if n_hist > K else history_ages
            n_keep = len(keep_ids)

            context = np.zeros((K, 256, 768), dtype=np.float32)
            context_mask = np.zeros((K,), dtype=bool)
            context_ages = np.zeros((K,), dtype=np.float32)
            for i in range(n_keep):
                tok = self._load_tokens(keep_ids[i])
                context[i] = tok
                context_mask[i] = True
                context_ages[i] = keep_ages[i]

            item['context'] = context
            item['context_mask'] = context_mask
            item['context_ages'] = context_ages

        return item

    @property
    def resolution(self):
        return self._resolution

    @property
    def spacing(self):
        return self._spacing

    @property
    def num_channels(self):
        return 1

    @property
    def cond_channels(self):
        return self._cond_channels

    @property
    def sigma_data(self):
        return self._sigma_data

    @property
    def max_history(self):
        return self._max_history

    @property
    def use_context(self):
        return self._use_context

    @property
    def dataset_kwargs(self):
        """Constructor kwargs sufficient to reconstruct this dataset (contract C7)."""
        return dict(
            class_name='training.dataset.PairDataset',
            path=self._path,
            tokens_dir=self._tokens_dir,
            split=self._split,
            target_encoding=self._target_encoding,
            cond_image=self._cond_image_mode,
            max_history=self._max_history,
            token_key=self._token_key,
            flip_axes=self._flip_axes,
            use_context=self._use_context,
            spacing=self._spacing,
        )

#----------------------------------------------------------------------------
