from functools import cache
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import conf.dataset


class Observe(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims

    def forward(self, x):
        raise NotImplementedError()

    @property
    def sparsity_factor(self):
        return 1.

    def get_sparsity_mask(self, **kwargs):
        return 1.

    @property
    def output_dims(self):
        return self.input_dims

    def linearize(self, x):
        raise NotImplementedError()


class Unflatten(Observe):
    def __init__(self, observe: Observe):
        input_dim = 1
        for d in observe.input_dims:
            input_dim *= d
        super().__init__((input_dim,))
        self.observe = observe

    def forward(self, x):
        return self.observe(x.view(-1, *self.observe.input_dims)).view(x.shape)

    @property
    def sparsity_factor(self):
        return self.observe.sparsity_factor

    def get_sparsity_mask(self):
        return self.observe.get_sparsity_mask()

    def linearize(self, x):
        return self.observe.linearize(x.view(-1, *self.observe.input_dims))


class Compose(Observe):
    def __init__(self, observes):
        if len(observes) == 0:
            raise ValueError('The list of observes must be greater than zero.')
        super().__init__(observes[0].input_dims)
        self.observe = nn.Sequential(*observes)

    def forward(self, x):
        return self.observe(x)

    def __getitem__(self, idx):
        return self.observe[idx]

    @property
    def sparsity_factor(self):
        factor = 1.
        for observe in self.observe:
            factor *= observe.sparsity_factor
        return factor

    @cache
    def get_sparsity_mask(self, **kwargs):
        mask = 1.
        for observe in self.observe:
            mask *= observe.get_sparsity_mask(**kwargs)
        return mask

    @property
    def output_dims(self):
        return self.observe[-1].output_dims

    def linearize(self, x):
        input_dim = 1
        for d in self.input_dims:
            input_dim *= d
        matrix = torch.eye(input_dim, device=x.device)
        for observe in self.observe:
            matrix = observe.linearize(x) @ matrix
        return matrix


class Compress(Observe):
    """
    Flattens the observation, removing all missing (masked) dimensions.
    """
    def __init__(self, sparsity_mask):
        super().__init__(sparsity_mask.shape)
        self.observed_indices = torch.argwhere(sparsity_mask)

    @cache
    def get_observed_indices(self, **kwargs):
        return tuple(self.observed_indices.mT.to(**kwargs))

    def forward(self, x):
        return x[(..., *self.get_observed_indices(device=x.device))]

    @property
    def output_dims(self):
        return self.observed_indices.shape[0]


class Identity(Observe):
    def forward(self, x):
        return x

    def linearize(self, x):
        return torch.eye(rearrange(x, 'batch ... -> batch (...)').shape[1], device=x.device)


class ATan(Observe):
    def forward(self, x):
        return x.atan()


class Tanh(Observe):
    def forward(self, x):
        return x.tanh()


class Pow(Observe):
    def __init__(self, input_dims, power):
        super().__init__(input_dims)
        self.power = power

    def forward(self, x):
        return x.pow(self.power)


class MaskedPatches(Observe):
    def __init__(self, input_dims, patch_count, patch_mask: conf.observe.PatchMask, fliplr=False, flipud=False):
        super().__init__(input_dims)
        self.patch_count = patch_count
        self.patch_dims = tuple(d // self.patch_count for d in self.input_dims[1:])
        self.patch_mask = patch_mask
        self.fliplr = fliplr
        self.flipud = flipud
        match patch_mask:
            case conf.observe.PatchMask.CENTER_LEFT:
                mask = torch.zeros(self.patch_dims)
                mask[:, int(math.ceil(self.patch_dims[1] / 2)) - 1] = 1.
            case conf.observe.PatchMask.CENTER_TOP:
                mask = torch.zeros(self.patch_dims)
                mask[int(math.ceil(self.patch_dims[0] / 2)) - 1] = 1.
            case conf.observe.PatchMask.LEFT:
                mask = torch.zeros(self.patch_dims)
                mask[:, 0] = 1.
            case conf.observe.PatchMask.TOP:
                mask = torch.zeros(self.patch_dims)
                mask[0] = 1.
            case conf.observe.PatchMask.DIAGONAL:
                mask = torch.eye(*self.patch_dims)
            case _:
                raise NotImplementedError()
        if fliplr:
            mask = mask.fliplr()
        if flipud:
            mask = mask.flip(0)
        self.mask = mask

    def forward(self, x):
        patch_counts = [f'patch_count_{i}' for i in range(len(self.patch_dims))]
        dims = [f'dim_{i}' for i in range(len(self.patch_dims))]
        descructure = ' '.join([f'({pc} {d})' for pc, d in zip(patch_counts, dims)])
        split = ' '.join(patch_counts) + ' ' + ' '.join(dims)
        x = rearrange(
            x,
            f'batch channel {descructure} -> batch channel {split}',
            **{pc: self.patch_count for pc in patch_counts},
        )
        x = x * self.get_mask(device=x.device)
        x = rearrange(
            x,
            f'batch channel {split} -> batch channel {descructure}',
            **{pc: self.patch_count for pc in patch_counts},
        )
        return x

    @cache
    def get_mask(self, **kwargs):
        return self.mask.to(**kwargs)

    @property
    def sparsity_factor(self):
        match self.patch_mask:
            case conf.observe.PatchMask.CENTER_LEFT | conf.observe.PatchMask.LEFT:
                observed_count = self.patch_dims[0]
            case conf.observe.PatchMask.CENTER_TOP | conf.observe.PatchMask.TOP:
                observed_count = self.patch_dims[1]
            case conf.observe.PatchMask.DIAGONAL:
                observed_count = min(self.patch_dims)
            case _:
                raise NotImplementedError()
        return observed_count / (self.patch_dims[0] * self.patch_dims[1])

    def get_sparsity_mask(self, **kwargs):
        return self.get_mask(**kwargs).tile(self.patch_count, self.patch_count)

    def __repr__(self):
        rep = f'{type(self).__name__}(patch_mask={self.patch_mask.name}'
        if self.fliplr:
            rep += ', fliplr=True'
        if self.flipud:
            rep += ', flipud=True'
        rep += ')'
        return rep


class RandomDimensions(Observe):
    def __init__(self, input_dims, rng: np.random.Generator, probability):
        super().__init__(input_dims)
        self.rng = rng
        self.probability = probability
        self.sparsity_mask = torch.tensor(
            self.rng.binomial(1, self.probability, size=self.input_dims[1:]),
        )

    def forward(self, x):
        mask = self.get_sparsity_mask(device=x.device)
        return x * mask

    @property
    def sparsity_factor(self):
        return self.probability

    @cache
    def get_sparsity_mask(self, **kwargs):
        return self.sparsity_mask.to(**kwargs)

    def __repr__(self):
        return f'{type(self).__name__}(p={self.probability})'


def build_observe(cfg_observes: list[conf.observe.Observe], *, cfg_dataset: conf.dataset.DynamicalSystemImpl = None, rng: np.random.Generator = None):
    input_dims = (cfg_dataset.channels, *cfg_dataset.spatial_dims)
    observes = []
    for cfg in sorted(cfg_observes):
        match cfg:
            case conf.observe.ObserveIdentity():
                observes.append(Identity(input_dims))
            case conf.observe.ObserveATan():
                observes.append(ATan(input_dims))
            case conf.observe.ObserveTanh():
                observes.append(Tanh(input_dims))
            case conf.observe.ObservePow():
                observes.append(Pow(input_dims, cfg.power))
            case conf.observe.ObserveMaskedPatches():
                observes.append(MaskedPatches(input_dims, cfg.patch_count, cfg.patch_mask, fliplr=cfg.fliplr, flipud=cfg.flipud))
            case conf.observe.ObserveRandomDimensions():
                if rng is None:
                    raise ValueError('The RandomDimensions observe requires a random number generator. '
                                     'Please pass the keyword argument rng.')
                observe = RandomDimensions(input_dims, rng, cfg.probability)
                rng = rng.spawn(1)[0]
                observes.append(observe)
            case conf.observe.ObserveCompress():
                observe = Compress(Compose(observes).get_sparsity_mask())
                input_dims = observe.output_dims
                observes.append(observe)
            case _:
                raise ValueError(f'Unknown observe: {cfg}')
    return Compose(observes)
