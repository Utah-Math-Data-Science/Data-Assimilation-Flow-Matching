from conf import inflation_scale


class InflationScale:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, sampled_state_centered):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NoScaling(InflationScale):
    pass


class ConstantScale(InflationScale):
    def forward(self, sampled_state_centered):
        return self.cfg.constant


def get_inflation_scale(cfg):
    if isinstance(cfg, inflation_scale.NoScaling):
        return NoScaling(cfg)
    elif isinstance(cfg, inflation_scale.ConstantScale):
        return ConstantScale(cfg)
    else:
        raise ValueError(f'Unknown inflation scale: {cfg}')
