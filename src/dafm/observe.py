import conf.observe


class Observe:
    def __init__(self, cfg, cfg_dataset):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset

    def forward(self, state):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Full(Observe):
    def forward(self, state):
        return state


class EveryNthDimension(Observe):
    def __init__(self, cfg, cfg_dataset):
        super().__init__(cfg, cfg_dataset)
        if not cfg.start_at_zero and cfg_dataset.state_dimension < cfg.n:
            raise ValueError(
                f'Cannot observe dimension {cfg.n} of a dataset with only {cfg_dataset.state_dimension} dimensions.'
                f' Please either set dataset.observe.start_at_zero=true to include dimension zero, or set dataset.observe.n to be between 1 and {cfg_dataset.state_dimension} (inclusive).'
            )

    def forward(self, state):
        if self.cfg.start_at_zero:
            return state[..., ::self.cfg.n]
        else:
            return state[..., 1::self.cfg.n]


class Exponentiate(Observe):
    def forward(self, state):
        return state**self.cfg.exponent


class ATan(Observe):
    def forward(self, state):
        return state.atan()


class ATanEveryNthDimension(EveryNthDimension):
    def forward(self, state):
        return super().forward(state).atan()


def get_observe(cfg):
    if isinstance(cfg.observe, conf.observe.Full):
        return Full(cfg.observe, cfg)
    elif isinstance(cfg.observe, conf.observe.EveryNthDimension):
        return EveryNthDimension(cfg.observe, cfg)
    elif isinstance(cfg.observe, conf.observe.Exponentiate):
        return Exponentiate(cfg.observe, cfg)
    elif isinstance(cfg.observe, conf.observe.ATan):
        return ATan(cfg.observe, cfg)
    elif isinstance(cfg.observe, conf.observe.ATanEveryNthDimension):
        return ATanEveryNthDimension(cfg.observe, cfg)
    else:
        raise ValueError(f'Unknown observe: {cfg.observe}')
