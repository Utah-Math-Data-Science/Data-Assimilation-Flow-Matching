import torch
from torch.utils.data.dataset import IterableDataset


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self):
        pass

    def __iter__(self):
        for i in range(100):
            yield torch.ones(100, 1), torch.tensor([i])
