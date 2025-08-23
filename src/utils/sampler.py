from collections import Counter
from torch.utils.data import WeightedRandomSampler

def make_class_aware_sampler(samples):
    counts = Counter([lbl for _, lbl in samples])
    weights = [1.0 / counts[lbl] for _, lbl in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True), counts