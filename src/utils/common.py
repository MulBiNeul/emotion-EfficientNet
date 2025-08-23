import os, random, yaml, torch, numpy as np
from types import SimpleNamespace

def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d

def load_cfg(path="configs/default.yaml"):
    with open(path, "r") as f:
        return _to_ns(yaml.safe_load(f))

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(pref="auto"):
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    if pref == "mps" or (pref == "auto" and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def override_cfg(cfg, **kvs):
    mapping = {
        "model_name": ("train", "model_name"),
        "lr": ("train", "lr"),
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "run_dir": ("save", "run_dir"),
        "img_size": ("data", "img_size"),
        "amp": ("train", "amp"),
    }
    for k, v in kvs.items():
        if v is None:
            continue
        if k in mapping:
            sec, key = mapping[k]
            setattr(getattr(cfg, sec), key, v)
    return cfg