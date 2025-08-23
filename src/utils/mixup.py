import math, random, numpy as np, torch

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def remix_lambda(lam_t, y_a, y_b, cls_counts, tau=5):
    if not isinstance(lam_t, torch.Tensor):
        lam_t = torch.tensor(lam_t, dtype=torch.float32, device=y_a.device)
    c1 = cls_counts[y_a.detach().cpu().numpy()]
    c2 = cls_counts[y_b.detach().cpu().numpy()]
    delta = np.log((c1+1e-6)/(c2+1e-6))
    adj = 1.0 / (1.0 + np.exp(-delta / tau))
    lam_np = lam_t.detach().cpu().numpy()
    lam_np = lam_np * (1.0 - adj) + (1.0 - lam_np) * adj
    return torch.tensor(lam_np, dtype=torch.float32, device=y_a.device)

def apply_mix(inputs, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    x = lam * inputs + (1 - lam) * inputs[index]
    return x, targets, targets[index], torch.full((inputs.size(0),), lam, device=inputs.device)

def apply_cutmix(inputs, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
    lam_eff = 1 - ((x2-x1)*(y2-y1) / (inputs.size(-1)*inputs.size(-2)))
    return inputs, targets, targets[index], torch.full((inputs.size(0),), lam_eff, device=inputs.device)

def mixup_cutmix_remix(inputs, targets, cfg, cls_counts):
    r = random.random()
    do_mix = cfg.train.use_mixup and (r < 0.5)
    do_cut = cfg.train.use_cutmix and (not do_mix)
    if not (do_mix or do_cut):
        return inputs, targets, targets, torch.ones(inputs.size(0), device=inputs.device)
    if do_mix:
        x, y_a, y_b, lam_t = apply_mix(inputs, targets, cfg.train.mixup_alpha)
    else:
        x, y_a, y_b, lam_t = apply_cutmix(inputs, targets, cfg.train.cutmix_alpha)
    if cfg.train.use_remix:
        lam_t = remix_lambda(lam_t, y_a, y_b, cls_counts)
    return x, y_a, y_b, lam_t