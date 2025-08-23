import torch, numpy as np
from torch import amp
from contextlib import nullcontext
from sklearn.metrics import f1_score

def _amp_ctx_and_scaler(device: str, enabled: bool):
    use_amp = enabled and (device == "cuda")
    ctx = amp.autocast("cuda") if use_amp else nullcontext()
    scaler = amp.GradScaler("cuda") if use_amp else None
    return ctx, scaler

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return f1_score(y_true, y_pred, average="macro"), (y_true, y_pred)

def train_one_epoch(model, loader, optimizer, device, criterion, amp_enabled, log_interval=None, scheduler=None):
    """iter 단위 출력 없음(요청사항)."""
    model.train()
    ctx, scaler = _amp_ctx_and_scaler(device, amp_enabled)
    running = 0.0
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler and getattr(scheduler, "step_per_iter", False):
            scheduler.step()
        running += loss.item()
    return running / max(1, len(loader))