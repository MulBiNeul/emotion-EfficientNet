import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .common import load_cfg, set_seed, get_device, ensure_dir, override_cfg
from ..datasets.emotion_dataset import build_loaders
from ..models.efficientnet import build_model
from ..engine.trainer import train_one_epoch, evaluate
from .metrics import full_report
from .plot import save_history, plot_curve
from ..losses import FocalLoss, ClassBalancedFocal, LogitAdjustedCE


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--model", choices=["efficientnet_b0"], default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--amp", default=None, choices=["auto", "true", "false"])
    ap.add_argument("--loss", default=None, choices=["ce_ls", "focal", "cb_focal", "logit_adjusted"])
    return ap.parse_args()


def build_loss(cfg, train_set):
    ltype = cfg.loss.type
    if ltype == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    elif ltype == "focal":
        return FocalLoss(gamma=cfg.loss.focal_gamma)
    elif ltype == "cb_focal":
        counts = np.zeros(cfg.data.num_classes, dtype=np.float32)
        for _, label in train_set.samples:
            counts[label] += 1
        return ClassBalancedFocal(samples_per_class=counts, beta=cfg.loss.cb_beta, gamma=cfg.loss.focal_gamma)
    elif ltype == "logit_adjusted":
        counts = np.zeros(cfg.data.num_classes, dtype=np.float32)
        for _, label in train_set.samples:
            counts[label] += 1
        priors = counts / max(1.0, counts.sum())
        return LogitAdjustedCE(priors=priors, la_tau=cfg.loss.la_tau)
    else:
        raise ValueError(f"Unknown loss type: {ltype}")


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    cfg = override_cfg(
        cfg,
        model_name=args.model,
        run_dir=args.run_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        amp=(True if args.amp == "true" else False if args.amp == "false" else "auto" if args.amp == "auto" else None),
    )
    if args.loss is not None:
        cfg.loss.type = args.loss

    # 시드/디바이스
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # 데이터
    train_set, val_set, _ = build_loaders(
        cfg.data.root, cfg.data.img_size, cfg.train.batch_size, cfg.train.num_workers
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.train.num_workers // 2),
        pin_memory=True,
    )
    class_names = train_set.classes

    # 모델 (EfficientNet-B0)
    model = build_model(cfg.train.model_name, num_classes=cfg.data.num_classes, pretrained=cfg.train.use_pretrained).to(device)

    # 백본 프리즈 옵션
    if cfg.train.freeze_backbone and hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    criterion = build_loss(cfg, train_set).to(device)

    # AMP
    amp_enabled = (cfg.train.amp == "auto" and device == "cuda") or (cfg.train.amp is True)

    # 저장 경로
    run_dir = cfg.save.run_dir
    ensure_dir(run_dir)
    best_ckpt = os.path.join(run_dir, cfg.save.best_name)
    best_metric = -1.0

    # 기록용
    history = {"train_loss": [], "val_macroF1": []}

    # 학습 루프
    t0 = time.time()
    for e in tqdm(range(1, cfg.train.epochs + 1), desc="Epochs", total=cfg.train.epochs):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion, amp_enabled, log_interval=None
        )
        val_macro, (y_t, y_p) = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(float(tr_loss))
        history["val_macroF1"].append(float(val_macro))

        # 리포트/혼동행렬 저장
        rep_txt, cm = full_report(y_t, y_p, class_names)
        with open(os.path.join(run_dir, "val_report.txt"), "w") as f:
            f.write(rep_txt)
        np.save(os.path.join(run_dir, "val_cm.npy"), cm)

        # 체크포인트 저장
        if val_macro > best_metric:
            best_metric = val_macro
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": e,
                    "val_macroF1": float(val_macro),
                    "class_to_idx": train_set.class_to_idx,
                },
                best_ckpt,
            )

    # 총 학습 시간 출력
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nBest Macro-F1: {best_metric:.4f}")
    print(f"Total training time: {h:02d}:{m:02d}:{s:02d}")

    # 학습 곡선 저장
    save_history(run_dir, history)
    plot_curve(history["train_loss"], "Train Loss", "loss", os.path.join(run_dir, "train_loss.png"))
    plot_curve(history["val_macroF1"], "Validation Macro-F1", "macroF1", os.path.join(run_dir, "val_macroF1.png"))


if __name__ == "__main__":
    main()