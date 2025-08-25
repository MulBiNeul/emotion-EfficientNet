import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from .common import load_cfg, get_device
from ..datasets.emotion_dataset import build_loaders
from ..models.efficientnet import build_model
from ..engine.trainer import evaluate
from .metrics import full_report
from .plot import plot_bar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--plot", action="store_true", help="클래스별 F1 막대 그래프 저장")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = get_device(cfg.device)

    train_set, val_set, test_set = build_loaders(cfg.data.root, cfg.data.img_size,
                                                 cfg.train.batch_size, cfg.train.num_workers)
    dset = val_set if args.split == "val" else test_set
    if dset is None:
        print(f"[Skip] no {args.split} set."); return
    loader = DataLoader(dset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers//2, pin_memory=True)

    from ..models.efficientnet import build_model
    model = build_model(cfg.train.model_name, cfg.data.num_classes, pretrained=False).to(device)
    ckpt = args.ckpt or os.path.join(cfg.save.run_dir, cfg.save.best_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()

    macro, (y_t, y_p) = evaluate(model, loader, device)
    rep, cm = full_report(y_t, y_p, dset.classes)
    print(f"{args.split} macro-F1: {macro:.4f}\n\n{rep}")
    np.save(os.path.join(cfg.save.run_dir, f"{args.split}_cm.npy"), cm)
    with open(os.path.join(cfg.save.run_dir, f"{args.split}_report.txt"), "w") as f:
        f.write(rep)

    if args.plot:
        from sklearn.metrics import f1_score
        per_class_f1 = []
        for c in range(len(dset.classes)):
            y_true_c = (y_t == c).astype(int)
            y_pred_c = (y_p == c).astype(int)
            per_class_f1.append(f1_score(y_true_c, y_pred_c, zero_division=0))
        plot_bar(dset.classes, per_class_f1,
                 title=f"{args.split.upper()} per-class F1",
                 out_path=os.path.join(cfg.save.run_dir, f"{args.split}_per_class_f1.png"))

if __name__ == "__main__":
    main()