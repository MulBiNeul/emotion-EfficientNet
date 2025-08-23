import os, argparse, torch
from .common import load_cfg, get_device, ensure_dir
from ..models.efficientnet import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = get_device(cfg.device)
    ensure_dir(cfg.save.run_dir)

    model = build_model(cfg.train.model_name, cfg.data.num_classes, pretrained=False).to(device)
    ckpt = args.ckpt or os.path.join(cfg.save.run_dir, cfg.save.best_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()

    example = torch.randn(1, 3, cfg.data.img_size, cfg.data.img_size, device=device)
    ts = torch.jit.trace(model, example)
    out = args.out or os.path.join(cfg.save.run_dir, f"{cfg.train.model_name}_ts.pt")
    ts.save(out)
    print(f"TorchScript saved: {out}")

if __name__ == "__main__":
    main()