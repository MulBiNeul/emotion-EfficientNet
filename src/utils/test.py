import os, glob, argparse, csv, torch
from .common import load_cfg, get_device
from .inference import build_infer_tf, predict_image
from ..datasets.emotion_dataset import build_loaders
from ..models.efficientnet import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--input", required=True, help="이미지 파일 혹은 폴더")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = get_device(cfg.device)

    _, val_set, _ = build_loaders(cfg.data.root, cfg.data.img_size,
                                  cfg.train.batch_size, cfg.train.num_workers)
    class_names = val_set.classes

    model = build_model(cfg.train.model_name, cfg.data.num_classes, pretrained=False).to(device)
    ckpt = args.ckpt or os.path.join(cfg.save.run_dir, cfg.save.best_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()

    tf = build_infer_tf(cfg.data.img_size)

    files = []
    if os.path.isdir(args.input):
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files += glob.glob(os.path.join(args.input, e))
    else:
        files = [args.input]
    files = sorted(files)
    if not files:
        print("No images found."); return

    out_csv = args.out_csv or os.path.join(cfg.save.run_dir, "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","pred","conf"] + [f"p_{c}" for c in class_names])
        for p in files:
            label, conf, prob = predict_image(model, p, device, tf, class_names)
            print(f"{os.path.basename(p)} -> {label} ({conf:.3f})")
            w.writerow([p, label, f"{conf:.6f}"] + [f"{x:.6f}" for x in prob])
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()