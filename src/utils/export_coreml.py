import os, argparse, torch, coremltools as ct
from .common import load_cfg, get_device, ensure_dir
from ..models.efficientnet import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = get_device(cfg.device)
    ensure_dir(cfg.save.run_dir)

    model = build_model(cfg.train.model_name, cfg.data.num_classes, pretrained=False).to(device)
    ckpt = args.ckpt or os.path.join(cfg.save.run_dir, cfg.save.best_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()

    example = torch.randn(1, 3, cfg.data.img_size, cfg.data.img_size, device=device)
    ts = torch.jit.trace(model, example).to("cpu")

    mlmodel = ct.convert(
        ts,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        inputs=[ct.TensorType(name="input", shape=example.cpu().shape)]
    )
    if args.fp16:
        mlmodel = ct.models.utils.convert_neural_network_weights_to_fp16(mlmodel)

    out_path = args.out or os.path.join(cfg.save.run_dir, "EmotionClassifier.mlmodel")
    mlmodel.save(out_path)
    print(f"CoreML saved: {out_path}")

if __name__ == "__main__":
    main()