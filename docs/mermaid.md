## 📊 Workflow (Mermaid)

flowchart TD

subgraph DATA[데이터셋]
A1["이미지 폴더 train/val/test"] --> A2[build_transforms]
A2 --> A3[build_loaders]
end

subgraph MODEL[모델]
B1["build_model EfficientNet-B0"]
B2["Linear classifier (7 classes)"]
B1 --> B2
end

subgraph TRAIN[학습]
C1[train_one_epoch]
C2["evaluate (val)"]
C3["loss functions (CE, Focal, CB-Focal, LogitAdjusted)"]
C1 --> C2
C3 --> C1
end

subgraph UTILS[유틸]
D1["metrics.py - macroF1"]
D2["plot.py - curves & bars"]
D3["sampler.py - class-aware"]
D4["mixup.py - mixup/cutmix/remix"]
end

DATA --> TRAIN
MODEL --> TRAIN
TRAIN -->|best.pt 저장| CKPT[(Checkpoint)]
CKPT --> E1[evaluate.py]
CKPT --> E2[test.py]
CKPT --> E3[inference.py]
