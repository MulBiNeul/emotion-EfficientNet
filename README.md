


📂 Source Code Overview (src/)

🗂 datasets

datasets/emotion_dataset.py
	•	build_transforms(img_size=224)
학습/평가용 전처리(transform) 정의 (Resize, Flip·ColorJitter, ToTensor, Normalize).
	•	build_datasets(data_root, img_size)
ImageFolder로 train/val(/test) 데이터셋 생성.
	•	build_loaders(data_root, img_size, batch_size, num_workers=8, drop_last=True)
위 데이터셋으로 DataLoader 3종(train/val/test) 반환.

⸻

🗂 engine

engine/trainer.py
	•	_amp_ctx_and_scaler(device, enabled)
AMP 지원 시 autocast, GradScaler 반환.
	•	evaluate(model, loader, device)
모델 평가 → macro-F1와 (y_true, y_pred) 반환.
	•	train_one_epoch(...)
1 epoch 학습. AMP/스케줄러 지원. 평균 loss 반환.

⸻

🗂 losses

losses/focal.py
	•	FocalLoss
CrossEntropy 기반 focal loss.
loss = ((1 - pt)^γ) * CE

losses/cb_focal.py
	•	ClassBalancedFocal
클래스 분포 기반 가중치 적용 후 focal loss.
	•	LogitAdjustedCE
클래스 priors 기반으로 로짓 보정한 CE loss.

losses/__init__.py
	•	통합 import: FocalLoss, ClassBalancedFocal, LogitAdjustedCE.

⸻

🗂 models

models/efficientnet.py
	•	build_efficientnet_b0(num_classes, pretrained)
EfficientNet-B0 불러와 classifier를 Linear(num_classes)로 교체.
	•	build_model(name, num_classes, pretrained)
현재 "efficientnet_b0"만 지원.

⸻

🗂 utils

utils/common.py
	•	load_cfg(path) : YAML 설정 → SimpleNamespace 변환.
	•	set_seed(seed) : random/numpy/torch 시드 고정.
	•	get_device(pref) : "cuda" | "mps" | "cpu" 결정.
	•	ensure_dir(p) : 디렉토리 생성.
	•	override_cfg(cfg, **kvs) : CLI 인자로 config 덮어쓰기.

utils/evaluate.py
	•	main() : 모델 로드 후 val/test 평가.
	•	출력: macro-F1, classification report(txt), confusion matrix(npy).
	•	--plot 옵션: per-class F1 막대그래프 저장.

utils/inference.py
	•	build_infer_tf(img_size) : 추론용 전처리 생성.
	•	predict_image(model, img_path, device, tf, class_names) :
단일 이미지 추론 → (pred_label, confidence, prob_dist) 반환.

utils/metrics.py
	•	macro_f1(y_true, y_pred) : macro-F1 계산.
	•	full_report(y_true, y_pred, target_names) : classification report + confusion matrix.

utils/mixup.py
	•	rand_bbox(size, lam) : CutMix bounding box 생성.
	•	remix_lambda(...) : 클래스 빈도 기반 λ 보정.
	•	apply_mix(...) : MixUp 적용.
	•	apply_cutmix(...) : CutMix 적용.
	•	mixup_cutmix_remix(...) : cfg 옵션에 따라 MixUp/CutMix/Remix 실행.

utils/plot.py
	•	save_history(run_dir, history) : 학습 이력 JSON 저장.
	•	plot_curve(y, title, ylabel, out_path) : 학습 곡선 시각화.
	•	plot_bar(labels, values, title, out_path) : per-class 성능 막대그래프.

utils/sampler.py
	•	make_class_aware_sampler(samples)
클래스 불균형 대응 WeightedRandomSampler + 클래스 개수 반환.

utils/test.py
	•	main() : 단일 이미지/폴더 예측 후 CSV(predictions.csv) 저장.
	•	CSV 컬럼: path, pred, conf, p_<class...>

utils/train.py
	•	parse_args() : CLI 인자 파싱.
	•	build_loss(cfg, train_set) : 설정에 맞는 loss 객체 생성.
	•	main() :
	•	config 로드/덮어쓰기 → 시드/디바이스 설정
	•	데이터로더/모델/optimizer/scheduler/loss 초기화
	•	epoch 루프: 학습 → 평가 → 기록 저장
	•	최고 성능 모델 저장(best.pt), 곡선 그래프 및 리포트 출력

⸻

📊 Workflow Diagram

## 📊 Workflow

![Workflow](docs/workflow.png)
