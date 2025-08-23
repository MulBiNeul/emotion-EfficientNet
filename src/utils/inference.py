import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD =[0.229,0.224,0.225]

def build_infer_tf(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

@torch.no_grad()
def predict_image(model, img_path, device, tf, class_names):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(prob.argmax())
    return class_names[idx], float(prob[idx]), prob