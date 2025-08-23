from torchvision import datasets, transforms

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD =[0.229,0.224,0.225]

def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def build_loaders(data_root, img_size, batch_size, num_workers):
    train_tf, eval_tf = build_transforms(img_size)
    train_set = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_set   = datasets.ImageFolder(f"{data_root}/val",   transform=eval_tf)
    try:
        test_set = datasets.ImageFolder(f"{data_root}/test", transform=eval_tf)
    except Exception:
        test_set = None
    return train_set, val_set, test_set