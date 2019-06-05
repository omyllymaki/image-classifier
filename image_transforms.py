from torchvision import transforms

from constants import IMAGE_NET_MEAN, IMAGE_NET_STD

TransformsTraining = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])

TransformsTest = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])
