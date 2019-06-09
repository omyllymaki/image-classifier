from torchvision import transforms

from constants import IMAGE_NET_MEAN, IMAGE_NET_STD, IMAGE_NET_IMAGE_SIZE

TransformsTraining = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=IMAGE_NET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])

TransformsTest = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=IMAGE_NET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])
