"""
Loads the MNIST dataset and converts its grayscale 28x28 images into
3-channel 224x224 images with ImageNet normalization, so they can be
used with our ImageNet-pretrained EfficientNet, MobileNetV3, and
ShuffleNet backbones.

MNIST wrapped for compatibility with our RGB, ImageNet-pretrained backbones.
MNIST is natively 1-channel 28x28; we replicate to 3 channels and resize to
224x224 so EfficientNet-B0 / MobileNetV3-Small / ShuffleNetV2 can consume it
with their pretrained weights intact.

This is a validation dataset only -- used to sanity-check that our EDL loss
(losses/evidential_loss.py) and ECE metric (metrics/ece.py) behave correctly
on real training dynamics, not synthetic data. It is NOT part of the thesis's
main HAM10000 experiments.
"""
from torchvision import datasets, transforms

MNIST_CLASSES = [str(i) for i in range(10)]


def get_mnist_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_mnist_datasets(root: str = "data/mnist"):
    """Returns (train_dataset, test_dataset), downloading MNIST if needed."""
    transform = get_mnist_transform()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train_ds, test_ds