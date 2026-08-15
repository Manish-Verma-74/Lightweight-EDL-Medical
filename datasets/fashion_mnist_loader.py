"""
Fashion-MNIST, used purely as an OUT-OF-DISTRIBUTION test set relative to
MNIST. Same image shape/format as MNIST (28x28 grayscale, 10 classes), but
completely different visual content (clothing items instead of digits) --
so a model trained on MNIST has never seen anything like it.

This lets us test the actual claim from Sensoy et al. (2018): a well-behaved
EDL model should show LOW confidence / HIGH uncertainty on inputs unlike its
training data, whereas a plain softmax model has no mechanism to do this and
will typically stay falsely confident even on nonsense inputs.
"""
from torchvision import datasets, transforms

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_fashion_mnist_transform():
    # Same transform as datasets/mnist_loader.py -- must match exactly so the
    # trained model receives inputs in the same format it was trained on.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_fashion_mnist_test_set(root: str = "data/fashion_mnist"):
    transform = get_fashion_mnist_transform()
    return datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)