"""
Builds a backbone CNN with its final layer replaced to output `num_classes`
raw evidence logits. No softmax is applied -- the EDL loss/predictions
functions handle turning these into evidence themselves.
"""
import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_BACKBONES = ["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"]


def get_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, num_classes)

    elif name == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        net = models.shufflenet_v2_x1_0(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone '{name}'. Supported: {SUPPORTED_BACKBONES}")

    return net


if __name__ == "__main__":
    # Quick shape sanity check across all three backbones
    dummy = torch.randn(2, 3, 224, 224)
    for name in SUPPORTED_BACKBONES:
        model = get_backbone(name, num_classes=7, pretrained=False)
        out = model(dummy)
        status = "OK" if out.shape == (2, 7) else "MISMATCH"
        print(f"[{status}] {name}: output shape = {tuple(out.shape)}")
