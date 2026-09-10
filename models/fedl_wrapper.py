import torch.nn as nn

from models.backbone_factory import get_backbone
from losses.fedl_loss import FEDLHead


class FEDLWrapper(nn.Module):
    """
    Wrap an existing lightweight CNN backbone with an F-EDL head.

    Supported:
        - EfficientNet-B0
        - MobileNetV3-Small
        - ShuffleNetV2
    """

    def __init__(
        self,
        backbone_name,
        num_classes,
        pretrained=True,
    ):
        super().__init__()

        # Build the normal backbone
        base_model = get_backbone(
            backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        # ---------------------------------------------------------
        # Torchvision-style models using .classifier
        # ---------------------------------------------------------
        if hasattr(base_model, "classifier"):

            classifier = base_model.classifier

            if isinstance(classifier, nn.Sequential):

                # We expect the LAST layer to be Linear.
                final_layer = classifier[-1]

                if not isinstance(final_layer, nn.Linear):
                    raise ValueError(
                        f"{backbone_name}: expected final "
                        f"classifier layer to be nn.Linear, "
                        f"but got {type(final_layer)}"
                    )

                in_features = final_layer.in_features

                # IMPORTANT:
                # Keep the feature-producing layers.
                # Replace ONLY the final classification layer.
                classifier[-1] = nn.Identity()

            else:

                if not isinstance(classifier, nn.Linear):
                    raise ValueError(
                        f"{backbone_name}: unsupported "
                        f"classifier type: {type(classifier)}"
                    )

                in_features = classifier.in_features

                base_model.classifier = nn.Identity()

        # ---------------------------------------------------------
        # Models using .fc
        # ---------------------------------------------------------
        elif hasattr(base_model, "fc"):

            fc = base_model.fc

            if not isinstance(fc, nn.Linear):
                raise ValueError(
                    f"{backbone_name}: unsupported fc type: "
                    f"{type(fc)}"
                )

            in_features = fc.in_features

            base_model.fc = nn.Identity()

        else:

            raise ValueError(
                f"Unsupported backbone structure: "
                f"{backbone_name}"
            )

        self.backbone = base_model

        self.fedl_head = FEDLHead(
            in_features=in_features,
            num_classes=num_classes,
        )

    def forward(self, x):

        features = self.backbone(x)

        alpha, p, tau = self.fedl_head(
            features
        )

        return alpha, p, tau