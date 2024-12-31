import torch
from torch import nn
from utils import unpack_feature_pyramid
import torch.nn.functional as F

class LRASPPHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        internal_channels: int,
        num_classes: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        quarter, embedding = unpack_feature_pyramid(in_channels)

        self.cbr = nn.Sequential(
            nn.Conv2d(embedding, internal_channels, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embedding, internal_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.quarter_classifier = nn.Conv2d(quarter, num_classes, 1)
        self.embedding_classifier = nn.Conv2d(internal_channels, num_classes, 1)

    def forward(self, feature_pyramid: list[torch.Tensor]) -> torch.Tensor:
        [quarter, embedding] = unpack_feature_pyramid(feature_pyramid)

        x = self.cbr(embedding)
        s = self.scale(embedding)
        x = x * s

        # dont write scale=4, because irregular input sizes can round down
        #  s.t. quarter != 4 * (quarter // 4)
        B, C, H, W = quarter.shape
        # print("Quarter: ", x.shape)
        x = F.interpolate(x, size=(H, W), mode="bilinear")
        # print(x.shape)

        x = self.quarter_classifier(quarter) + self.embedding_classifier(x)
        # use scale=4 here, because original image dimension is not given
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        # print("Img: ", x.shape)
        return x