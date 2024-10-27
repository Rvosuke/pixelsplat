import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet

from ....dataset.types import BatchedViews
from .backbone import Backbone


@dataclass
class BackboneResnetCfg:
    """数据类`BackboneResnetCfg`，用于配置`BackboneResnet`的参数

    Attributes:
        name: 指定骨干网络的名称，固定为"resnet"。
        model: 指定使用的ResNet模型版本，可以选择不同深度的ResNet或预训练的`dino_resnet50`。
        num_layers: 指定从ResNet中提取的层数。
        use_first_pool: 决定是否在第一层后使用最大池化层。
        d_out: 指定输出特征的通道数。
    """
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int


class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: ResNet

    def __init__(self, cfg: BackboneResnetCfg, d_in: int) -> None:
        super().__init__(cfg)

        assert d_in == 3
        # 使用`InstanceNorm2d`作为归一化层，可以更好地处理小批量数据，提高模型的泛化能力。
        # 如果选择`dino_resnet50`，则从`torch.hub`加载预训练模型；
        # 否则，从`torchvision.models`加载指定的ResNet模型，并替换默认的归一化层。
        # 定义归一化层为InstanceNorm2d
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )
        # 加载指定的ResNet模型
        if cfg.model == "dino_resnet50":
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        else:
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer)

        # Set up projections
        # 设置用于特征投影的层
        # 使用`nn.ModuleDict`存储不同层的投影模块。
        # 添加一个1x1卷积，将各层的输出通道数投影到统一的`cfg.d_out`。
        self.projections = nn.ModuleDict({})
        # 对于每一层，获取其最后一个卷积层的输出通道数`d_layer_out`。
        for index in range(1, cfg.num_layers):
            key = f"layer{index}"
            block = getattr(self.model, key)
            conv_index = 1
            # 获取当前层的输出通道数
            try:
                while True:
                    d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                    conv_index += 1
            except AttributeError:
                pass
            # 添加1x1卷积用于通道数投影
            self.projections[key] = nn.Conv2d(d_layer_out, cfg.d_out, 1)

        # 为第一层添加投影
        self.projections["layer0"] = nn.Conv2d(
            self.model.conv1.out_channels, cfg.d_out, 1
        )

    def forward(
        self,
        context: BatchedViews,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # 1. 输入`context["image"]`形状为`(batch, view, channels, height, width)`。
        # 2. 使用`einops.rearrange`将批次和视角维度合并，便于模型处理。因为ResNet的输入要求为`(batch, channels, height, width)`。
        # 3. 通过ResNet的初始卷积、归一化和激活层。
        # 4. 使用之前定义的投影层`layer0`对初始特征进行通道数投影，并存储在`features`列表中。
        # 合并批次和视角维度
        b, v, _, h, w = context["image"].shape
        x = rearrange(context["image"], "b v c h w -> (b v) c h w")

        # Run the images through the resnet.
        # 通过ResNet的初始卷积和归一化层
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features = [self.projections["layer0"](x)]

        # Propagate the input through the resnet's layers.
        # 逐层通过ResNet
        for index in range(1, self.cfg.num_layers):
            key = f"layer{index}"
            if index == 0 and self.cfg.use_first_pool:
                x = self.model.maxpool(x)
            x = getattr(self.model, key)(x)
            features.append(self.projections[key](x))

        # Upscale the features.
        # 将特征上采样到原始尺寸
        # 1. 使用双线性插值，将各层的特征上采样到输入图像的尺寸。
        # 2. 将所有层的特征在通道维度上相加，融合多层信息。
        # 3. 使用`einops.rearrange`将合并的批次和视角维度恢复，得到最终的特征输出。
        features = [
            F.interpolate(f, (h, w), mode="bilinear", align_corners=True)
            for f in features
        ]
        features = torch.stack(features).sum(dim=0)

        # Separate batch dimensions.
        return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
