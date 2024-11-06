"""
本文件定义了一个用于多视图3D重建或神经渲染的模块，主要涉及在多视图之间进行极线采样（Epipolar Sampling）。
包含了一个数据类`EpipolarSampling`，用于存储采样结果，以及一个类`EpipolarSampler`，用于执行极线采样的操作。
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

from ....geometry.epipolar_lines import project_rays
from ....geometry.projection import get_world_rays, sample_image_grid
from ....misc.heterogeneous_pairings import (
    Index,
    generate_heterogeneous_index,
    generate_heterogeneous_index_transpose,
)


@dataclass
class EpipolarSampling:
    features: Float[Tensor, "batch view other_view ray sample channel"]
    valid: Bool[Tensor, "batch view other_view ray"]
    xy_ray: Float[Tensor, "batch view ray 2"]
    xy_sample: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_near: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_far: Float[Tensor, "batch view other_view ray sample 2"]
    origins: Float[Tensor, "batch view ray 3"]
    directions: Float[Tensor, "batch view ray 3"]


class EpipolarSampler(nn.Module):
    num_samples: int
    index_v: Index
    transpose_v: Index
    transpose_ov: Index

    def __init__(
            self,
            num_views: int,
            num_samples: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Generate indices needed to sample only other views.
        _, index_v = generate_heterogeneous_index(num_views)
        t_v, t_ov = generate_heterogeneous_index_transpose(num_views)
        self.register_buffer("index_v", index_v, persistent=False)
        self.register_buffer("transpose_v", t_v, persistent=False)
        self.register_buffer("transpose_ov", t_ov, persistent=False)

    def forward(
            self,
            images: Float[Tensor, "batch view channel height width"],
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
    ) -> EpipolarSampling:
        device = images.device
        b, v, _, _, _ = images.shape

        # 生成投射到其他视图上的光线。
        xy_ray, origins, directions = self.generate_image_rays(images, extrinsics, intrinsics)

        # 选择要投影的相机外参和内参。对于每个上下文视图，这意味着批次中的所有其他上下文视图。
        # origins 已经分析过了，是相机的位置，directions 是射线的单位方向向量。注意同一个视角的origins是一样的, 但是directions不一样。
        # 这里的extrinsics和intrinsics是其他视角针对原始视角的。实际上每个视角的外参和内参都是一样的，只是针对不同的视角便于计算。
        # 例如extrinsics[0, 1, 2]表示视角2的相机外参。
        projection = project_rays(
            rearrange(origins, "b v r xyz -> b v () r xyz"),
            rearrange(directions, "b v r xyz -> b v () r xyz"),
            rearrange(self.collect(extrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(self.collect(intrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(near, "b v -> b v () ()"),
            rearrange(far, "b v -> b v () ()"),
        )

        # Generate sample points.
        s = self.num_samples
        sample_depth = (torch.arange(s, device=device) + 0.5) / s
        sample_depth = rearrange(sample_depth, "s -> s ()")
        xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0)
        xy_min = xy_min * projection["overlaps_image"][..., None]
        xy_min = rearrange(xy_min, "b v ov r xy -> b v ov r () xy")
        xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0)
        xy_max = xy_max * projection["overlaps_image"][..., None]
        xy_max = rearrange(xy_max, "b v ov r xy -> b v ov r () xy")
        xy_sample = xy_min + sample_depth * (xy_max - xy_min)

        # 样本的形状是(batch, view, other_view, ...)。
        # 但是，在转置之前，视角维度指的是光线发射的视角，而不是样本提取的视角。
        # 因此，我们需要对样本进行转置，以使视角维度指的是样本提取的视角。
        # 如果不是为了效率而去掉对角线，这将是一个字面上的转置。
        # 在我们的情况下，就好像对角线被重新添加，然后进行转置，最后又去掉对角线。
        samples = self.transpose(xy_sample)
        samples = F.grid_sample(
            rearrange(images, "b v c h w -> (b v) c h w"),
            rearrange(2 * samples - 1, "b v ov r s xy -> (b v) (ov r s) () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        samples = rearrange(
            samples, "(b v) c (ov r s) () -> b v ov r s c", b=b, v=v, ov=v - 1, s=s
        )
        samples = self.transpose(samples)

        # Zero out invalid samples.
        samples = samples * projection["overlaps_image"][..., None, None]

        half_span = 0.5 / s
        return EpipolarSampling(
            features=samples,
            valid=projection["overlaps_image"],
            xy_ray=xy_ray,
            xy_sample=xy_sample,
            xy_sample_near=xy_min + (sample_depth - half_span) * (xy_max - xy_min),
            xy_sample_far=xy_min + (sample_depth + half_span) * (xy_max - xy_min),
            origins=origins,
            directions=directions,
        )

    @staticmethod
    def generate_image_rays(
            images: Float[Tensor, "batch view channel height width"],
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> tuple[
        Float[Tensor, "batch view ray 2"],  # xy
        Float[Tensor, "batch view ray 3"],  # origins
        Float[Tensor, "batch view ray 3"],  # directions
    ]:
        """Generate the rays along which Gaussians are defined. For now, these rays are
        simply arranged in a grid.

        Args:
            images: 批量输入的多视角图像。
            extrinsics: 同上，多视角的相机外参。
            intrinsics: 同上，多视角的相机内参。

        Returns:
            - `xy`： 所有批量+视角的像素网格的归一化坐标, ray = (h * w), 是直接将索引展平了.
            - `origins`： 批量+多视角的射线起点，实际上是相机的位置。
            - `directions`： 批量+多视角的射线的单位方向向量，经过了归一化。
        """
        b, v, _, h, w = images.shape
        xy, _ = sample_image_grid((h, w), device=images.device)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def transpose(
            self,
            x: Shaped[Tensor, "batch view other_view *rest"],
    ) -> Shaped[Tensor, "batch view other_view *rest"]:
        b, v, ov, *_ = x.shape
        t_b = torch.arange(b, device=x.device)
        t_b = repeat(t_b, "b -> b v ov", v=v, ov=ov)
        t_v = repeat(self.transpose_v, "v ov -> b v ov", b=b)
        t_ov = repeat(self.transpose_ov, "v ov -> b v ov", b=b)
        return x[t_b, t_v, t_ov]

    def collect(
            self,
            target: Shaped[Tensor, "batch view ..."],
    ) -> Shaped[Tensor, "batch view view-1 ..."]:
        """将目标张量收集到一个可以跨视图维度广播的张量中。这是通过为每个视图重复张量来完成的，但不包括视图本身。
        实际上, 最终返回的张量的形状是 (batch, view, other_view, ...)。
        这意味着对于每个视角, 找到除其本身外的其他视角的相关张量(内外参矩阵)。
        """
        b, v, *_ = target.shape
        index_b = torch.arange(b, device=target.device)
        index_b = repeat(index_b, "b -> b v ov", v=v, ov=v - 1)
        index_v = repeat(self.index_v, "v ov -> b v ov", b=b)  # 对应index_v的详细解释见misc/heterogeneous_pairings.py
        return target[index_b, index_v]  # 这里使用了高级索引，实现了收集操作
