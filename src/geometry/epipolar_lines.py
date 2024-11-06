import itertools
from typing import Iterable, Literal, Optional, TypedDict

import torch
from einops import einsum, repeat
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .projection import (
    get_world_rays,
    homogenize_points,
    homogenize_vectors,
    intersect_rays,
    project_camera_space,
)


def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)


def _is_in_front_of_camera(
    xyz: Float[Tensor, "*batch 3"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified points in camera space are in front of the camera."""
    return xyz[..., -1] > -epsilon


def _is_positive_t(
    t: Float[Tensor, " *batch"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified t value is positive."""
    return t > -epsilon


class PointProjection(TypedDict):
    t: Float[Tensor, " *batch"]  # ray parameter, as in xyz = origin + t * direction
    xy: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # A "valid" projection satisfies two conditions:
    # 1. It is in front of the camera (i.e., its 3D Z coordinate is positive).
    # 2. It is within the image frame (i.e., its 2D coordinates are between 0 and 1).
    valid: Bool[Tensor, " *batch"]


def _intersect_image_coordinate(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    dimension: Literal["x", "y"],
    coordinate_value: float,
) -> PointProjection:
    """计算相机空间光线的投影与一条与图像框架平行的线（水平或垂直）的交点。

    极线与边框相交有三种情况：
    1. 有一个交点，且值在[0, 1]之间，即交点在图像框内，这是最理想的情况。
    2. 有一个交点，但值不在[0, 1]之间，我们将其标记为无效。
    3. 无交点，这种情况下交点被设置为无穷远。
    实际上对于第3种情况，并不需要我们手动识别，坐标值在计算过程中会通过“除以0”得到无穷。

    Args:
        intrinsics: 相机内参矩阵，形状为[*#batch, 3, 3]。
        origins: 射线在目标相机坐标系的起点，形状为[*#batch, 3]。
        directions: 射线在目标相机坐标系的方向，形状为[*#batch, 3]。
        dimension: 字符串，值为"x"或"y"，表示要计算的图像坐标的维度。
        coordinate_value: 浮点数，表示要计算的图像

    Returns:
        - PointProjection字典，包含以下键：
        - t： 张量，形状为*batch，表示射线参数，即射线方程中的参数t，满足xyz = origin + t * direction。
        - xy： 张量，形状为*batch 2，表示交点的图像坐标（有效交点需在0到1之间）。
        - valid： 布尔张量，形状为*batch，表示交点是否有效
    """

    # Define shorthands.
    dim = "xy".index(dimension)
    other_dim = 1 - dim
    fs = intrinsics[..., dim, dim]  # focal length, same coordinate
    fo = intrinsics[..., other_dim, other_dim]  # focal length, other coordinate
    cs = intrinsics[..., dim, 2]  # principal point, same coordinate
    co = intrinsics[..., other_dim, 2]  # 主点，其他坐标
    os = origins[..., dim]  # ray origin, same coordinate
    oo = origins[..., other_dim]  # ray origin, other coordinate
    ds = directions[..., dim]  # ray direction, same coordinate
    do = directions[..., other_dim]  # ray direction, other coordinate
    oz = origins[..., 2]  # ray origin, z coordinate
    dz = directions[..., 2]  # ray direction, z coordinate
    c = (coordinate_value - cs) / fs  # coefficient (computed once and factored out), 找出边框点在三维空间中的坐标

    # 计算交点处 t 的值。
    # Note: Infinite values of t are fine. No need to handle division by zero.
    t_numerator = c * oz - os
    t_denominator = ds - c * dz
    t = t_numerator / t_denominator

    # 计算交点处另一坐标的值。
    # 注意：无限坐标值是可以接受的，我们无需处理“除以零”的问题。
    # 具体来说，当计算出的坐标值为无穷时，说明射线与图像边框平行，没有交点。
    coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
    coordinate_denominator = dz * os - ds * oz
    coordinate_other = co + coordinate_numerator / coordinate_denominator

    # 将交点的坐标值组合为一个张量。
    coordinate_same = torch.ones_like(coordinate_other) * coordinate_value
    xy = [coordinate_same]
    xy.insert(other_dim, coordinate_other)
    xy = torch.stack(xy, dim=-1)
    xyz = origins + t[..., None] * directions

    # These will all have exactly the same batch shape (no broadcasting necessary). In
    # terms of jaxtyping annotations, they all match *batch, not just *#batch.
    # 关于t值，反映的是从源相机出发的射线长度，其实很有可能射线与图像边框只有一个交点，所以那些没交点的t值是无穷的，也被视作无效。
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


def _compare_projections(
    intersections: Iterable[PointProjection],
    reduction: Literal["min", "max"],
) -> PointProjection:
    """从一组射线与图像边框的交点中，选择一个最佳的交点（最小或最大t值的交点, 一般是最小），并返回对应的射线参数t、图像坐标xy以及有效性valid。

    Args:
        intersections: 一个可迭代的PointProjection，其中每个元素都是一个字典，包含t、xy和valid。
        reduction: 字符串，值为"min"或"max"，表示要选择最小或最大的t值。
    """

    # default_collate：这是PyTorch中用于将一组数据样本合并为一个批次的函数。对于字典类型，它会递归地将相同键的值合并为一个张量。
    # eg. default_collate([{"a": torch.tensor(1), "b": torch.tensor(2)}, {"a": torch.tensor(3), "b": torch.tensor(4)}])
    # 返回：{"a": torch.tensor([1, 3]), "b": torch.tensor([2, 4])}
    # 假设intersections包含4个PointProjection，对应左、右、上、下边框的交点。
    # default_collate会将它们的t值堆叠为一个形状为[4, *batch]的张量，同理对xy和valid。
    # clone()：用于复制张量，避免在后续操作中修改原始数据。
    intersections = {k: v.clone() for k, v in default_collate(intersections).items()}
    t = intersections["t"]  # shape: [4, *batch]
    xy = intersections["xy"]  # shape: [4, *batch, 2]
    valid = intersections["valid"]  # shape: [4, *batch]

    # Make sure out-of-bounds values are not chosen.
    # 在进行最小或最大值计算时，确保无效的t值不会被选中。将无效位置设置为与目标反向的正无穷或负无穷。
    # 如果reduction为"min"，则将无效的t值设置为正无穷torch.inf，这样在min操作中，它们不会被选为最小值。
    # 如果reduction为"max"，则将无效的t值设置为负无穷-torch.inf，这样在max操作中，它们不会被选为最大值。
    lowest_priority = {
        "min": torch.inf,
        "max": -torch.inf,
    }[reduction]
    t[~valid] = lowest_priority  # 使用布尔索引，将valid为False的位置(即无效位置)的t值替换为lowest_priority。

    # Run the reduction (either t.min() or t.max()).
    # getattr是Python的内置函数，用于获取对象的属性或方法。这里相当于调用t.min(dim=0)或t.max(dim=0)。
    # reduce: 得到沿第0维（交点维度）的最小或最大t值，形状为[*batch]。
    # selector: 得到最小或最大t值对应的索引，即哪个交点产生了这个t值，形状为[*batch]。
    reduced, selector = getattr(t, reduction)(dim=0)

    # Index the results.
    # gather函数用于根据指定的索引从张量中提取数据。gather(0...)表示沿第0维进行索引。
    # 因为是对批量操作， 所以需要repeat函数将selector的形状从[*batch]扩展为[*batch, 2]，以便与xy进行广播。
    # 当然，当四个交点都是无效值的时候，返回的t也是无穷的。
    return {
        "t": reduced,
        "xy": xy.gather(0, repeat(selector, "... -> () ... xy", xy=2))[0],
        "valid": valid.gather(0, selector[None])[0],
    }


def _compute_point_projection(
    xyz: Float[Tensor, "*#batch 3"],
    t: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> PointProjection:
    xy = project_camera_space(xyz, intrinsics)
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def project_rays(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    near: Optional[Float[Tensor, "*#batch"]] = None,
    far: Optional[Float[Tensor, "*#batch"]] = None,
    epsilon: float = 1e-6,
) -> RaySegmentProjection:
    """顾名思义，这个函数的目的是将极线投影到图像平面上。
    首先要思考一些问题：
    1. 图像平面内的直线如何表示？
    2. 目标平面如何表示？
    """
    # Transform the rays into camera space.
    # 注意输入的origins和directions是世界坐标系下的。我们假设他们都是对应当前视角v1的射线, 现在要计算这些射线在其他视角v2的投影。
    # 而如前述, extrinsics和intrinsics是其他视角的相机参数。
    world_to_cam = torch.linalg.inv(extrinsics)
    origins = homogenize_points(origins)
    origins = einsum(world_to_cam, origins, "... i j, ... j -> ... i")
    directions = homogenize_vectors(directions)
    directions = einsum(world_to_cam, directions, "... i j, ... j -> ... i")
    origins = origins[..., :3]
    directions = directions[..., :3]

    # 计算与图像边框的交点
    # 我们无法预知极线具体与哪个边框相交，所以需要分别计算与左、右、上、下边框的交点，然后从中选择最佳的交点。
    # 有一个问题：如果极线与图像边框没有交点，那么我们应该如何处理？这里的处理方式是将交点设置为无穷远，这样在后续计算中，这些无效的交点不会被选中。
    frame_intersections = (
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 0.0),  # 左边框
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 1.0),  # 右边框
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 0.0),  # 上边框
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 1.0),  # 下边框
    )
    frame_intersection_min = _compare_projections(frame_intersections, "min")
    frame_intersection_max = _compare_projections(frame_intersections, "max")

    if near is None:
        # Compute the ray's projection at zero depth. If an origin's depth (z value) is
        # within epsilon of zero, this can mean one of two things:
        # 1. The origin is at the camera's position. In this case, use the direction
        #    instead (the ray is probably coming from the camera).
        # 2. The origin isn't at the camera's position, and randomly happens to be on
        #    the plane at zero depth. In this case, its projection is outside the image
        #    plane, and is thus marked as invalid.
        origins_for_projection = origins.clone()  # 创建origins的副本，防止修改原始数据。origins_for_projection将用于计算投影。
        mask_depth_zero = origins_for_projection[..., -1] < epsilon  # mask_depth_zero是一个布尔张量，标记射线起点的z坐标接近于零的视角。
        mask_at_camera = origins_for_projection.norm(dim=-1) < epsilon  # 标记射线起点在相机坐标系原点（相机位置）的位置。
        origins_for_projection[mask_at_camera] = directions[mask_at_camera]  # 对于位于相机位置的射线, 可以使用方向向量来计算投影。此时的极线投影就是一个点。
        projection_at_zero = _compute_point_projection(
            origins_for_projection,
            torch.zeros_like(frame_intersection_min["t"]),  # 对应的t值全部为零，因为我们在深度为零处计算投影。
            intrinsics,
        )
        # 对于z接近零但不在相机位置的射线，将其投影标记为无效。
        # 当z=0时，点位于相机的成像平面上，可能导致投影计算中的除零错误，避免数值不稳定性。
        projection_at_zero["valid"][mask_depth_zero & ~mask_at_camera] = False
    else:
        # If a near plane is specified, use it instead.
        t_near = near.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_zero = _compute_point_projection(
            origins + near[..., None] * directions,
            t_near,
            intrinsics,
        )

    if far is None:
        # Compute the ray's projection at infinite depth. Using the projection function
        # with directions (vectors) instead of points may seem wonky, but is equivalent
        # to projecting the point at (origins + infinity * directions).
        projection_at_infinity = _compute_point_projection(
            directions,
            torch.ones_like(frame_intersection_min["t"]) * torch.inf,
            intrinsics,
        )
    else:
        # If a far plane is specified, use it instead.
        t_far = far.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_infinity = _compute_point_projection(
            origins + far[..., None] * directions,
            t_far,
            intrinsics,
        )

    # Build the result by handling cases for ray intersection.
    result = {
        "t_min": torch.empty_like(projection_at_zero["t"]),
        "t_max": torch.empty_like(projection_at_infinity["t"]),
        "xy_min": torch.empty_like(projection_at_zero["xy"]),
        "xy_max": torch.empty_like(projection_at_infinity["xy"]),
        "overlaps_image": torch.empty_like(projection_at_zero["valid"]),
    }

    for min_valid, max_valid in itertools.product([True, False], [True, False]):
        min_mask = projection_at_zero["valid"] ^ (not min_valid)
        max_mask = projection_at_infinity["valid"] ^ (not max_valid)
        mask = min_mask & max_mask
        min_value = projection_at_zero if min_valid else frame_intersection_min
        max_value = projection_at_infinity if max_valid else frame_intersection_max
        result["t_min"][mask] = min_value["t"][mask]
        result["t_max"][mask] = max_value["t"][mask]
        result["xy_min"][mask] = min_value["xy"][mask]
        result["xy_max"][mask] = max_value["xy"][mask]
        result["overlaps_image"][mask] = (min_value["valid"] & max_value["valid"])[mask]

    return result


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def lift_to_3d(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3"]:
    """Calculate the 3D positions that correspond to the specified 2D points on the
    epipolar lines defined by the origins and directions. The extrinsics and intrinsics
    are for the images the 2D points lie on.
    """

    xy_origins, xy_directions = get_world_rays(xy, extrinsics, intrinsics)
    return intersect_rays(origins, directions, xy_origins, xy_directions)


def get_depth(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, " *batch"]:
    """Calculate the depths that correspond to the specified 2D points on the epipolar
    lines defined by the origins and directions. The extrinsics and intrinsics are for
    the images the 2D points lie on.
    """
    xyz = lift_to_3d(origins, directions, xy, extrinsics, intrinsics)
    return (xyz - origins).norm(dim=-1)
