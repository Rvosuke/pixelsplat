import torch
from einops import repeat
from jaxtyping import Int
from torch import Tensor

Index = Int[Tensor, "n n-1"]


def generate_heterogeneous_index(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate indices for all pairs except self-pairs."""
    arange = torch.arange(n, device=device)

    # Generate an index that represents the item itself.
    # 例如初始化时，n=3，index_self = tensor([[0, 0], [1, 1], [2, 2]])
    index_self = repeat(arange, "h -> h w", w=n - 1)

    # Generate an index that represents the other items.
    # 例如初始化时，n=3，index_other = tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    # 加上对角线的1，得到tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3]])
    # 去掉最后一列，得到tensor([[1, 2], [0, 2], [0, 1]])
    index_other = repeat(arange, "w -> h w", h=n).clone()
    index_other += torch.ones((n, n), device=device, dtype=torch.int64).triu()
    index_other = index_other[:, :-1]

    return index_self, index_other


def generate_heterogeneous_index_transpose(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate an index that can be used to "transpose" the heterogeneous index.
    Applying the index a second time inverts the "transpose."
    """
    arange = torch.arange(n, device=device)
    ones = torch.ones((n, n), device=device, dtype=torch.int64)

    index_self = repeat(arange, "w -> h w", h=n).clone()
    index_self = index_self + ones.triu()

    index_other = repeat(arange, "h -> h w", w=n)
    index_other = index_other - (1 - ones.triu())

    return index_self[:, :-1], index_other[:, :-1]
