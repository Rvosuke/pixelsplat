## _intersect_image_coordinate

### 推导`coordinate_other`的计算

#### 1. 问题描述

- **已知：**
  - 相机内参矩阵`K`，包含焦距`f_x`、`f_y`和主点坐标`c_x`、`c_y`。
  - 射线的起点`o = (o_x, o_y, o_z)`和方向`d = (d_x, d_y, d_z)`。
  - 固定的图像坐标维度（`x`或`y`）及其值`coordinate_value`。
- **目标：**
  - 计算射线参数`t`，使得射线投影在固定维度的图像坐标上等于`coordinate_value`。
  - 计算另一个维度的图像坐标`coordinate_other`。

#### 2. 固定维度的投影关系

以固定`x`维度为例，图像坐标与相机坐标的关系为：

$$
u = f_x \cdot \frac{X}{Z} + c_x
$$

其中：

- $(X, Y, Z)$是相机坐标系下的点。
- $u$是图像坐标系下的横坐标。

我们定义：

$$
c = \frac{u - c_x}{f_x} = \frac{X}{Z}
$$

射线方程：

$$
X = o_x + t \cdot d_x \\
Y = o_y + t \cdot d_y \\
Z = o_z + t \cdot d_z
$$

代入得到：

$$
c = \frac{o_x + t \cdot d_x}{o_z + t \cdot d_z}
$$

#### 3. 解关于`t`的方程

将上式整理：

$$
c \cdot (o_z + t \cdot d_z) = o_x + t \cdot d_x \\
c \cdot o_z + c \cdot t \cdot d_z = o_x + t \cdot d_x \\
(c \cdot d_z - d_x) \cdot t = o_x - c \cdot o_z \\
t = \frac{o_x - c \cdot o_z}{c \cdot d_z - d_x}
$$

这与代码中的计算对应：

```python
t_numerator = c * oz - os
t_denominator = ds - c * dz
t = t_numerator / t_denominator
```

#### 4. 计算另一个坐标值`coordinate_other`

我们需要计算另一个维度（以`y`为例）的图像坐标：

$$
v = f_y \cdot \frac{Y}{Z} + c_y
$$

其中：

$$
Y = o_y + t \cdot d_y \\
Z = o_z + t \cdot d_z
$$

因此：

$$
v = f_y \cdot \frac{o_y + t \cdot d_y}{o_z + t \cdot d_z} + c_y
$$

我们的目标是找到一个表达式，避免直接除以`(o_z + t \cdot d_z)`，以防止除零错误。

##### **重写分子和分母**

首先，我们定义：

$$
N = o_y + t \cdot d_y \\
D = o_z + t \cdot d_z
$$

因此：

$$
v = f_y \cdot \frac{N}{D} + c_y
$$

将`t`的表达式代入`N`和`D`：

$$
N = o_y + \left( \frac{o_x - c \cdot o_z}{c \cdot d_z - d_x} \right) \cdot d_y \\
D = o_z + \left( \frac{o_x - c \cdot o_z}{c \cdot d_z - d_x} \right) \cdot d_z
$$

为简化计算，我们将`N`和`D`统一写成：

$$
N = \frac{ (o_y \cdot (c \cdot d_z - d_x) + d_y \cdot (o_x - c \cdot o_z) ) }{ c \cdot d_z - d_x } \\
D = \frac{ (o_z \cdot (c \cdot d_z - d_x) + d_z \cdot (o_x - c \cdot o_z) ) }{ c \cdot d_z - d_x }
$$

因此，分子和分母都有相同的分母，可以抵消：

$$
\frac{N}{D} = \frac{ (o_y \cdot (c \cdot d_z - d_x) + d_y \cdot (o_x - c \cdot o_z) ) }{ (o_z \cdot (c \cdot d_z - d_x) + d_z \cdot (o_x - c \cdot o_z) ) }
$$

将分子和分母重新整理：

$$
\begin{align*}
N &= o_y (c d_z - d_x) + d_y (o_x - c o_z) \\
D &= o_z (c d_z - d_x) + d_z (o_x - c o_z)
\end{align*}
$$

为方便对应代码中的变量，我们可以做如下替换：

- $ o_y = o_o $
- $ d_y = d_o $
- $ o_x = o_s $
- $ d_x = d_s $
- $ o_z = o_z $
- $ d_z = d_z $
- $ c $ 已定义

#### 5. 对应代码中的计算

代码中：

```python
coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
coordinate_denominator = dz * os - ds * oz
coordinate_other = co + coordinate_numerator / coordinate_denominator
```

**分子对应：**

$$
\text{coordinate\_numerator} = f_o \left[ o_o (c d_z - d_s) + d_o (o_s - c o_z) \right]
$$

这与我们推导的分子一致。

**分母对应：**

需要注意的是，代码中的分母为：

$$
\text{coordinate\_denominator} = d_z o_s - d_s o_z
$$

而我们在推导中得到的分母是：

$$
D_{\text{numerator}} = o_z (c d_z - d_x) + d_z (o_x - c o_z)
$$

通过代数变换，我们可以发现：

$$
D_{\text{numerator}} = (o_z c d_z - o_z d_x + d_z o_x - d_z c o_z) \\
= c o_z d_z - o_z d_x + d_z o_x - c o_z d_z \\
= - o_z d_x + d_z o_x
$$

简化后：

$$
D_{\text{numerator}} = o_x d_z - o_z d_x
$$

这与代码中的`coordinate_denominator`对应，只是符号相反。

**因此，代码中的分母实际上是：**

$$
\text{coordinate\_denominator} = d_z o_s - d_s o_z = o_s d_z - o_z d_s
$$

与我们推导的分母一致，说明代码和公式是对应的。

#### 6. 确认符号和一致性

注意在推导过程中，符号的正负可能会有所变化，这取决于代数整理的方式。但最终结果应与代码中的实现一致。关键是确认变量之间的对应关系。

#### 7. 关于坐标值是否归一化

在代码中，计算出的`coordinate_other`是通过相机内参（焦距`f_o`和主点坐标`c_o`）计算的，得到的值是在像素坐标系下的坐标值。

例如：

```python
coordinate_other = co + coordinate_numerator / coordinate_denominator
```

其中，`co`是主点坐标，通常以像素为单位。因此，`coordinate_other`也是以像素为单位的坐标值。

然而，在后续的代码中，函数`_is_in_bounds(xy)`检查坐标是否在`0`到`1`之间：

```python
def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)
```

这意味着有效的`xy`应该是归一化后的图像坐标，即在`0`到`1`之间。

**因此，计算出的`coordinate_other`是任意实数，若在0到1之间表示有效，其余都为无效，无穷表示没有交点。**