# 二维前缀和 / 积分图（integral image）

## 算法定义

输入二维张量 `X[H, W]`，输出同形状 `S[H, W]`，其中每个位置是其左上矩形区域（含自身）的元素和：

```text
S[i, j] = Σ_{p=0..i, q=0..j} X[p, q]
```

即含边界的二维前缀和（目标检测 / 图像里的积分图，可 O(1) 求任意矩形区域和）。所有位置同步计算，不读本轮已写出的结果。

## Shape / dtype / gradient

- 输入：`X[H, W]`，float32，连续 CUDA 张量
- 输出：`S[H, W]`，float32
- 默认：`H = W = 2048`（verify 用；2D 前缀和 float32 累加误差随规模涨，4096 会擦破 atol，故默认 2048）；bench 用 `IMG_SIZE=4096` 进计算主导区。`IMG_SIZE` 同时设 H/W，`IMG_H`、`IMG_W` 可分别覆盖
- `grad_inputs = ["X"]`，无标量参数
- 反向：由 `S[i,j] = Σ_{p≤i,q≤j} X[p,q]` 得 `∂S[i,j]/∂X[p,q] = 1` 当 `p≤i 且 q≤j`，故
  ```text
  dX[p, q] = Σ_{i≥p, j≥q} dS[i, j]
  ```
  即上游梯度 `dS` 的**二维后缀和**（从右下向左上累加）。
