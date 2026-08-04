# 动态权重二维网格非线性演化

## 算法定义

输入二维能量场 `E[H,W]`。每个输出位置读取旧状态中以自身为中心、包含中心点的
`3x3` 邻域；越界位置按零处理。每个邻域元素根据自身数值实时生成动态权重，所有
位置同步更新，不读取本轮已经写出的结果。

动态权重和外层激活均采用 logistic sigmoid：

```text
w(x) = sigmoid(k * x)
S[i,j] = sum_{di=-1..1, dj=-1..1}
           w(E_tilde[i+di,j+dj]) * E_tilde[i+di,j+dj]
E_prime[i,j] = sigmoid(S[i,j])
```

其中 `E_tilde` 是 `E` 的零填充扩展，标量参数 `k=1.0`，不对 `k` 求梯度。

## Shape / dtype / gradient

- 输入：`E[H,W]`，float32，连续 CUDA 张量
- 输出：`E_prime[H,W]`，float32
- 默认：`H=W=2048`
- 环境覆盖：`DGE_SIZE` 同时设置 H/W，`DGE_H`、`DGE_W` 可分别覆盖
- 标量参数：`k=1.0`，可用 `DGE_K` 覆盖
- `grad_inputs=["E"]`，仅对能量场求梯度

CUDA candidate 使用共享内存 tile 复用邻域数据。反向按输入格点 gather 它影响的
最多九个输出，避免全局原子加。
