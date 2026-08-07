# Warp-local sparse weighted contrast

输入是逻辑形状为 `[G, 32]` 的非结构化稀疏标量张量 `T`。稀疏值存放在
`T_values[NNZ]` 中，`slot_to_nnz[g, j]` 给出 Warp 组 `g` 的 lane `j` 对应的
稀疏值下标；`-1` 表示空槽。每个逻辑行映射到一个 CUDA Warp，不存在跨 Warp
的拓扑关联。

局部拓扑由可学习权重 `W[G, 32, 32]` 和布尔掩码
`edge_mask[G, 32, 32]` 给出。令空槽的值为零，则有效目标槽的前向为

```
S[g, i] = sum(j=0..31) edge_mask[g, i, j] * W[g, i, j] * T[g, j]
R[g, i] = tanh(S[g, i])
```

输出 `R_values[NNZ]` 与 `T_values` 使用相同的稀疏顺序。`W` 不自动归一化，
不要求对称；自环是否存在由 `edge_mask[g, i, i]` 决定。这里“对比”严格按上述
加权聚合公式解释，不额外引入 `T_j - T_i`。

输入与输出：

- `T_values: [NNZ]`, float32，需要梯度。
- `slot_to_nnz: [G, 32]`, int64，不需要梯度；每个 `[0, NNZ)` 下标恰好出现一次。
- `W: [G, 32, 32]`, float32，需要梯度。
- `edge_mask: [G, 32, 32]`, bool，不需要梯度。
- `R_values: [NNZ]`, float32。

反向对 `T_values` 和 `W` 求梯度。默认 `G=4096`，输入密度为 50%；随机非自环
拓扑密度为 25%，生成的测试输入强制保留自环。`G` 可由环境变量
`WSC_GROUPS` 覆盖。
