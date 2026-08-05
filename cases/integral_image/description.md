# Integral image / 2D prefix sum

Input `X[H, W]` is a contiguous CUDA tensor of dtype `float32`. The output
`S[H, W]` has the same shape and dtype, with

`S[i, j] = sum(X[p, q] for p <= i and q <= j)`.

The backward pass is required for `X`. For an upstream tensor `G`, the
gradient is the lower-right suffix sum:

`dX[p, q] = sum(G[i, j] for i >= p and j >= q)`.

The default benchmark uses `H = W = 4096`; `II_SIZE`, `II_H`, and `II_W`
can override the dimensions.
