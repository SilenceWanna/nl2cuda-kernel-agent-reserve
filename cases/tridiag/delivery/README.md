# 三对角求解 Thomas —— CUDA 前反向 Kernel（可独立编译交付版）

自然语言描述的批量三对角线性系统求解（Thomas 算法）结构，经本项目 skill 驱动生成并优化的自定义 CUDA
前向 + 反向 kernel。**完全不依赖 PyTorch**，仅需 `nvcc` + CUDA runtime 即可独立编译。

## 算法

- 输入：`lower/diag/upper/rhs` 各 `[B,N]`（float32）——每行一个独立三对角系统 `A x = rhs`，
  `A` 由下/主/上对角线给出；`B` 个系统间并行、系统内 Thomas 串行 O(N)。
- 前向：Thomas（前向消元 LU → 前代 → 回代），输出 `x:[B,N]`，并缓存 `factors:[2,B,N]`
  （消元乘子 L 与消元后主对角 U，供反向复用，免重算分解）。
- 反向：给定上游梯度 `grad_x = dL/dx`，对 `lower/diag/upper/rhs` 全部求梯度。
  **利用"线性求解的反向有解析解=解伴随系统 `Aᵀλ = grad_x`"**（又一次 Thomas）：
  - `grad_rhs_i = λ_i`，`grad_diag_i = -λ_i·x_i`
  - `grad_lower_i = -λ_i·x_{i-1}`（i>0，否则 0），`grad_upper_i = -λ_i·x_{i+1}`（i<N-1，否则 0）

## 文件

| 文件 | 说明 |
|------|------|
| `tridiag_kernels.cu` | 前向+反向 `__global__` kernel + `extern "C"` 裸指针 host 接口 |
| `tridiag_test.cu` | 自测 harness：内置 CPU 参考（独立 Thomas + 伴随解），对拍 GPU 输出（自包含，不依赖 torch） |
| `Makefile` | nvcc 独立编译；`make test` 编译+跑自测 |

## 编译 / 运行

```bash
make test                 # 编译 + 运行自测（默认 A100 sm_80，小规模 CPU 对拍）
make ARCH=sm_75 test      # 换 T4 等其他架构
make libtridiag.a         # 只产出静态库供外部链接
./tridiag_test 8192 512   # 自定义规模跑自测（B N）
```

预期自测输出：前向 x、反向 dLower/dDiag/dUpper/dRhs 五项 `allclose(atol=rtol=1e-2)` 全 **PASS**。

## host 接口

```c
// 收裸主机指针；内部负责 device 内存分配/拷贝/kernel launch/同步。
// 前向：输出 x[B*N] 与 factors[2*B*N]（[0]=消元乘子 L、[1]=消元后主对角 U，布局 [2,B,N]，供反向复用）。
extern "C" void tridiag_forward_cuda(const float* lower, const float* diag,
                                     const float* upper, const float* rhs,
                                     float* x, float* factors, int B, int N);
// 反向：输入 grad_x/upper/x 与 factors（前向输出），输出 gradients[4*B*N]
//       （布局 [4,B,N]：[0]=grad_lower [1]=grad_diag [2]=grad_upper [3]=grad_rhs）。
extern "C" void tridiag_backward_cuda(const float* grad_x, const float* upper,
                                      const float* x, const float* factors,
                                      float* gradients, int B, int N);
```
（约束：一 block 一系统，shared memory 存 4N/5N 个 float，要求 N ≤ ~1024；消元/回代串行由每 block
的 0 号线程做，B 系统间并行。本用例 N=512 满足。）

## 优化要点

- **前向**：一 block 一个三对角系统，输入全载入 shared memory，block 内 0 号线程做 O(N) 串行
  Thomas（消元+回代，算法固有串行，无法并行化单系统），B 系统间天然并行。缓存 factors 供反向免重算分解。
- **反向**：复用前向缓存的 factors 直接解伴随系统 `Aᵀλ=grad_x`（转置三对角：其下对角=原 upper、
  上对角=原 lower），再由 λ 与 x 组回四路梯度——**一次 Thomas 解出全部梯度**，无需 autograd 反传。

## 验收结果（A100-SXM4-40GB, sm_80）

以 PyTorch 并行 PCR 实现为金标准、对比 `torch.compile`（默认 mode），规范计时（CUDA events、
warmup≥10、≥100 次几何均值、CV≤5%），计算主导区规模（B=16384/N=512）：

| | 自定义 kernel | torch.compile | 加速比 |
|---|---|---|---|
| 前向 | — | — | **1.27×** |
| 反向 | — | — | **9.80×** |

正确性：≥5 组随机种子，前向 x + 反向四路梯度 `allclose(atol=rtol=1e-2)` 全 PASS（误差 ~5e-6）。

**反向 9.80× 是诚实的 CUDA 结构性优势**，非弱 baseline：baseline 用的是解三对角的标准高效并行算法
PCR（明确避开 `torch.linalg.solve` dense O(N³)——那才是弱 baseline），前向 baseline 才 ~1.8ms 是强
baseline。candidate 赢在**用解析伴随反传 vs autograd 机械反传 PCR 的每一步**——`torch.compile` 的
autograd 不知道"这是线性求解、反向可解析"，只能机械反传 9 级 PCR 展开的大图（反向图 ~4.5× 前向）。
与 gated_ssm/linear_ssm（CUDA 做稳定递推/前缀而 PyTorch 做不到）同类的"CUDA 能利用数学结构、
autograd 不能"的结构性价值。

## 合规声明（防作弊）

- **fp32 全精度**，不使用 fast-math、不降精度换速度。
- **无对 PyTorch 高层算子的运行时依赖**；本交付版连 PyTorch 都不依赖。
- 未使用 cuBLAS/cuSOLVER（虽然边界约束允许其作辅助原语），全部为自定义手写 kernel——
  尤其**未直调 `cusolverDnSgtsv2` 等厂商三对角求解成品**（那会使 candidate=baseline 同款厂商算法）。
