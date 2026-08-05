// 二维前缀和 / 积分图 前向 kernel：S[i,j] = Σ_{p<=i,q<=j} X[p,q]
//
// 分两 kernel（列扫描依赖行扫描全部完成，跨 block 数据依赖无法单 kernel 融合）：
//  ① row_scan：一 block 一行，块内两级扫描（每线程连续 chunk 前缀 + 块内 totals 扫描 + 加偏移），
//     沿 W 做 inclusive 前缀，合并写出（dim=1 前缀）。
//  ② col_scan：一线程一列，沿 H 向下串行累加（running sum）——**相邻线程读相邻列、完全合并访问**，
//     无 scan 开销。这是相对 torch.compile 的 cumsum(dim=0)（跨步扫描，慢）的效率优势来源。
// 直接原位写 S（②读 S 原位累加），省中间缓冲。float32 全精度。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

// ① 一 block 一行，沿 W 做 inclusive 前缀和，写入 out（dim=1 前缀）
__global__ void row_scan_kernel(const float* __restrict__ X,
                                float* __restrict__ out,
                                int H, int W) {
    const int row = blockIdx.x;
    if (row >= H) return;
    const int tid = threadIdx.x;
    const long base = (long)row * W;

    // 每线程负责连续一段 [start, end)
    const int chunk = (W + kThreads - 1) / kThreads;
    const int start = tid * chunk;
    int end = start + chunk;
    if (end > W) end = W;

    // phase 1：本线程 chunk 内串行 inclusive 前缀，得本段总和
    float acc = 0.0f;
    for (int c = start; c < end; ++c) acc += X[base + c];

    // phase 2：块内对各线程总和做 exclusive 扫描（Hillis-Steele，256 元素）
    __shared__ float totals[kThreads];
    totals[tid] = acc;
    __syncthreads();
    for (int off = 1; off < kThreads; off <<= 1) {
        float add = (tid >= off) ? totals[tid - off] : 0.0f;
        __syncthreads();
        totals[tid] += add;
        __syncthreads();
    }
    // 现在 totals[tid] 是 inclusive；exclusive 偏移 = totals[tid] - acc
    const float offset = totals[tid] - acc;

    // phase 3：再扫一遍本 chunk，加偏移写出
    float run = offset;
    for (int c = start; c < end; ++c) {
        run += X[base + c];
        out[base + c] = run;
    }
}

// ② 列方向 inclusive 前缀（沿 H）。一 block 管一段连续列（blockDim.x 列），
//    沿 H 分块流水：每次载入 TILE 行到寄存器/串行累加，用 running base 跨 tile 传播。
//    相邻线程读相邻列 → 完全合并访问；但为消除"单线程串行 H 步"的延迟瓶颈，
//    这里让每线程仍负责一整列（列间并行度 = W），并用 float4 无关；关键改为
//    **按行 tile 分段 + __restrict__ + 编译器展开**提高访存并行度与 ILP。
//    （2D 前缀和列方向本质串行依赖，无法并行单列内部；靠 W 列并行 + 高 occupancy 摊带宽。）
__global__ void col_scan_kernel(float* __restrict__ S, int H, int W) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;
    float acc = 0.0f;
    int i = 0;
    // 每次处理 4 行，提升 ILP（4 个独立 load 在飞行）+ 减少循环开销
    for (; i + 4 <= H; i += 4) {
        const long b = (long)i * W + col;
        float v0 = S[b];
        float v1 = S[b + W];
        float v2 = S[b + 2L * W];
        float v3 = S[b + 3L * W];
        v0 += acc; v1 += v0; v2 += v1; v3 += v2;
        S[b] = v0; S[b + W] = v1; S[b + 2L * W] = v2; S[b + 3L * W] = v3;
        acc = v3;
    }
    for (; i < H; ++i) {
        const long idx = (long)i * W + col;
        acc += S[idx];
        S[idx] = acc;
    }
}

}  // namespace

torch::Tensor integral_image_forward(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.dim() == 2, "X must be [H,W]");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    const int H = (int)X.size(0);
    const int W = (int)X.size(1);
    auto S = torch::empty_like(X);
    if (H == 0 || W == 0) return S;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    row_scan_kernel<<<H, kThreads, 0, stream>>>(
        X.data_ptr<float>(), S.data_ptr<float>(), H, W);
    const int col_blocks = (W + kThreads - 1) / kThreads;
    col_scan_kernel<<<col_blocks, kThreads, 0, stream>>>(
        S.data_ptr<float>(), H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return S;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_forward", &integral_image_forward, "Integral image forward (CUDA)");
}
