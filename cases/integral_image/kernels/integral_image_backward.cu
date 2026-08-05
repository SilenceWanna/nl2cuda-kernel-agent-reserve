// 二维前缀和 反向 kernel：dX[p,q] = Σ_{i>=p, j>=q} dS[i,j]，即上游梯度 dS 的 2D 后缀和。
//
// 与前向镜像（前向是"左上前缀和"，反向是"右下后缀和"）：
//  ① row_suffix：一 block 一行，沿 W 做 inclusive **后缀**和（从右往左），两级扫描同前向。
//  ② col_suffix：一线程一列，沿 H **从下往上**串行累加（原位）——合并访问、无 scan 开销。
// float32 全精度。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

// ① 一 block 一行，沿 W 做 inclusive 后缀和（从右往左），写入 out
__global__ void row_suffix_kernel(const float* __restrict__ dS,
                                  float* __restrict__ out,
                                  int H, int W) {
    const int row = blockIdx.x;
    if (row >= H) return;
    const int tid = threadIdx.x;
    const long base = (long)row * W;

    // 每线程负责连续一段 [start, end)（tid 越大越靠右）
    const int chunk = (W + kThreads - 1) / kThreads;
    const int start = tid * chunk;
    int end = start + chunk;
    if (end > W) end = W;

    // phase 1：本 chunk 内总和
    float acc = 0.0f;
    for (int c = start; c < end; ++c) acc += dS[base + c];

    // phase 2：块内做"后缀" exclusive 扫描——tid 的右侧所有线程总和之和
    __shared__ float totals[kThreads];
    totals[tid] = acc;
    __syncthreads();
    // inclusive 后缀扫描（右向左 Hillis-Steele）
    for (int off = 1; off < kThreads; off <<= 1) {
        float add = (tid + off < kThreads) ? totals[tid + off] : 0.0f;
        __syncthreads();
        totals[tid] += add;
        __syncthreads();
    }
    // totals[tid] = 从 tid 到末尾的 inclusive 后缀；exclusive（右侧）= totals[tid] - acc
    const float right_offset = totals[tid] - acc;

    // phase 3：本 chunk 内从右往左 inclusive 后缀 + 右偏移
    float run = right_offset;
    for (int c = end - 1; c >= start; --c) {
        run += dS[base + c];
        out[base + c] = run;
    }
}

// ② 一线程一列，沿 H 从下往上串行累加（原位）
__global__ void col_suffix_kernel(float* __restrict__ dX, int H, int W) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;
    float acc = 0.0f;
    for (int i = H - 1; i >= 0; --i) {
        const long idx = (long)i * W + col;
        acc += dX[idx];
        dX[idx] = acc;
    }
}

}  // namespace

torch::Tensor integral_image_backward(torch::Tensor dS) {
    TORCH_CHECK(dS.is_cuda(), "dS must be CUDA");
    TORCH_CHECK(dS.scalar_type() == torch::kFloat32, "dS must be float32");
    TORCH_CHECK(dS.dim() == 2, "dS must be [H,W]");
    TORCH_CHECK(dS.is_contiguous(), "dS must be contiguous");
    const int H = (int)dS.size(0);
    const int W = (int)dS.size(1);
    auto dX = torch::empty_like(dS);
    if (H == 0 || W == 0) return dX;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    row_suffix_kernel<<<H, kThreads, 0, stream>>>(
        dS.data_ptr<float>(), dX.data_ptr<float>(), H, W);
    const int col_blocks = (W + kThreads - 1) / kThreads;
    col_suffix_kernel<<<col_blocks, kThreads, 0, stream>>>(
        dX.data_ptr<float>(), H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_backward", &integral_image_backward, "Integral image backward (CUDA)");
}
