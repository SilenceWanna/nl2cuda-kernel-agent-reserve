#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__inline__ __device__ void reduce_stats(
        float sum,
        float sumsq,
        float inv_D,
        float eps,
        float* __restrict__ mean_out,
        float* __restrict__ inv_std_out,
        int row,
        float& mean,
        float& inv_std) {
    __shared__ float warp_sums[32];
    __shared__ float warp_sumsq[32];
    __shared__ float stats[2];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int warp_count = (BLOCK_SIZE + 31) / 32;

    sum = warp_sum(sum);
    sumsq = warp_sum(sumsq);
    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_sumsq[warp] = sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        float total = lane < warp_count ? warp_sums[lane] : 0.0f;
        float total_sq = lane < warp_count ? warp_sumsq[lane] : 0.0f;
        total = warp_sum(total);
        total_sq = warp_sum(total_sq);
        if (lane == 0) {
            const float m = total * inv_D;
            float variance = total_sq * inv_D - m * m;
            variance = variance > 0.0f ? variance : 0.0f;
            const float r = rsqrtf(variance + eps);
            stats[0] = m;
            stats[1] = r;
            mean_out[row] = m;
            inv_std_out[row] = r;
        }
    }
    __syncthreads();
    mean = stats[0];
    inv_std = stats[1];
}

template <int BLOCK_SIZE>
__global__ void layernorm_forward_scalar_kernel(
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ Y,
        float* __restrict__ mean,
        float* __restrict__ inv_std,
        int B,
        int D,
        float eps) {
    const int row = blockIdx.x;
    if (row >= B) {
        return;
    }
    const int base = row * D;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float value = X[base + d];
        sum += value;
        sumsq += value * value;
    }

    float m;
    float r;
    reduce_stats<BLOCK_SIZE>(sum, sumsq, 1.0f / static_cast<float>(D), eps,
                             mean, inv_std, row, m, r);
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float xhat = (X[base + d] - m) * r;
        Y[base + d] = xhat * gamma[d] + beta[d];
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_forward_1024_kernel(
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ Y,
        float* __restrict__ mean,
        float* __restrict__ inv_std,
        int B,
        float eps) {
    const int row = blockIdx.x;
    if (row >= B) {
        return;
    }
    const float4* X4 = reinterpret_cast<const float4*>(X + row * 1024);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* Y4 = reinterpret_cast<float4*>(Y + row * 1024);
    const int i = threadIdx.x;
    const float4 value = X4[i];
    const float sum = value.x + value.y + value.z + value.w;
    const float sumsq = value.x * value.x + value.y * value.y +
                        value.z * value.z + value.w * value.w;
    float m;
    float r;
    reduce_stats<BLOCK_SIZE>(sum, sumsq, 1.0f / 1024.0f, eps,
                             mean, inv_std, row, m, r);

    const float4 g = gamma4[i];
    const float4 b = beta4[i];
    float4 out;
    out.x = (value.x - m) * r * g.x + b.x;
    out.y = (value.y - m) * r * g.y + b.y;
    out.z = (value.z - m) * r * g.z + b.z;
    out.w = (value.w - m) * r * g.w + b.w;
    Y4[i] = out;
}

bool aligned_16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0x0f) == 0;
}

void check_inputs(const torch::Tensor& X, const torch::Tensor& gamma,
                  const torch::Tensor& beta) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "X, gamma, and beta must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    beta.scalar_type() == torch::kFloat32,
                "X, gamma, and beta must be float32");
    TORCH_CHECK(X.dim() == 2 && gamma.dim() == 1 && beta.dim() == 1,
                "expected X[B,D], gamma[D], beta[D]");
    TORCH_CHECK(X.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1),
                "gamma and beta must have length D");
}

}  // namespace

std::vector<torch::Tensor> layernorm_forward(
        torch::Tensor X,
        torch::Tensor gamma,
        torch::Tensor beta,
        double eps) {
    check_inputs(X, gamma, beta);
    const int B = static_cast<int>(X.size(0));
    const int D = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    auto stats_options = X.options().dtype(torch::kFloat32);
    auto mean = torch::empty({B}, stats_options);
    auto inv_std = torch::empty({B}, stats_options);
    if (B == 0) {
        return {Y, mean, inv_std};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool use_1024 = D == 1024 && aligned_16(X.data_ptr<float>()) &&
                          aligned_16(gamma.data_ptr<float>()) &&
                          aligned_16(beta.data_ptr<float>()) &&
                          aligned_16(Y.data_ptr<float>());
    if (use_1024) {
        layernorm_forward_1024_kernel<kBlockSize><<<B, kBlockSize, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            B, static_cast<float>(eps));
    } else {
        layernorm_forward_scalar_kernel<kBlockSize><<<B, kBlockSize, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            B, D, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {Y, mean, inv_std};
}
