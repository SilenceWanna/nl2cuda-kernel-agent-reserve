#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> layernorm_forward(
        torch::Tensor X,
        torch::Tensor gamma,
        torch::Tensor beta,
        double eps);

namespace {

constexpr int kBlockSize = 256;
constexpr int kRowsPerBlock = 32;

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__inline__ __device__ void reduce_pair(float first, float second,
                                       float& total_first, float& total_second) {
    __shared__ float warp_first[32];
    __shared__ float warp_second[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int warp_count = (BLOCK_SIZE + 31) / 32;

    first = warp_sum(first);
    second = warp_sum(second);
    if (lane == 0) {
        warp_first[warp] = first;
        warp_second[warp] = second;
    }
    __syncthreads();

    if (warp == 0) {
        float a = lane < warp_count ? warp_first[lane] : 0.0f;
        float b = lane < warp_count ? warp_second[lane] : 0.0f;
        a = warp_sum(a);
        b = warp_sum(b);
        if (lane == 0) {
            warp_first[0] = a;
            warp_second[0] = b;
        }
    }
    __syncthreads();
    total_first = warp_first[0];
    total_second = warp_second[0];
}

__global__ void layernorm_backward_fused_1024_kernel(
        const float* __restrict__ grad_Y,
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ mean,
        const float* __restrict__ inv_std,
        float* __restrict__ grad_X,
        float* __restrict__ grad_gamma,
        float* __restrict__ grad_beta,
        int B) {
    const int row_start = blockIdx.x * kRowsPerBlock;
    const int i = threadIdx.x;
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4 gamma_value = gamma4[i];
    float4 gamma_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 beta_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int r = 0; r < kRowsPerBlock; ++r) {
        const int row = row_start + r;
        float first = 0.0f;
        float second = 0.0f;
        float4 x_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 gy_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < B) {
            const float4* x4 = reinterpret_cast<const float4*>(X + row * 1024);
            const float4* gy4 = reinterpret_cast<const float4*>(grad_Y + row * 1024);
            x_value = x4[i];
            gy_value = gy4[i];
            const float m = mean[row];
            const float inv = inv_std[row];
            const float x0 = (x_value.x - m) * inv;
            const float x1 = (x_value.y - m) * inv;
            const float x2 = (x_value.z - m) * inv;
            const float x3 = (x_value.w - m) * inv;
            first = gy_value.x * gamma_value.x + gy_value.y * gamma_value.y +
                    gy_value.z * gamma_value.z + gy_value.w * gamma_value.w;
            second = gy_value.x * gamma_value.x * x0 +
                     gy_value.y * gamma_value.y * x1 +
                     gy_value.z * gamma_value.z * x2 +
                     gy_value.w * gamma_value.w * x3;
        }

        float total_first;
        float total_second;
        reduce_pair<kBlockSize>(first, second, total_first, total_second);

        if (row < B) {
            const float inv_D = 1.0f / 1024.0f;
            const float mean_first = total_first * inv_D;
            const float mean_second = total_second * inv_D;
            const float m = mean[row];
            const float inv = inv_std[row];
            const float x0 = (x_value.x - m) * inv;
            const float x1 = (x_value.y - m) * inv;
            const float x2 = (x_value.z - m) * inv;
            const float x3 = (x_value.w - m) * inv;
            const float p0 = gy_value.x * gamma_value.x;
            const float p1 = gy_value.y * gamma_value.y;
            const float p2 = gy_value.z * gamma_value.z;
            const float p3 = gy_value.w * gamma_value.w;
            float4 dx;
            dx.x = inv * (p0 - mean_first - x0 * mean_second);
            dx.y = inv * (p1 - mean_first - x1 * mean_second);
            dx.z = inv * (p2 - mean_first - x2 * mean_second);
            dx.w = inv * (p3 - mean_first - x3 * mean_second);
            float4* dx4 = reinterpret_cast<float4*>(grad_X + row * 1024);
            dx4[i] = dx;

            gamma_acc.x += gy_value.x * x0;
            gamma_acc.y += gy_value.y * x1;
            gamma_acc.z += gy_value.z * x2;
            gamma_acc.w += gy_value.w * x3;
            beta_acc.x += gy_value.x;
            beta_acc.y += gy_value.y;
            beta_acc.z += gy_value.z;
            beta_acc.w += gy_value.w;
        }
    }

    float4* dg4 = reinterpret_cast<float4*>(grad_gamma);
    float4* db4 = reinterpret_cast<float4*>(grad_beta);
    atomicAdd(&dg4[i].x, gamma_acc.x);
    atomicAdd(&dg4[i].y, gamma_acc.y);
    atomicAdd(&dg4[i].z, gamma_acc.z);
    atomicAdd(&dg4[i].w, gamma_acc.w);
    atomicAdd(&db4[i].x, beta_acc.x);
    atomicAdd(&db4[i].y, beta_acc.y);
    atomicAdd(&db4[i].z, beta_acc.z);
    atomicAdd(&db4[i].w, beta_acc.w);
}

template <int BLOCK_SIZE>
__global__ void layernorm_backward_x_kernel(
        const float* __restrict__ grad_Y,
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ mean,
        const float* __restrict__ inv_std,
        float* __restrict__ grad_X,
        int B,
        int D) {
    const int row = blockIdx.x;
    if (row >= B) {
        return;
    }
    const int base = row * D;
    const float m = mean[row];
    const float inv = inv_std[row];
    float sum_first = 0.0f;
    float sum_second = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float xhat = (X[base + d] - m) * inv;
        const float p = grad_Y[base + d] * gamma[d];
        sum_first += p;
        sum_second += p * xhat;
    }
    float total_first;
    float total_second;
    reduce_pair<BLOCK_SIZE>(sum_first, sum_second, total_first, total_second);
    const float mean_first = total_first / static_cast<float>(D);
    const float mean_second = total_second / static_cast<float>(D);
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float xhat = (X[base + d] - m) * inv;
        const float p = grad_Y[base + d] * gamma[d];
        grad_X[base + d] = inv * (p - mean_first - xhat * mean_second);
    }
}

constexpr int kParamRows = 32;

__global__ void layernorm_backward_param_kernel(
        const float* __restrict__ grad_Y,
        const float* __restrict__ X,
        const float* __restrict__ mean,
        const float* __restrict__ inv_std,
        float* __restrict__ grad_gamma,
        float* __restrict__ grad_beta,
        int B,
        int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = blockIdx.y * kParamRows;
    if (d >= D) {
        return;
    }
    float dg = 0.0f;
    float db = 0.0f;
    for (int r = 0; r < kParamRows; ++r) {
        const int row = row_start + r;
        if (row < B) {
            const int index = row * D + d;
            const float xhat = (X[index] - mean[row]) * inv_std[row];
            dg += grad_Y[index] * xhat;
            db += grad_Y[index];
        }
    }
    atomicAdd(grad_gamma + d, dg);
    atomicAdd(grad_beta + d, db);
}

void check_inputs(const torch::Tensor& grad_Y, const torch::Tensor& X,
                  const torch::Tensor& gamma, const torch::Tensor& mean,
                  const torch::Tensor& inv_std) {
    TORCH_CHECK(grad_Y.is_cuda() && X.is_cuda() && gamma.is_cuda() &&
                    mean.is_cuda() && inv_std.is_cuda(),
                "backward inputs must be CUDA tensors");
    TORCH_CHECK(grad_Y.scalar_type() == torch::kFloat32 &&
                    X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    mean.scalar_type() == torch::kFloat32 &&
                    inv_std.scalar_type() == torch::kFloat32,
                "backward inputs must be float32");
    TORCH_CHECK(grad_Y.dim() == 2 && X.dim() == 2 && gamma.dim() == 1 &&
                    mean.dim() == 1 && inv_std.dim() == 1,
                "invalid backward tensor ranks");
    TORCH_CHECK(grad_Y.is_contiguous() && X.is_contiguous() &&
                    gamma.is_contiguous() && mean.is_contiguous() &&
                    inv_std.is_contiguous(),
                "backward inputs must be contiguous");
    TORCH_CHECK(grad_Y.sizes() == X.sizes(), "grad_Y and X must have the same shape");
    TORCH_CHECK(gamma.size(0) == X.size(1) && mean.size(0) == X.size(0) &&
                    inv_std.size(0) == X.size(0),
                "backward shape mismatch");
}

}  // namespace

std::vector<torch::Tensor> layernorm_backward(
        torch::Tensor grad_Y,
        torch::Tensor X,
        torch::Tensor gamma,
        torch::Tensor mean,
        torch::Tensor inv_std) {
    check_inputs(grad_Y, X, gamma, mean, inv_std);
    const int B = static_cast<int>(X.size(0));
    const int D = static_cast<int>(X.size(1));
    auto grad_X = torch::empty_like(X);
    auto grad_gamma = torch::zeros_like(gamma);
    auto grad_beta = torch::zeros_like(gamma);
    if (B == 0) {
        return {grad_X, grad_gamma, grad_beta};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == 1024) {
        const int blocks = (B + kRowsPerBlock - 1) / kRowsPerBlock;
        layernorm_backward_fused_1024_kernel<<<blocks, kBlockSize, 0, stream>>>(
            grad_Y.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(), grad_X.data_ptr<float>(),
            grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(), B);
    } else {
        layernorm_backward_x_kernel<kBlockSize><<<B, kBlockSize, 0, stream>>>(
            grad_Y.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(), grad_X.data_ptr<float>(),
            B, D);
        const dim3 block(kBlockSize);
        const dim3 grid((D + kBlockSize - 1) / kBlockSize,
                        (B + kParamRows - 1) / kParamRows);
        layernorm_backward_param_kernel<<<grid, block, 0, stream>>>(
            grad_Y.data_ptr<float>(), X.data_ptr<float>(), mean.data_ptr<float>(),
            inv_std.data_ptr<float>(), grad_gamma.data_ptr<float>(),
            grad_beta.data_ptr<float>(), B, D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_X, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("layernorm_backward", &layernorm_backward, "LayerNorm backward (CUDA)");
}
