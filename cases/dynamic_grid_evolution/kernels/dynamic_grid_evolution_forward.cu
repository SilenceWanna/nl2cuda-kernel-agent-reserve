// Shared-memory tiled forward kernel for the confirmed 3x3 grid evolution.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kSharedX = kBlockX + 2;
constexpr int kSharedY = kBlockY + 2;

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void dynamic_grid_evolution_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    float k) {
    __shared__ float tile[kSharedY][kSharedX];

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int thread_id = local_y * blockDim.x + local_x;
    const int thread_count = blockDim.x * blockDim.y;
    const int origin_x = blockIdx.x * kBlockX;
    const int origin_y = blockIdx.y * kBlockY;

    for (int index = thread_id; index < kSharedX * kSharedY;
         index += thread_count) {
        const int shared_y = index / kSharedX;
        const int shared_x = index - shared_y * kSharedX;
        const int global_x = origin_x + shared_x - 1;
        const int global_y = origin_y + shared_y - 1;
        float value = 0.0f;
        if (global_x >= 0 && global_x < width && global_y >= 0 &&
            global_y < height) {
            value = input[static_cast<size_t>(global_y) * width + global_x];
        }
        tile[shared_y][shared_x] = value;
    }
    __syncthreads();

    const int x = origin_x + local_x;
    const int y = origin_y + local_y;
    if (x >= width || y >= height) {
        return;
    }

    float weighted_sum = 0.0f;
#pragma unroll
    for (int offset_y = 0; offset_y < 3; ++offset_y) {
#pragma unroll
        for (int offset_x = 0; offset_x < 3; ++offset_x) {
            const float value = tile[local_y + offset_y][local_x + offset_x];
            weighted_sum += sigmoid(k * value) * value;
        }
    }
    output[static_cast<size_t>(y) * width + x] = sigmoid(weighted_sum);
}

}  // namespace

torch::Tensor dynamic_grid_evolution_forward(torch::Tensor input, double k) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [H, W]");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0, "input dimensions must be positive");

    c10::cuda::CUDAGuard device_guard(input.device());
    input = input.contiguous();
    const int height = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    auto output = torch::empty_like(input);

    const dim3 threads(kBlockX, kBlockY);
    const dim3 blocks((width + kBlockX - 1) / kBlockX,
                      (height + kBlockY - 1) / kBlockY);
    const auto stream = at::cuda::getCurrentCUDAStream();
    dynamic_grid_evolution_forward_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width,
        static_cast<float>(k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dynamic_grid_evolution_forward",
        &dynamic_grid_evolution_forward,
        "Dynamic grid evolution forward (CUDA)");
}
