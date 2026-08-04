// Gather-style backward kernel. Each input reads the at-most-nine outputs it affects.

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

__global__ void dynamic_grid_evolution_backward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int height,
    int width,
    float k) {
    __shared__ float output_tile[kSharedY][kSharedX];
    __shared__ float grad_tile[kSharedY][kSharedX];

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
        float output_value = 0.0f;
        float grad_value = 0.0f;
        if (global_x >= 0 && global_x < width && global_y >= 0 &&
            global_y < height) {
            const size_t global_index =
                static_cast<size_t>(global_y) * width + global_x;
            output_value = output[global_index];
            grad_value = grad_output[global_index];
        }
        output_tile[shared_y][shared_x] = output_value;
        grad_tile[shared_y][shared_x] = grad_value;
    }
    __syncthreads();

    const int x = origin_x + local_x;
    const int y = origin_y + local_y;
    if (x >= width || y >= height) {
        return;
    }

    float output_adjoint_sum = 0.0f;
#pragma unroll
    for (int offset_y = 0; offset_y < 3; ++offset_y) {
#pragma unroll
        for (int offset_x = 0; offset_x < 3; ++offset_x) {
            const float evolved =
                output_tile[local_y + offset_y][local_x + offset_x];
            const float upstream =
                grad_tile[local_y + offset_y][local_x + offset_x];
            output_adjoint_sum += upstream * evolved * (1.0f - evolved);
        }
    }

    const size_t input_index = static_cast<size_t>(y) * width + x;
    const float value = input[input_index];
    const float gate = sigmoid(k * value);
    const float local_derivative = gate + k * value * gate * (1.0f - gate);
    grad_input[input_index] = local_derivative * output_adjoint_sum;
}

}  // namespace

torch::Tensor dynamic_grid_evolution_backward(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor grad_output,
    double k) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && grad_output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                    output.scalar_type() == torch::kFloat32 &&
                    grad_output.scalar_type() == torch::kFloat32,
                "all tensors must be float32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [H, W]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
    TORCH_CHECK(grad_output.sizes() == input.sizes(),
                "grad_output shape must match input");

    c10::cuda::CUDAGuard device_guard(input.device());
    TORCH_CHECK(output.device() == input.device() &&
                    grad_output.device() == input.device(),
                "all tensors must be on the same CUDA device");
    input = input.contiguous();
    output = output.contiguous();
    grad_output = grad_output.contiguous();

    const int height = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    auto grad_input = torch::empty_like(input);

    const dim3 threads(kBlockX, kBlockY);
    const dim3 blocks((width + kBlockX - 1) / kBlockX,
                      (height + kBlockY - 1) / kBlockY);
    const auto stream = at::cuda::getCurrentCUDAStream();
    dynamic_grid_evolution_backward_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        height,
        width,
        static_cast<float>(k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dynamic_grid_evolution_backward",
        &dynamic_grid_evolution_backward,
        "Dynamic grid evolution backward (CUDA)");
}
