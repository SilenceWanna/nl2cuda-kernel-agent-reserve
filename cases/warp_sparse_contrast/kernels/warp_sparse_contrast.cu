// Warp-local sparse weighted contrast, forward and backward.
//
// One hardware warp owns one logical [32] sparse group. Forward evaluates each
// active weight row as a coalesced dot product and reduces it with shuffle
// instructions. Backward broadcasts each output adjoint with shuffle and emits
// dT_values and dW in one fused traversal of W.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}

__global__ void warp_sparse_forward_kernel(
        const float* __restrict__ t_values,
        const int64_t* __restrict__ slot_to_nnz,
        const float* __restrict__ weights,
        const bool* __restrict__ edge_mask,
        float* __restrict__ r_values,
        int groups) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int group = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (group >= groups) {
        return;
    }

    const size_t slot_base = static_cast<size_t>(group) * kWarpSize;
    const int value_index = static_cast<int>(slot_to_nnz[slot_base + lane]);
    const bool source_valid = value_index >= 0;
    const float source_value = source_valid ? t_values[value_index] : 0.0f;
    const size_t weight_base = static_cast<size_t>(group) * kWarpSize * kWarpSize;

    #pragma unroll
    for (int target_lane = 0; target_lane < kWarpSize; ++target_lane) {
        const int output_index = __shfl_sync(kFullMask, value_index, target_lane);
        if (output_index < 0) {
            continue;
        }

        const size_t edge = weight_base + target_lane * kWarpSize + lane;
        float contribution = 0.0f;
        if (source_valid && edge_mask[edge]) {
            contribution = weights[edge] * source_value;
        }
        const float sum = warp_sum(contribution);
        if (lane == 0) {
            r_values[output_index] = tanhf(sum);
        }
    }
}

__global__ void warp_sparse_backward_kernel(
        const float* __restrict__ t_values,
        const int64_t* __restrict__ slot_to_nnz,
        const float* __restrict__ weights,
        const bool* __restrict__ edge_mask,
        const float* __restrict__ r_values,
        const float* __restrict__ grad_r_values,
        float* __restrict__ grad_t_values,
        float* __restrict__ grad_weights,
        int groups) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int group = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (group >= groups) {
        return;
    }

    const size_t slot_base = static_cast<size_t>(group) * kWarpSize;
    const int value_index = static_cast<int>(slot_to_nnz[slot_base + lane]);
    const bool source_valid = value_index >= 0;
    const float source_value = source_valid ? t_values[value_index] : 0.0f;

    float output_adjoint = 0.0f;
    if (source_valid) {
        const float r = r_values[value_index];
        output_adjoint = grad_r_values[value_index] * (1.0f - r * r);
    }

    const size_t weight_base = static_cast<size_t>(group) * kWarpSize * kWarpSize;
    float grad_source = 0.0f;

    #pragma unroll
    for (int target_lane = 0; target_lane < kWarpSize; ++target_lane) {
        const int target_index = __shfl_sync(kFullMask, value_index, target_lane);
        const float target_adjoint = __shfl_sync(
            kFullMask, output_adjoint, target_lane
        );
        const size_t edge = weight_base + target_lane * kWarpSize + lane;

        float grad_weight = 0.0f;
        if (target_index >= 0 && source_valid && edge_mask[edge]) {
            grad_weight = target_adjoint * source_value;
            grad_source += target_adjoint * weights[edge];
        }
        grad_weights[edge] = grad_weight;
    }

    if (source_valid) {
        grad_t_values[value_index] = grad_source;
    }
}

void check_inputs(
        const torch::Tensor& t_values,
        const torch::Tensor& slot_to_nnz,
        const torch::Tensor& weights,
        const torch::Tensor& edge_mask) {
    TORCH_CHECK(
        t_values.is_cuda() && slot_to_nnz.is_cuda()
            && weights.is_cuda() && edge_mask.is_cuda(),
        "all inputs must be CUDA tensors"
    );
    TORCH_CHECK(t_values.scalar_type() == torch::kFloat32, "T_values must be float32");
    TORCH_CHECK(slot_to_nnz.scalar_type() == torch::kInt64, "slot_to_nnz must be int64");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(edge_mask.scalar_type() == torch::kBool, "edge_mask must be bool");
    TORCH_CHECK(t_values.dim() == 1, "T_values must have shape [NNZ]");
    TORCH_CHECK(
        slot_to_nnz.dim() == 2 && slot_to_nnz.size(1) == kWarpSize,
        "slot_to_nnz must have shape [G, 32]"
    );
    TORCH_CHECK(
        weights.dim() == 3 && weights.size(0) == slot_to_nnz.size(0)
            && weights.size(1) == kWarpSize && weights.size(2) == kWarpSize,
        "W must have shape [G, 32, 32]"
    );
    TORCH_CHECK(edge_mask.sizes() == weights.sizes(), "edge_mask must match W");
    TORCH_CHECK(
        t_values.device() == slot_to_nnz.device()
            && t_values.device() == weights.device()
            && t_values.device() == edge_mask.device(),
        "all inputs must be on the same CUDA device"
    );
    TORCH_CHECK(t_values.is_contiguous(), "T_values must be contiguous");
    TORCH_CHECK(slot_to_nnz.is_contiguous(), "slot_to_nnz must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(edge_mask.is_contiguous(), "edge_mask must be contiguous");
    TORCH_CHECK(t_values.numel() <= INT_MAX, "NNZ exceeds the int32 kernel index range");
    TORCH_CHECK(slot_to_nnz.size(0) <= INT_MAX, "G exceeds the int32 kernel index range");
}

}  // namespace

torch::Tensor warp_sparse_forward(
        torch::Tensor t_values,
        torch::Tensor slot_to_nnz,
        torch::Tensor weights,
        torch::Tensor edge_mask) {
    check_inputs(t_values, slot_to_nnz, weights, edge_mask);
    const c10::cuda::CUDAGuard device_guard(t_values.device());
    const int groups = static_cast<int>(slot_to_nnz.size(0));
    auto r_values = torch::empty_like(t_values);

    const int blocks = (groups + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto stream = at::cuda::getCurrentCUDAStream();
    warp_sparse_forward_kernel<<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(
        t_values.data_ptr<float>(),
        slot_to_nnz.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        edge_mask.data_ptr<bool>(),
        r_values.data_ptr<float>(),
        groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return r_values;
}

std::vector<torch::Tensor> warp_sparse_backward(
        torch::Tensor t_values,
        torch::Tensor slot_to_nnz,
        torch::Tensor weights,
        torch::Tensor edge_mask,
        torch::Tensor r_values,
        torch::Tensor grad_r_values) {
    check_inputs(t_values, slot_to_nnz, weights, edge_mask);
    TORCH_CHECK(r_values.is_cuda() && grad_r_values.is_cuda(), "backward tensors must be CUDA");
    TORCH_CHECK(
        r_values.scalar_type() == torch::kFloat32
            && grad_r_values.scalar_type() == torch::kFloat32,
        "backward tensors must be float32"
    );
    TORCH_CHECK(
        r_values.sizes() == t_values.sizes()
            && grad_r_values.sizes() == t_values.sizes(),
        "backward tensors must have shape [NNZ]"
    );
    TORCH_CHECK(
        r_values.device() == t_values.device()
            && grad_r_values.device() == t_values.device(),
        "backward tensors must be on the same CUDA device as T_values"
    );
    const c10::cuda::CUDAGuard device_guard(t_values.device());
    r_values = r_values.contiguous();
    grad_r_values = grad_r_values.contiguous();

    auto grad_t_values = torch::empty_like(t_values);
    auto grad_weights = torch::empty_like(weights);
    const int groups = static_cast<int>(slot_to_nnz.size(0));
    const int blocks = (groups + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto stream = at::cuda::getCurrentCUDAStream();
    warp_sparse_backward_kernel<<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(
        t_values.data_ptr<float>(),
        slot_to_nnz.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        edge_mask.data_ptr<bool>(),
        r_values.data_ptr<float>(),
        grad_r_values.data_ptr<float>(),
        grad_t_values.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_t_values, grad_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &warp_sparse_forward, "Warp sparse contrast forward (CUDA)");
    module.def("backward", &warp_sparse_backward, "Warp sparse contrast backward (CUDA)");
}
