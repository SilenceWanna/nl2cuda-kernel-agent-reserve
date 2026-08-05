#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kRowsPerBlock = 32;
constexpr int kScanThreads = 16;
constexpr int kColumnThreads = 64;

// This is the same 32-element shared-memory scan structure used by PyTorch's
// innermost cumsum kernel, which keeps fp32 accumulation order comparable.
template <bool Suffix>
__global__ void innermost_scan_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int rows,
        int cols) {
    __shared__ float buffers[kRowsPerBlock][2 * kScanThreads];
    float* row_buffer = buffers[threadIdx.y];
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    float block_total = 0.0f;

    for (int block_col = 0; block_col < cols; block_col += 2 * kScanThreads) {
        const int logical_col1 = block_col + threadIdx.x;
        const int logical_col2 = block_col + kScanThreads + threadIdx.x;
        const int col1 = Suffix ? cols - 1 - logical_col1 : logical_col1;
        const int col2 = Suffix ? cols - 1 - logical_col2 : logical_col2;

        if (row < rows) {
            row_buffer[threadIdx.x] =
                    logical_col1 < cols
                    ? input[static_cast<size_t>(row) * cols + col1]
                    : 0.0f;
            row_buffer[kScanThreads + threadIdx.x] =
                    logical_col2 < cols
                    ? input[static_cast<size_t>(row) * cols + col2]
                    : 0.0f;
        }
        __syncthreads();

        if (row < rows && threadIdx.x == 0) {
            row_buffer[0] = row_buffer[0] + block_total;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int active = kScanThreads, distance = 1;
             active >= 1;
             active >>= 1, distance <<= 1) {
            if (row < rows && threadIdx.x < active) {
                const unsigned int offset =
                        (2 * threadIdx.x + 1) * distance - 1;
                row_buffer[offset + distance] =
                        row_buffer[offset] + row_buffer[offset + distance];
            }
            __syncthreads();
        }

        #pragma unroll
        for (unsigned int active = 2, distance = kScanThreads / 2;
             distance >= 1;
             active <<= 1, distance >>= 1) {
            if (row < rows && threadIdx.x < active - 1) {
                const unsigned int offset =
                        2 * (threadIdx.x + 1) * distance - 1;
                row_buffer[offset + distance] =
                        row_buffer[offset] + row_buffer[offset + distance];
            }
            __syncthreads();
        }

        if (row < rows) {
            if (logical_col1 < cols) {
                output[static_cast<size_t>(row) * cols + col1] =
                        row_buffer[threadIdx.x];
            }
            if (logical_col2 < cols) {
                output[static_cast<size_t>(row) * cols + col2] =
                        row_buffer[kScanThreads + threadIdx.x];
            }
            block_total = row_buffer[2 * kScanThreads - 1];
        }
        __syncthreads();
    }
}

template <bool Suffix>
__global__ void column_scan_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int rows,
        int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }

    float sum = 0.0f;
    for (int logical_row = 0; logical_row < rows; ++logical_row) {
        const int row = Suffix ? rows - 1 - logical_row : logical_row;
        const size_t index = static_cast<size_t>(row) * cols + col;
        sum += input[index];
        output[index] = sum;
    }
}

template <bool Suffix>
void launch_innermost_scan(
        const torch::Tensor& input,
        torch::Tensor& output,
        int rows,
        int cols,
        cudaStream_t stream) {
    const dim3 threads(kScanThreads, kRowsPerBlock);
    const dim3 blocks((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    innermost_scan_kernel<Suffix><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
}

template <bool Suffix>
void launch_column_scan(
        const torch::Tensor& input,
        torch::Tensor& output,
        int rows,
        int cols,
        cudaStream_t stream) {
    const dim3 threads(kColumnThreads);
    const dim3 blocks((cols + kColumnThreads - 1) / kColumnThreads);
    column_scan_kernel<Suffix><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
}

void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype float32");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0,
                "input dimensions must be positive");
    TORCH_CHECK(input.size(0) <= INT32_MAX && input.size(1) <= INT32_MAX,
                "input dimensions are too large");
}

torch::Tensor integral_image_forward(torch::Tensor input) {
    check_input(input);
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    auto row_prefix = torch::empty_like(input);
    auto output = torch::empty_like(input);
    cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(input.device().index()).stream();

    launch_innermost_scan<false>(input, row_prefix, rows, cols, stream);
    launch_column_scan<false>(row_prefix, output, rows, cols, stream);
    return output;
}

torch::Tensor integral_image_backward(torch::Tensor grad_output) {
    check_input(grad_output);
    const int rows = static_cast<int>(grad_output.size(0));
    const int cols = static_cast<int>(grad_output.size(1));
    auto row_suffix = torch::empty_like(grad_output);
    auto output = torch::empty_like(grad_output);
    cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(grad_output.device().index()).stream();

    launch_column_scan<true>(grad_output, row_suffix, rows, cols, stream);
    launch_innermost_scan<true>(row_suffix, output, rows, cols, stream);
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integral_image_forward", &integral_image_forward,
          "2D integral image forward (CUDA)");
    m.def("integral_image_backward", &integral_image_backward,
          "2D integral image backward (CUDA)");
}
