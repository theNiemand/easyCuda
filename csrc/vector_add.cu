#include "vector_add.h"

template<int BLK, int TPB, int VPT>
__global__ void vector_add_kernel(
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    __nv_bfloat16* Z,
    const int32_t num_ele)
{
    const int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    register __nv_bfloat16 x_local[VPT];
    register __nv_bfloat16 y_local[VPT];
    register __nv_bfloat16 z_local[VPT];

    for (int32_t offset = tid * VPT; offset < num_ele; offset += BLK * TPB * VPT)
    {
        // load x and y g2s, vectorize
        *reinterpret_cast<float4*>(x_local) = *reinterpret_cast<const float4*>(X + offset);
        *reinterpret_cast<float4*>(y_local) = *reinterpret_cast<const float4*>(Y + offset);

        // x + y, elementwise
        #pragma unroll
        for (int32_t i = 0; i < VPT; i++)
        {
            z_local[i] = __float2bfloat16(
                __bfloat162float(x_local[i]) +
                __bfloat162float(y_local[i]));
        }

        // store z s2g, vectorize
        *reinterpret_cast<float4*>(Z + offset) = *reinterpret_cast<float4*>(z_local);
    }
}


torch::Tensor vector_add(const torch::Tensor& x, const torch::Tensor& y)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");

    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same size");

    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must be the same type");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16 type");

    const torch::Tensor x_contiguous = x.is_contiguous() ? x : x.contiguous();
    const torch::Tensor y_contiguous = y.is_contiguous() ? y : y.contiguous();
    torch::Tensor z = torch::empty_like(x_contiguous);

    const int32_t num_ele = x_contiguous.numel();
    constexpr int VPT = 8;
    constexpr int TPB = 128;
    constexpr int BLK = 256;

    TORCH_CHECK(num_ele % VPT == 0, "x must be divisible by ", VPT);

    vector_add_kernel<BLK, TPB, VPT>
    <<<TPB, BLK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(z.data_ptr()),
        num_ele);

    return z;
}