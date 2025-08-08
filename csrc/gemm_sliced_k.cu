#include "gemm_sliced_k.h"
#include "cute/tensor.hpp"

template<typename T, int BLOCK_M, int BLOCK_N, int TileK, typename MMA>
__global__ void gemm_sliced_k_kernel(
    const T* Aptr,
    const T* Bptr,
    T* Cptr,
    int M,
    int N,
    int K
    )
{
    cute::Tensor A = cute::make_tensor(cute::make_gmem_ptr(Aptr), cute::make_shape(M, K), cute::make_stride(K, cute::Int<1>{}));
    cute::Tensor B = cute::make_tensor(cute::make_gmem_ptr(Bptr), cute::make_shape(N, K), cute::make_stride(K, cute::Int<1>{}));
    cute::Tensor C = cute::make_tensor(cute::make_gmem_ptr(Cptr), cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    
    cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<BLOCK_M>{}, cute::Int<TileK>{}), cute::make_coord(ix, cute::_));    // [BLOCK_M, TileK, num_tile_k]
    cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<BLOCK_N>{}, cute::Int<TileK>{}), cute::make_coord(iy, cute::_));    // [BLOCK_N, TileK, num_tile_k]
    cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<BLOCK_M>{}, cute::Int<BLOCK_N>{}), cute::make_coord(ix, iy));       // [BLOCK_M, BLOCK_N]

    MMA mma;

    auto thr_mma = mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);    // [MMA, MMA_M, MMA_K, num_tile_k]
    auto tBgB = thr_mma.partition_B(gB);    // [MMA, MMA_N, MMA_K, num_tile_k]
    auto tCgC = thr_mma.partition_C(gC);    // [MMA, MMA_M, MMA_N]

    auto tArA = thr_mma.partition_fragment_A(gA(cute::_, cute::_, 0));    // [MMA, MMA_M, MMA_K]
    auto tBrB = thr_mma.partition_fragment_B(gB(cute::_, cute::_, 0));    // [MMA, MMA_N, MMA_K]
    auto tCrC = thr_mma.partition_fragment_C(gC(cute::_, cute::_));       // [MMA, MMA_M, MMA_N]

    cute::clear(tCrC);  // clear the accumulator, set all elements to 0
    
    int num_tile_k = cute::size<2>(gA);

    for (int i_tile = 0; i_tile < num_tile_k; ++i_tile) {
        cute::copy(tAgA(cute::_, cute::_, cute::_, i_tile), tArA);
        cute::copy(tBgB(cute::_, cute::_, cute::_, i_tile), tBrB);

        cute::gemm(mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}


// sliced-k strategy
// one block deal with one TileM * TileN sub-matrix in C
// suitable for large M N, relatively small K
torch::Tensor gemm_sliced_k(
    const torch::Tensor& A, // [M, K] activation
    const torch::Tensor& B  // [N, K] weight
    )
{   
    
    TORCH_CHECK(A.is_cuda(), "A must be on the GPU");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have compatible dimensions");

    torch::Tensor A_ = A.is_contiguous() ? A : A.contiguous();
    torch::Tensor B_ = B.is_contiguous() ? B : B.contiguous();

    int M = A_.size(0);
    int N = B_.size(0);
    int K = A_.size(1);

    torch::Tensor C = torch::zeros({M, N}, A_.options()); // [M, N] output

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int TILE_K = 128;

    TORCH_CHECK(M % BLOCK_M == 0, "M must be divisible by BLOCK_M");
    TORCH_CHECK(N % BLOCK_N == 0, "N must be divisible by BLOCK_N");
    TORCH_CHECK(K % TILE_K == 0, "K must be divisible by TILE_K");

    dim3 grid(M / BLOCK_M, N / BLOCK_N);

    // m16n8k16
    // F32 BF16 BF16 F32 -> D A B C
    using mma_op = cute::SM80_16x8x16_F32BF16BF16F32_TN;    // T stands for transpose, N stands for normal(col-major)
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    using MMA = decltype(cute::make_tiled_mma(mma_atom{},
            cute::make_layout(cute::Shape<cute::_1, cute::_1, cute::_1>{}))); // thr_layout_mnk,
                                                                                                // Represents the expansion along the MNK dimensions,
                                                                                                // manifested as an increase in the number of threads(on Ampere, means warps) within a block.
    
    dim3 block(cute::size(MMA{}));  // on Ampere(SM80), mma is a warp-level instruction, defualt size is 32 if no increasing.
    
    gemm_sliced_k_kernel<__nv_bfloat16, BLOCK_M, BLOCK_N, TILE_K, MMA>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    (reinterpret_cast<const __nv_bfloat16*>(A_.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(B_.data_ptr()), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K);

    return C;
}