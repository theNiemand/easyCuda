#include "ops.h"
#include "cute/tensor.hpp"


template<class ElementA,
        class ElementB,
        class ASmemLayout,
        class BSmemLayout>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<ASmemLayout>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<BSmemLayout>> B;
};


template<class ProblemShape, class CTATiler,
        class ElementA, class AStride, class ASmemLayout, class G2SCopyA, class S2RCopyAtomA,
        class ElementB, class BStride, class BSmemLayout, class G2SCopyB, class S2RCopyAtomB,
        class ElementC, class CStride, class TiledMMA>
__global__
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void gemm_sliced_k_sm80_optim_kernel(
  ProblemShape shapeMNK, CTATiler cta_tiler,
  const ElementA* __restrict__ TA, AStride dA, ASmemLayout sA_layout, G2SCopyA g2s_copy_a, S2RCopyAtomA s2r_copy_atom_a,
  const ElementB* __restrict__ TB, BStride dB, BSmemLayout sB_layout, G2SCopyB g2s_copy_b, S2RCopyAtomB s2r_copy_atom_b,
  ElementC* __restrict__ TC, CStride dC, TiledMMA mmaC
) {
  using namespace cute;

  // Full Tensor
  Tensor A = make_tensor(make_gmem_ptr(TA), select<0,2>(shapeMNK), dA);
  Tensor B = make_tensor(make_gmem_ptr(TB), select<1,2>(shapeMNK), dB);
  Tensor C = make_tensor(make_gmem_ptr(TC), select<0,1>(shapeMNK), dC);

  // gmem tensor, this cta
  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(A, cta_tiler, coord, Step<_1, X, _1>{});   // [TileM, TileK, num_tiles_k]
  Tensor gB = local_tile(B, cta_tiler, coord, Step<X, _1, _1>{});   // [TileN, TileK, num_tiles_k]
  Tensor gC = local_tile(C, cta_tiler, coord, Step<_1, _1, X>{});   // [TileM, TileN]

  // "alloc" shared memory
  extern __shared__ char shared_memory[];
  using _SharedStorage =  SharedStorage<ElementA, ElementB, ASmemLayout, BSmemLayout>;
  _SharedStorage& smem = *reinterpret_cast<_SharedStorage*>(shared_memory);

  // smem tensor
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);  // [TileM, TileK, stages]
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);  // [TileN, TileK, stages]

  // alloc register memory across threads
  ThrMMA thr_mma = mmaC.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));  // [MMA, MMA_M, MMA_K]
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));  // [MMA, MMA_N, MMA_K]
  Tensor tCrC = thr_mma.partition_fragment_C(gC);           // [MMA, MMA_M, MMA_N]
  
  // g2s memory copy partition across threads
  ThrCopy thr_g2s_copy_a = g2s_copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s_copy_a.partition_S(gA);             // [CPY, CPY_M, CPY_K, num_tiles_k]
  Tensor tAsA = thr_g2s_copy_a.partition_D(sA);             // [CPY, CPY_M, CPY_K, stages]

  ThrCopy thr_g2s_copy_b = g2s_copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_g2s_copy_b.partition_S(gB);             // [CPY, CPY_N, CPY_K, num_tiles_k]
  Tensor tBsB = thr_g2s_copy_b.partition_D(sB);             // [CPY, CPY_N, CPY_K, stages]

  // s2r memory copy partition.(smem partition acrosss threads, rmem retile)
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_copy_atom_a, mmaC);
  ThrCopy thr_s2r_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_copy_a.partition_S(sA);             // [CPY, MMA_M, MMA_K, stages]
  Tensor tXrA = thr_s2r_copy_a.retile_D(tCrA);              // [CPY, MMA_M, MMA_K]

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_copy_atom_b, mmaC);
  ThrCopy thr_s2r_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_copy_b.partition_S(sB);             // [CPY, MMA_N, MMA_K, stages]
  Tensor tXrB = thr_s2r_copy_b.retile_D(tCrB);              // [CPY, MMA_N, MMA_K]

  // r2g partition for C.(gmem partition)
  Tensor tCgC = thr_mma.partition_C(gC);                    // [MMA, MMA_M, MMA_N]

  // pre-fetch all stages but the last
  int num_tiles_k =  size<3>(tAgA);
  int tile_id = 0;
  auto STAGES = size<3>(tAsA);

  CUTE_UNROLL
  for (int stage_id = 0; stage_id < STAGES - 1; ++stage_id) {
    copy(g2s_copy_a, tAgA(_, _, _, tile_id), tAsA(_, _, _, stage_id));
    copy(g2s_copy_b, tBgB(_, _, _, tile_id), tBsB(_, _, _, stage_id));
    cp_async_fence();

    --num_tiles_k;
    if (num_tiles_k > 0) {
      ++tile_id;
    }
  }

  // pre-fetch only first micro_stage
  int stage_read_id = 0;
  int stage_write_id = STAGES - 1;
  auto MICRO_STAGES = size<2>(tCrA);

  Tensor tXsA_p = tXsA(_,_,_,stage_read_id);
  Tensor tXsB_p = tXsB(_,_,_,stage_read_id);

  // just need one stage arrived
  if (MICRO_STAGES > 1) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();    // neccessary ?

    copy(s2r_copy_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_, _, _0{}));
    copy(s2r_copy_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_, _, _0{}));
    // no need for "commit-wait" like instrution in ldmatrix and mma async ? 
  }

  clear(tCrC);

  // main loop
  CUTE_NO_UNROLL
  while (num_tiles_k > -(STAGES - 1)) {
    CUTE_UNROLL
    for (int micro_stage_id = 0; micro_stage_id < MICRO_STAGES; ++micro_stage_id) {
      if (micro_stage_id == MICRO_STAGES - 1) {
         // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,stage_read_id);
        tXsB_p = tXsB(_,_,_,stage_read_id);

        // wait for next stage arrived
        cp_async_wait<STAGES - 2>();
        __syncthreads();
      }

      // load next micro stage
      auto micro_stage_id_next = (micro_stage_id + Int<1>{}) % MICRO_STAGES;      // static
      copy(s2r_copy_atom_a, tXsA_p(_,_,micro_stage_id_next), tXrA(_,_,micro_stage_id_next));
      copy(s2r_copy_atom_b, tXsB_p(_,_,micro_stage_id_next), tXrB(_,_,micro_stage_id_next));

      if (micro_stage_id == 0)
      {
        // commit next stage
        copy(g2s_copy_a, tAgA(_,_,_,tile_id), tAsA(_,_,_,stage_write_id));
        copy(g2s_copy_b, tBgB(_,_,_,tile_id), tBsB(_,_,_,stage_write_id));
        cp_async_fence();

        --num_tiles_k;
        if (num_tiles_k > 0) { ++tile_id; }

        stage_write_id = stage_read_id;
        stage_read_id = (stage_read_id == STAGES-1) ? 0 : stage_read_id + 1;
      }
      
      // compute on current micro stage
      gemm(mmaC, tCrA(_,_,micro_stage_id), tCrB(_,_,micro_stage_id), tCrC);
    }
  }

  copy(tCrC, tCgC);
}


torch::Tensor gemm_sliced_k_sm80_optim(
  const torch::Tensor &A,   // [M, K]
  const torch::Tensor &B    // [N, K]
) {  
  // device check
  TORCH_CHECK(A.is_cuda(), "A must be on the GPU");
  TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

  // shape check
  TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
  TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
  TORCH_CHECK(A.size(1) == B.size(1), "A and B must have compatible dimensions");

  // type check
  TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be a bfloat16 tensor");
  TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be a bfloat16 tensor");

  // contiguous guarantee
  torch::Tensor A_contiguous = A.is_contiguous() ? A : A.contiguous();
  torch::Tensor B_contiguous = B.is_contiguous() ? B : B.contiguous();

  // alloc C tensor
  torch::Tensor C = torch::zeros({A.size(0), B.size(0)}, A.options());

  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);

  using namespace cute;

  // tile problem shape
  auto TileM = _128{};
  auto TileN = _128{};
  auto TileK = _64{};
  auto stages = _3{};

  // check shape can fit in tile
  TORCH_CHECK(M % TileM == 0, "M must be divisible by TileM");
  TORCH_CHECK(N % TileN == 0, "N must be divisible by TileN");
  TORCH_CHECK(K % TileK == 0, "K must be divisible by TileK");

  auto shapeMNK = make_shape(M, N, K);
  auto cta_tiler = make_shape(TileM, TileN, TileK);

  using ElementA = __nv_bfloat16;
  using ElementB = __nv_bfloat16;
  using ElementC = __nv_bfloat16;

  auto dA = make_stride(K, _1{});
  auto dB = make_stride(K, _1{});
  auto dC = make_stride(N, _1{});

  auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                  Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  
  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(TileM, TileK, stages));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(TileN, TileK, stages));

  auto g2s_copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, __nv_bfloat16>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _8>>{});
  auto g2s_copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, __nv_bfloat16>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _8>>{});
  
  auto s2r_copy_atom_a = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>{};
  auto s2r_copy_atom_b = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>{};

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F32BF16BF16F32_TN{},
                                Layout<Shape<_2, _2>>{},
                                Tile<_32, _32, _16>{});
  
  auto kernel_fptr = gemm_sliced_k_sm80_optim_kernel<decltype(shapeMNK), decltype(cta_tiler),
                                                    ElementA, decltype(dA), decltype(sA_layout), decltype(g2s_copy_a), decltype(s2r_copy_atom_a),
                                                    ElementB, decltype(dB), decltype(sB_layout), decltype(g2s_copy_b), decltype(s2r_copy_atom_b),
                                                    ElementC, decltype(dC), decltype(mmaC)>;
  
  int smem_size = int(sizeof(SharedStorage<ElementA, ElementB, decltype(sA_layout), decltype(sB_layout)>));   // try static alloc
  cudaStream_t stream = 0;
  
  dim3 grid(M / TileM, N / TileN);
  dim3 block(size(mmaC));

  // "cudaFuncAttributeMaxDynamicSharedMemorySize"
  // - The requested maximum size in bytes of dynamically-allocated shared memory.
  // The sum of this value and the function attribute sharedSizeBytes cannot exceed the device attribute "cudaDevAttrMaxSharedMemoryPerBlockOptin".
  // The maximal size of requestable dynamic shared memory may differ by GPU architecture.
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  // "cudaFuncAttributePreferredSharedMemoryCarveout"
  // - On devices where the L1 cache and shared memory use the same hardware resources
  // this sets the shared memory carveout preference, in percent of the total shared memory. See cudaDevAttrMaxSharedMemoryPerMultiprocessor.
  // This is only a hint, and the driver can choose a different ratio if required to execute the function.
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  
  // Returns the maximum shared memory (SMEM) allocation allowed per block,  
  // already considered partition limits.
  // int device_id = 0;
  // int carveout_max = 0;
  // cudaDeviceGetAttribute(&carveout_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
  // printf("Max supported carveout: %d bytes\n", carveout_max);

  
  kernel_fptr<<<grid, block, smem_size, stream>>>(shapeMNK, cta_tiler,
                              reinterpret_cast<const __nv_bfloat16*>(A_contiguous.data_ptr()), dA, sA_layout, g2s_copy_a, s2r_copy_atom_a,
                              reinterpret_cast<const __nv_bfloat16*>(B_contiguous.data_ptr()), dB, sB_layout, g2s_copy_b, s2r_copy_atom_b,
                              reinterpret_cast<__nv_bfloat16*>(C.data_ptr()), dC, mmaC);

  // 错误检查
  // cudaError_t err = cudaGetLastError();

  // if (err != cudaSuccess) {
  //   std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
  // }

  // cudaDeviceSynchronize();

  // err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
  // }
 
  return C;
}