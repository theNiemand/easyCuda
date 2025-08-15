import unittest
import torch
from easyCuda.cuda_kernel import gemm_sliced_k_sm80_optim
from utils.basic import snr_error, benchmark

class gemmSlicedKTest(unittest.TestCase):
    def setUp(self):
        self.M = [512, 1024, 4 * 1024, 16 * 1024]
        self.N = [512, 1024, 4 * 1024, 16 * 1024]
        self.K = [512, 1024, 4 * 1024, 16 * 1024]
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda:0')

    def test_accuracy(self):
        for M in self.M:
            for N in self.N:
                for K in self.K:
                    with self.subTest(M=M, N=N, K=K):
                        A = torch.randn(M, K, dtype=self.dtype, device=self.device)
                        B = torch.randn(N, K, dtype=self.dtype, device=self.device)

                        C_pred = gemm_sliced_k_sm80_optim(A, B)
                        C_real = torch.matmul(A, B.T).to(torch.bfloat16)

                        print(f"{snr_error(C_pred, C_real) = }")
                        
                        self.assertTrue(
                            snr_error(C_pred, C_real) < 1e-2,
                            "accuracy test failed, got {}, expected {}".format(C_pred, C_real)
                        )


    def test_performance(self):
        for M in self.M:
            for N in self.N:
                for K in self.K:
                    with self.subTest(M=M, N=N, K=K):
                        A = torch.randn(M, K, dtype=self.dtype, device=self.device)
                        B = torch.randn(N, K, dtype=self.dtype, device=self.device)

                        shape = [M, N, K]
                        flop = 2 * M * N * K

                        benchmark(gemm_sliced_k_sm80_optim, shape, flop, A, B)
                        benchmark(torch.matmul, shape, flop, A, B.T)


if __name__ == "__main__":
    unittest.main()
