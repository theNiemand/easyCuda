import unittest
import torch
from easyCuda.kernel import gemm_sliced_k

class gemmSlicedKTest(unittest.TestCase):
    def setUp(self):
        self.M = [1024, 1024 * 16]
        self.N = [1024, 1024 * 16]
        self.K = [1024, 1024 * 16]
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda:0')

    def test_accuracy(self):
        for M in self.M:
            for N in self.N:
                for K in self.K:
                    with self.subTest(M=M, N=N, K=K):
                        A = torch.randn(M, K, dtype=self.dtype, device=self.device)
                        B = torch.randn(N, K, dtype=self.dtype, device=self.device)

                        C_pred = gemm_sliced_k(A, B)
                        C_real = torch.matmul(A, B.T).to(torch.bfloat16)
                        
                        self.assertTrue(
                            torch.allclose(C_pred, C_real, atol=1e-2, rtol=1e-2),
                            "accuracy test failed, expected {}, got {}".format(C_real, C_pred)
                        )


if __name__ == "__main__":
    unittest.main()